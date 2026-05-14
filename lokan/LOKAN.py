import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

import sympy
from sympy import *
from sympy.printing import latex

from .LOKANLayer import LOKANLayer
from .Symbolic_KANLayer import Symbolic_KANLayer

from .spline import curve2coef
from .utils import SYMBOLIC_LIB
from .nexus_plot import plot_nexus_network


class LOKAN(nn.Module):
    """
    Learnable Operations KAN (LOKAN).

    A modification of MultKAN where the discrete sum/product topology is
    replaced by a *soft, learnable* logit layer inside every KANLayer.
    Each edge (i→j) carries a B-spline activation plus a probability
    distribution over operation groups (summation or multiplicative),
    parameterised by ``logits`` and decoded via softmax.

    The model supports:
    - Numerical (spline) and symbolic front-ends per edge.
    - Backward attribution scoring.
    - Symbolic regression via ``suggest_symbolic`` / ``auto_symbolic`` /
      ``symbolic_formula``.
    - Visualization via ``plot`` (delegates to ``nexus_plot.py``).
    """

    def __init__(self,
                 width=None, grid=3, k=3,
                 noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0,
                 base_fun='silu',
                 symbolic_enabled=True, affine_trainable=False,
                 grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True,
                 seed=1,
                 save_act=True, sparse_init=False,
                 device='cpu'):
        """
        Initialize a LOKAN model.

        Parameters
        ----------
        width : list of int or list of [int, int]
            Network width per layer.  Each entry is either a plain ``int``
            ``n`` (expanded to ``[n, 0]``) or a ``[n_sum, n_mult]`` pair.
            Since the learnable logit layer replaces explicit mult nodes,
            the recommended form is a plain list of ints.
        grid : int or list of int
            Number of B-spline grid intervals; scalar or one value per layer.
        k : int or list of int
            B-spline polynomial order; scalar or one value per layer.
        noise_scale : float
            Amplitude of noise added to initial spline coefficients.
        scale_base_mu : float
            Mean of the Normal initializer for the base-function scale.
        scale_base_sigma : float
            Std of the Normal initializer for the base-function scale.
        base_fun : str
            Residual activation before the spline.
            One of ``'silu'``, ``'identity'``, ``'zero'``.
        symbolic_enabled : bool
            Enable the symbolic front-end alongside the numerical splines.
        affine_trainable : bool
            Make per-node affine (scale/bias) parameters trainable.
        grid_eps : float
            Blend between adaptive and uniform grid update
            (0 = fully adaptive, 1 = fully uniform).
        grid_range : list of two float
            Initial uniform grid range ``[lo, hi]``.
        sp_trainable : bool
            Make spline scale parameters trainable.
        sb_trainable : bool
            Make base-function scale parameters trainable.
        seed : int
            Random seed for reproducibility.
        save_act : bool
            Cache intermediate activations during ``forward`` (required for
            visualization and attribution).
        sparse_init : bool
            Initialize the spline mask with a sparse nearest-neighbour pattern.
        device : str
            PyTorch device string (``'cpu'`` or ``'cuda'``).
        """
        super(LOKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.act_fun = []
        self.depth   = len(width) - 1

        for i in range(len(width)):
            if type(width[i]) == int or type(width[i]) == np.int64:
                width[i] = [width[i], 0]

        self.width    = width
        width_in  = self.width_in
        width_out = self.width_out

        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x * 0.

        self.grid_eps   = grid_eps
        self.grid_range = grid_range

        for l in range(self.depth):
            grid_l = grid[l] if isinstance(grid, list) else grid
            k_l    = k[l]    if isinstance(k,    list) else k

            sp_batch = LOKANLayer(
                in_dim=width_in[l], out_dim=width_out[l + 1],
                num=grid_l, k=k_l,
                noise_scale=noise_scale,
                scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma,
                scale_sp=1., base_fun=base_fun,
                grid_eps=grid_eps, grid_range=grid_range,
                sp_trainable=sp_trainable, sb_trainable=sb_trainable,
                sparse_init=sparse_init,
            )
            self.act_fun.append(sp_batch)

        self.node_bias    = []
        self.node_scale   = []
        self.subnode_bias  = []
        self.subnode_scale = []

        for l in range(self.depth):
            exec(f'self.node_bias_{l}    = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l}   = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l}  = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')

        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid     = grid
        self.k        = k
        self.base_fun = base_fun

        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l + 1])
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun     = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable     = sp_trainable
        self.sb_trainable     = sb_trainable

        self.save_act = save_act

        self.node_scores    = None
        self.edge_scores    = None
        self.subnode_scores = None

        self.cache_data = None
        self.acts       = None

        self.device = device
        self.to(device)

        self.input_id = torch.arange(self.width_in[0],)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device):
        """
        Move the model and all sub-modules to *device*.

        Parameters
        ----------
        device : str or torch.device
            Target device (e.g. ``'cpu'``, ``'cuda'``).

        Returns
        -------
        LOKAN
            ``self`` (for chaining).
        """
        super(LOKAN, self).to(device)
        self.device = device
        for kanlayer in self.act_fun:
            kanlayer.to(device)
        for symbolic_kanlayer in self.symbolic_fun:
            symbolic_kanlayer.to(device)
        return self

    # ------------------------------------------------------------------
    # Width properties
    # ------------------------------------------------------------------

    @property
    def width_in(self):
        """
        Number of input nodes at each layer.

        Returns
        -------
        list of int
            Length ``depth + 1``; entry ``l`` is ``width[l][0] + width[l][1]``.
        """
        return [self.width[l][0] + self.width[l][1] for l in range(len(self.width))]

    @property
    def width_out(self):
        """
        Number of output sub-nodes (spline outputs) at each layer.

        Returns
        -------
        list of int
            Length ``depth + 1``; entry ``l`` is ``width[l][0]``.
        """
        return [self.width[l][0] for l in range(len(self.width))]

    @property
    def n_sum(self):
        """
        Number of addition nodes at each hidden layer.

        Returns
        -------
        list of int
            Length ``depth - 1`` (hidden layers only).
        """
        return [self.width[l][0] for l in range(1, len(self.width) - 1)]

    @property
    def n_mult(self):
        """
        Number of explicit multiplication nodes at each hidden layer.

        In LOKAN this is always 0 since multiplication is handled by the
        learnable logit layer; the property is kept for API compatibility.

        Returns
        -------
        list of int
            Length ``depth - 1`` (hidden layers only).
        """
        return [self.width[l][1] for l in range(1, len(self.width) - 1)]

    @property
    def feature_score(self):
        """
        Attribution scores for each input feature.

        Calls ``attribute()`` internally; requires a prior forward pass.

        Returns
        -------
        torch.Tensor or None
            1-D tensor of length ``width_in[0]``, or ``None`` if activations
            have not been cached.
        """
        self.attribute()
        return None if self.node_scores is None else self.node_scores[0]

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def update_grid_from_samples(self, x):
        """
        Refine the B-spline grids using the empirical distribution of *x*.

        Performs a forward pass to cache layer inputs, then updates each
        layer's grid so that knots are placed at data quantiles blended with
        a uniform grid (blend ratio controlled by ``grid_eps``).

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(batch, width_in[0])``.

        Example
        -------
        >>> model = LOKAN(width=[2, 5, 1], grid=5, k=3, seed=0)
        >>> x = torch.linspace(-2, 2, 200)[:, None].expand(-1, 2)
        >>> model.update_grid_from_samples(x)
        """
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def update_grid(self, x):
        """
        Alias for ``update_grid_from_samples``.

        Kept so that external training loops that call ``update_grid``
        continue to work without modification.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(batch, width_in[0])``.
        """
        self.update_grid_from_samples(x)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        """
        Run a forward pass through the network.

        Activations, pre/post-spline values, and scale statistics are cached
        on ``self`` when ``save_act=True``.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(batch, width_in[0])``.
        singularity_avoiding : bool
            When ``True``, the symbolic branch uses singularity-safe function
            variants (e.g. soft ``1/x`` near zero).
        y_th : float
            Singularity threshold passed to the symbolic safe functions.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, width_in[-1])``.

        Example
        -------
        >>> model = LOKAN(width=[2, 5, 1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100, 2)
        >>> model(x).shape
        torch.Size([100, 1])
        """
        x = x[:, self.input_id.long()]
        assert x.shape[1] == self.width_in[0]

        self.cache_data = x

        self.acts               = []
        self.acts_premult       = []
        self.spline_preacts     = []
        self.spline_postsplines = []
        self.spline_postacts    = []
        self.acts_scale         = []
        self.acts_scale_spline  = []
        self.subnode_actscale   = []
        self.edge_actscale      = []

        self.acts.append(x)

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            if self.symbolic_enabled:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](
                    x, singularity_avoiding=singularity_avoiding, y_th=y_th
                )
            else:
                x_symbolic        = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic

            if self.save_act:
                self.subnode_actscale.append(torch.std(x, dim=0).detach())

            x = self.subnode_scale[l][None, :] * x + self.subnode_bias[l][None, :]

            if self.save_act:
                postacts             = postacts_numerical + postacts_symbolic
                input_range          = torch.std(preacts, dim=0) + 0.1
                output_range_spline  = torch.std(postacts_numerical, dim=0)
                output_range         = torch.std(postacts, dim=0)

                self.edge_actscale.append(output_range)
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())
                self.acts_premult.append(x.detach())

            x = self.node_scale[l][None, :] * x + self.node_bias[l][None, :]
            self.acts.append(x.detach())

        return x

    # ------------------------------------------------------------------
    # Activation caching helper
    # ------------------------------------------------------------------

    def get_act(self, x=None):
        """
        Collect intermediate activations without permanently changing ``save_act``.

        If *x* is ``None``, the last cached input (``self.cache_data``) is
        used.  If *x* is a dataset dict, ``x['train_input']`` is used.

        Parameters
        ----------
        x : torch.Tensor, dict, or None
            Input batch or dataset dictionary.

        Raises
        ------
        Exception
            If no input is available (no argument and no cached data).

        Example
        -------
        >>> model = LOKAN(width=[2, 3, 1], grid=5, k=3, seed=0)
        >>> x = torch.rand(50, 2)
        >>> model.get_act(x)
        >>> len(model.acts)  # depth + 1 = 3
        3
        """
        if isinstance(x, dict):
            x = x['train_input']
        if x is None:
            if self.cache_data is not None:
                x = self.cache_data
            else:
                raise Exception("missing input data x")
        old_save_act = self.save_act
        self.save_act = True
        self.forward(x)
        self.save_act = old_save_act

    # ------------------------------------------------------------------
    # Mode switching (numerical / symbolic)
    # ------------------------------------------------------------------

    def set_mode(self, l, i, j, mode, mask_n=None):
        """
        Set the computation mode for the (l, i→j) edge.

        Parameters
        ----------
        l : int
            Layer index.
        i : int
            Input neuron index.
        j : int
            Output neuron index.
        mode : str
            - ``'n'``  — numerical (spline) only.
            - ``'s'``  — symbolic only.
            - ``'sn'`` or ``'ns'`` — both active simultaneously.
            - any other value — dead edge (both disabled).
        mask_n : float or None
            Custom numerical mask weight for ``'sn'`` mode.
        """
        if mode == "s":
            mask_n = 0.;  mask_s = 1.
        elif mode == "n":
            mask_n = 1.;  mask_s = 0.
        elif mode in ("sn", "ns"):
            mask_n = 1. if mask_n is None else mask_n;  mask_s = 1.
        else:
            mask_n = 0.;  mask_s = 0.

        self.act_fun[l].mask.data[i][j]      = mask_n
        self.symbolic_fun[l].mask.data[j, i] = mask_s

    # ------------------------------------------------------------------
    # Symbolic edge management
    # ------------------------------------------------------------------

    def fix_symbolic(self, l, i, j, fun_name,
                     fit_params_bool=True,
                     a_range=(-10, 10), b_range=(-10, 10),
                     verbose=True, random=False):
        """
        Fix the (l, i→j) activation to a symbolic function.

        Switches the edge to symbolic mode (spline disabled) and optionally
        fits the affine parameters ``a, b, c, d`` so that
        ``c * fun(a*x + b) + d`` best matches the cached spline output.

        Parameters
        ----------
        l : int
            Layer index.
        i : int
            Input neuron index.
        j : int
            Output neuron index.
        fun_name : str
            Key in ``SYMBOLIC_LIB`` (e.g. ``'sin'``, ``'x^2'``, ``'1/x'``).
        fit_params_bool : bool
            If ``True``, fit ``a, b, c, d`` from cached activations.
            If ``False``, use the identity initialisation ``[1, 0, 1, 0]``.
        a_range : tuple of float
            Search range for ``a`` during fitting.
        b_range : tuple of float
            Search range for ``b`` during fitting.
        verbose : bool
            Print fitting details.
        random : bool
            Initialise affine params randomly instead of as identity.

        Returns
        -------
        float or None
            R² coefficient of determination (when ``fit_params_bool=True``),
            or ``None`` otherwise.

        Example
        -------
        >>> model = LOKAN(width=[2, 5, 1], grid=5, k=3, noise_scale=1., seed=0)
        >>> x = torch.normal(0, 1, size=(100, 2))
        >>> model(x)
        >>> model.fix_symbolic(0, 1, 3, 'sin')
        """
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            r2 = None
        else:
            x    = self.acts[l][:, i]
            mask = self.act_fun[l].mask
            y    = self.spline_postacts[l][:, j, i]
            r2   = self.symbolic_fun[l].fix_symbolic(
                i, j, fun_name, x, y,
                a_range=a_range, b_range=b_range, verbose=verbose,
            )
            if mask[i, j] == 0:
                r2 = -1e8

        self.set_mode(l, i, j, mode="s")
        return r2

    def unfix_symbolic(self, l, i, j):
        """
        Revert the (l, i→j) edge from symbolic back to numerical (spline) mode.

        The symbolic function name is reset to ``'0'`` and the numerical mask
        is re-enabled.

        Parameters
        ----------
        l : int
            Layer index.
        i : int
            Input neuron index.
        j : int
            Output neuron index.
        """
        self.set_mode(l, i, j, mode="n")
        self.symbolic_fun[l].funs_name[j][i] = "0"

    def unfix_symbolic_all(self):
        """
        Revert every edge in the network to numerical (spline) mode.
        """
        for l in range(len(self.width) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    self.unfix_symbolic(l, i, j)

    # ------------------------------------------------------------------
    # Activation range inspection
    # ------------------------------------------------------------------

    def get_range(self, l, i, j, verbose=True):
        """
        Return the input/output range of the (l, i→j) spline activation.

        Requires a prior forward pass with ``save_act=True``.

        Parameters
        ----------
        l : int
            Layer index.
        i : int
            Input neuron index.
        j : int
            Output neuron index.
        verbose : bool
            If ``True``, print the range values.

        Returns
        -------
        x_min, x_max, y_min, y_max : float
            Scalar numpy floats describing the activation input/output range.

        Example
        -------
        >>> model = LOKAN(width=[2, 3, 1], grid=5, k=3, noise_scale=1., seed=0)
        >>> x = torch.normal(0, 1, size=(100, 2))
        >>> model(x)
        >>> model.get_range(0, 0, 0)
        """
        x = self.spline_preacts[l][:, j, i]
        y = self.spline_postacts[l][:, j, i]
        x_min = torch.min(x).cpu().detach().numpy()
        x_max = torch.max(x).cpu().detach().numpy()
        y_min = torch.min(y).cpu().detach().numpy()
        y_max = torch.max(y).cpu().detach().numpy()
        if verbose:
            print('x range: [' + '%.2f' % x_min, ',', '%.2f' % x_max, ']')
            print('y range: [' + '%.2f' % y_min, ',', '%.2f' % y_max, ']')
        return x_min, x_max, y_min, y_max

    # ------------------------------------------------------------------
    # Attribution (backward importance scores)
    # ------------------------------------------------------------------

    def attribute(self, l=None, i=None, out_score=None, plot=True):
        """
        Compute backward attribution scores for nodes and edges.

        Scores are propagated from the output layer backward through the
        network using cached activation-scale statistics.  Results are stored
        on ``self.node_scores``, ``self.edge_scores``, and
        ``self.subnode_scores``.

        Parameters
        ----------
        l : int or None
            If given, compute scores up to layer *l* only.
        i : int or None
            If given together with *l*, return the attribution of output
            neuron *i* in layer *l* back to the inputs and optionally plot
            a bar chart.
        out_score : torch.Tensor or None
            1-D tensor of output weights.  ``None`` uses the identity matrix
            (uniform output weights).
        plot : bool
            When both *l* and *i* are given, draw a bar chart of input
            attributions.

        Returns
        -------
        torch.Tensor or None
            Attribution tensor when *l* is specified; ``None`` otherwise.

        Example
        -------
        >>> model = LOKAN(width=[3, 5, 1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> x = torch.rand(200, 3)
        >>> model(x)
        >>> model.attribute()
        >>> model.feature_score
        """
        if l is not None:
            self.attribute()
            out_score = self.node_scores[l]

        if self.acts is None:
            self.get_act()

        def score_node2subnode(node_score, width):
            return node_score[:, :width[0]]

        node_scores    = []
        subnode_scores = []
        edge_scores    = []

        l_query = l
        l_end   = l if l is not None else self.depth

        out_dim    = self.width_in[l_end]
        node_score = (
            torch.eye(out_dim).requires_grad_(True)
            if out_score is None
            else torch.diag(out_score).requires_grad_(True)
        )
        node_scores.append(node_score)

        device = self.act_fun[0].grid.device

        for l in range(l_end, 0, -1):
            subnode_score = score_node2subnode(node_score, self.width[l])
            subnode_scores.append(subnode_score)

            edge_score = torch.einsum(
                'ij,ki,i->kij',
                self.edge_actscale[l - 1],
                subnode_score.to(device),
                1 / (self.subnode_actscale[l - 1] + 1e-4),
            )
            edge_scores.append(edge_score)
            node_score = torch.sum(edge_score, dim=1)
            node_scores.append(node_score)

        self.node_scores_all    = list(reversed(node_scores))
        self.edge_scores_all    = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores    = [torch.mean(ls, dim=0) for ls in self.node_scores_all]
        self.edge_scores    = [torch.mean(ls, dim=0) for ls in self.edge_scores_all]
        self.subnode_scores = [torch.mean(ls, dim=0) for ls in self.subnode_scores_all]

        if l_query is not None:
            if i is None:
                return self.node_scores_all[0]
            if plot:
                in_dim = self.width_in[0]
                plt.figure(figsize=(in_dim, 3))
                plt.bar(range(in_dim), self.node_scores_all[0][i].cpu().detach().numpy())
                plt.xticks(range(in_dim))
            return self.node_scores_all[0][i]

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def get_reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef=0., lamb_coefdiff=0.):
        """
        Compute a sparsity regularization term for use in custom training loops.

        Encourages sparse activation patterns (few active edges per node) and,
        optionally, small or smooth spline coefficients.

        Parameters
        ----------
        reg_metric : str
            Which activation-scale statistic to regularize.  One of:

            - ``'edge_forward_spline_n'`` — normalized spline output std.
            - ``'edge_forward_sum'``      — combined (spline + symbolic) std.
            - ``'edge_forward_spline_u'`` — unnormalized spline output std.
            - ``'edge_backward'``         — backward attribution edge scores.
        lamb_l1 : float
            Weight for the L1 (mean activation magnitude) term.
        lamb_entropy : float
            Weight for the row/column entropy term (encourages sparsity).
        lamb_coef : float
            Weight for L1 penalty on spline coefficients.
        lamb_coefdiff : float
            Weight for L1 penalty on spline coefficient differences
            (encourages smooth splines).

        Returns
        -------
        torch.Tensor
            Scalar differentiable regularization loss.

        Example
        -------
        >>> model = LOKAN(width=[2, 3, 1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100, 2)
        >>> model(x)
        >>> reg = model.get_reg('edge_forward_spline_n', 1.0, 2.0)
        """
        metric_map = {
            'edge_forward_spline_n': self.acts_scale_spline,
            'edge_forward_sum':      self.acts_scale,
            'edge_forward_spline_u': self.edge_actscale,
            'edge_backward':         self.edge_scores,
        }
        if reg_metric not in metric_map:
            raise ValueError(f'reg_metric={reg_metric!r} not recognized; '
                             f'choose from {list(metric_map)}')
        acts_scale = metric_map[reg_metric]

        reg_ = 0.
        for vec in acts_scale:
            l1      = torch.sum(vec)
            p_row   = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
            p_col   = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
            ent_row = -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
            ent_col = -torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
            reg_   += lamb_l1 * l1 + lamb_entropy * (ent_row + ent_col)

        for layer in self.act_fun:
            coeff_l1      = torch.sum(torch.mean(torch.abs(layer.coef), dim=1))
            coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(layer.coef)), dim=1))
            reg_         += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    def disable_symbolic_in_fit(self, lamb):
        """
        Temporarily disable the symbolic branch to speed up training steps.

        Returns the old values of ``save_act`` and ``symbolic_enabled`` so
        the caller can restore them after the loss computation.

        The symbolic branch is disabled if:
        - ``lamb == 0`` (no regularization requires activation caching), or
        - no symbolic masks are currently active.

        Parameters
        ----------
        lamb : float
            Regularization strength used in the current training step.

        Returns
        -------
        old_save_act : bool
        old_symbolic_enabled : bool

        Example
        -------
        >>> old_act, old_sym = model.disable_symbolic_in_fit(lamb=0.)
        >>> loss = criterion(model(x), y)
        >>> model.save_act, model.symbolic_enabled = old_act, old_sym
        """
        old_save_act = self.save_act
        if lamb == 0.:
            self.save_act = False

        no_symbolic = all(
            torch.sum(torch.abs(self.symbolic_fun[l].mask)) == 0
            for l in range(len(self.symbolic_fun))
        )
        old_symbolic_enabled = self.symbolic_enabled
        if no_symbolic:
            self.symbolic_enabled = False

        return old_save_act, old_symbolic_enabled

    # ------------------------------------------------------------------
    # Symbolic regression
    # ------------------------------------------------------------------

    def suggest_symbolic(self, l, i, j,
                         a_range=(-10, 10), b_range=(-10, 10),
                         lib=None, topk=5, verbose=True,
                         r2_loss_fun=lambda x: np.log2(1 + 1e-5 - x),
                         c_loss_fun=lambda x: x,
                         weight_simple=0.8):
        """
        Suggest the best-fitting symbolic function for edge (l, i→j).

        Iterates over candidate functions, fits the affine parameters
        ``a, b, c, d`` for ``c * f(a*x + b) + d``, and ranks results by a
        weighted combination of fitting quality (R²) and symbolic complexity.

        Parameters
        ----------
        l : int
            Layer index.
        i : int
            Input neuron index.
        j : int
            Output neuron index.
        a_range : tuple of float
            Search range for the scale parameter ``a``.
        b_range : tuple of float
            Search range for the shift parameter ``b``.
        lib : list of str or None
            Subset of ``SYMBOLIC_LIB`` keys to search.  ``None`` = full library.
        topk : int
            Number of top candidates to display when ``verbose=True``.
        verbose : bool
            Print a ranked table of candidates.
        r2_loss_fun : callable
            ``r2 -> loss`` mapping (default penalises low R² logarithmically).
        c_loss_fun : callable
            ``complexity -> loss`` mapping (default is identity).
        weight_simple : float
            Weight on complexity loss vs. R² loss.
            1.0 = purely prefer simplest function; 0.0 = purely best fit.

        Returns
        -------
        best_name : str
            Name of the top-ranked function.
        best_fun : tuple
            Corresponding entry from ``SYMBOLIC_LIB``.
        best_r2 : float
            R² of the top function.
        best_c : int
            Complexity score of the top function.

        Example
        -------
        >>> model = LOKAN(width=[2, 1, 1], grid=5, k=3, seed=0)
        >>> x = torch.rand(200, 2)
        >>> model(x)
        >>> model.suggest_symbolic(0, 1, 0)
        """
        symbolic_lib = {k: SYMBOLIC_LIB[k] for k in lib} if lib is not None else SYMBOLIC_LIB

        r2s, cs = [], []
        for name, content in symbolic_lib.items():
            r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False)
            if r2 == -1e8:
                r2s.append(-1e8)
            else:
                r2s.append(r2.item())
                self.unfix_symbolic(l, i, j)
            cs.append(content[2])

        r2s      = np.array(r2s)
        cs       = np.array(cs)
        r2_loss  = r2_loss_fun(r2s).astype('float')
        cs_loss  = c_loss_fun(cs)
        loss     = weight_simple * cs_loss + (1 - weight_simple) * r2_loss
        sorted_ids = np.argsort(loss)[:topk]

        if verbose:
            results = {
                'function':        [list(symbolic_lib)[k] for k in sorted_ids],
                'fitting r2':      r2s[sorted_ids],
                'r2 loss':         r2_loss[sorted_ids],
                'complexity':      cs[sorted_ids],
                'complexity loss': cs_loss[sorted_ids],
                'total loss':      loss[sorted_ids],
            }
            print(pd.DataFrame(results))

        best_name = list(symbolic_lib)[sorted_ids[0]]
        best_fun  = list(symbolic_lib.values())[sorted_ids[0]]
        best_r2   = float(r2s[sorted_ids[0]])
        best_c    = int(cs[sorted_ids[0]])
        return best_name, best_fun, best_r2, best_c

    def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10),
                      lib=None, verbose=1, weight_simple=0.8, r2_threshold=0.0):
        """
        Automatically assign symbolic functions to every active edge.

        For each edge the best candidate from ``suggest_symbolic`` is applied
        via ``fix_symbolic`` if its R² exceeds *r2_threshold*.  Already
        symbolic edges are skipped; dead edges are set to the zero function.

        Parameters
        ----------
        a_range : tuple of float
            Search range for ``a``.
        b_range : tuple of float
            Search range for ``b``.
        lib : list of str or None
            Function library subset.  ``None`` = full ``SYMBOLIC_LIB``.
        verbose : int
            Verbosity level (0 = silent, 1 = one line per edge, 2 = full).
        weight_simple : float
            Simplicity bias passed to ``suggest_symbolic``.
        r2_threshold : float
            Minimum R² required to fix an edge symbolically.

        Example
        -------
        >>> model = LOKAN(width=[2, 1, 1], grid=5, k=3, seed=0)
        >>> x = torch.rand(200, 2)
        >>> model(x)
        >>> model.auto_symbolic()
        """
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    sym_active = self.symbolic_fun[l].mask[j, i] > 0.
                    num_active = self.act_fun[l].mask[i][j] > 0.

                    if sym_active and not num_active:
                        print(f'skipping ({l},{i},{j}) — already symbolic')
                    elif not sym_active and not num_active:
                        self.fix_symbolic(l, i, j, '0', verbose=verbose > 1)
                        print(f'fixing ({l},{i},{j}) with 0')
                    else:
                        name, _, r2, c = self.suggest_symbolic(
                            l, i, j, a_range=a_range, b_range=b_range,
                            lib=lib, verbose=False, weight_simple=weight_simple,
                        )
                        if r2 >= r2_threshold:
                            self.fix_symbolic(l, i, j, name, verbose=verbose > 1)
                            if verbose >= 1:
                                print(f'fixing ({l},{i},{j}) with {name}, r2={r2:.4f}, c={c}')
                        else:
                            print(f'({l},{i},{j}): best={name}, r2={r2:.4f} < {r2_threshold} — skipped')

    def symbolic_formula(self, var=None, normalizer=None, output_normalizer=None, simplify=False):
        """
        Extract the closed-form symbolic expression of the network.

        Requires that all active edges have been fixed to symbolic functions
        (via ``fix_symbolic`` or ``auto_symbolic``).

        The learnable operation logits affect *numerical* computation only;
        this method reads the *symbolic front-end* (``Symbolic_KANLayer``)
        and returns a pure sympy expression.

        Parameters
        ----------
        var : list or None
            Input variable names or sympy ``Symbol`` objects.
            ``None`` → auto-named ``x_1, x_2, ...``.
        normalizer : [means, stds] or None
            If provided, inputs are first normalised as ``(x - mean) / std``.
        output_normalizer : [means, stds] or None
            If provided, outputs are de-normalised as ``y * std + mean``.
        simplify : bool
            Apply ``sympy.simplify`` to each output expression.

        Returns
        -------
        exprs : list of sympy.Expr
            One expression per output node.
        x0 : list of sympy.Symbol
            The raw (un-normalised) input symbols.

        Raises
        ------
        RuntimeError
            If any active edge has not been converted to a symbolic function.

        Example
        -------
        >>> model = LOKAN(width=[2, 1, 1], grid=5, k=3, seed=0)
        >>> x = torch.rand(200, 2)
        >>> model(x)
        >>> model.auto_symbolic()
        >>> exprs, syms = model.symbolic_formula()
        >>> exprs[0]
        """
        symbolic_acts = []
        x = []

        if var is None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(v) for v in var]

        x0 = x

        if normalizer is not None:
            mean, std = normalizer
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            y = []
            for j in range(self.width_out[l + 1]):
                yj = sympy.Integer(0)
                for i in range(self.width_in[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun  = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj = yj + c * sympy_fun(a * x[i] + b) + d
                    except Exception:
                        raise RuntimeError(
                            f'Edge ({l},{i},{j}) is not symbolic. '
                            'Run auto_symbolic() or fix_symbolic() first.'
                        )
                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                y.append(sympy.simplify(yj) if simplify else yj)

            for j in range(self.width_in[l + 1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]

            x = y
            symbolic_acts.append(x)

        if output_normalizer is not None:
            means, stds = output_normalizer
            out_layer   = symbolic_acts[-1]
            assert len(out_layer) == len(means), 'output_normalizer size mismatch'
            symbolic_acts[-1] = [out_layer[i] * stds[i] + means[i] for i in range(len(out_layer))]

        self.symbolic_acts = symbolic_acts
        return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, folder="./figures", beta=3, metric='backward', scale=0.5, tick=False,
             sample=False, in_vars=None, out_vars=None, title=None, varscale=1.0,
             tau_decode=0.05, save_path=None):
        """
        Draw the LOKAN network diagram with logit-colored edges.

        Edges are colored by the dominant operation decoded from
        ``softmax(logits / tau_decode)``:

        - Blue   → summation path (last logit slot).
        - Orange → multiplicative group 0.
        - Green  → multiplicative group 1.
        - Red    → multiplicative group 2.

        Spline activation curves are embedded as small insets on each edge.
        Output neurons are annotated with Σ or Π based on a majority vote
        over their incoming edges.

        Requires a prior forward pass with ``save_act=True`` so that
        activation statistics are available for edge opacity.

        Parameters
        ----------
        folder : str
            Unused; kept for API compatibility with MultKAN.
        beta : float
            ``tanh`` sharpness for edge alpha transparency from activation
            scale scores.
        metric : str
            Edge importance metric used for opacity:
            ``'forward_n'``, ``'forward_u'``, or ``'backward'``.
            ``'backward'`` calls ``attribute()`` internally.
        scale : float
            Overall figure size multiplier.
        tick : bool
            Show tick marks on spline inset axes.
        sample : bool
            Scatter data points on spline insets.
        in_vars : list or None
            Input variable labels (strings or sympy expressions).
        out_vars : list or None
            Output variable labels (strings or sympy expressions).
        title : str or None
            Figure title.
        varscale : float
            Font size multiplier for variable labels.
        tau_decode : float
            Softmax temperature for operation decoding (lower = sharper).
        save_path : str or None
            If set, save the figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        return plot_nexus_network(
            self, folder=folder, beta=beta, metric=metric,
            scale=scale, tick=tick, sample=sample,
            in_vars=in_vars, out_vars=out_vars, title=title,
            varscale=varscale, tau_decode=tau_decode, save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_params(self):
        """
        Return an iterator over all trainable model parameters.

        Returns
        -------
        iterator
            Same as ``self.parameters()``.
        """
        return self.parameters()


loKAN = LOKAN
