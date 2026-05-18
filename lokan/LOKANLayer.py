import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .spline import *
from .utils import sparse_mask

import math
from itertools import combinations


class LOKANLayer(nn.Module):
    """
    Single LOKAN layer: B-spline activations + learnable operation logits.

    For every input–output pair (i, j) there is a B-spline curve
    ``φ(i→j)`` parameterised by ``coef[i, j, :]``.  The scalar outputs of
    all input splines feeding into output neuron *j* are then combined via a
    soft mixture of operations decoded from ``logits[i, j, :]``:

    - The last logit slot corresponds to the **summation** path.
    - All other slots correspond to **multiplicative** groups.
    """

    def __init__(self,
                 in_dim=3,
                 out_dim=2,
                 num=5,
                 k=3,
                 noise_scale=0.5,
                 scale_base_mu=0.0,
                 scale_base_sigma=1.0,
                 scale_sp=1.0,
                 base_fun=torch.nn.SiLU(),
                 grid_eps=0.02,
                 grid_range=[-1, 1],
                 sp_trainable=True,
                 sb_trainable=True,
                 save_plot_data=True,
                 device='cpu',
                 sparse_init=False):
        """
        Initialize a LOKANLayer.

        Parameters
        ----------
        in_dim : int
            Number of input neurons.
        out_dim : int
            Number of output neurons.
        num : int
            Number of B-spline grid intervals (``G`` in the grid shape).
        k : int
            B-spline polynomial order.
        noise_scale : float
            Amplitude of the random noise added to initial spline coefficients.
        scale_base_mu : float
            Mean of the Normal initializer for the base-function scale
            ``scale_base``.
        scale_base_sigma : float
            Std of the Normal initializer for ``scale_base``.
        scale_sp : float
            Initial value of the spline scale ``scale_sp`` (before masking).
        base_fun : callable
            Residual activation applied element-wise to the raw input before
            the B-spline evaluation (e.g. ``torch.nn.SiLU()``).
        grid_eps : float
            Blend between adaptive and uniform grid on update
            (0 = fully adaptive, 1 = fully uniform).
        grid_range : list of two float
            Initial grid extent ``[lo, hi]``.
        sp_trainable : bool
            Allow ``scale_sp`` to be updated by the optimizer.
        sb_trainable : bool
            Allow ``scale_base`` to be updated by the optimizer.
        save_plot_data : bool
            Unused flag kept for API compatibility.
        device : str
            PyTorch device string.
        sparse_init : bool
            If ``True``, initialize the connectivity mask with a sparse
            nearest-neighbor pattern instead of full connectivity.
        """
        super(LOKANLayer, self).__init__()

        self.out_dim = out_dim
        self.in_dim  = in_dim
        self.num     = num
        self.k       = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)

        noises    = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        self.scale_base = torch.nn.Parameter(
            scale_base_mu * 1 / np.sqrt(in_dim) +
            scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(in_dim)
        ).requires_grad_(sb_trainable)

        self.scale_sp = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask
        ).requires_grad_(sp_trainable)

        self.base_fun = base_fun
        self.grid_eps = grid_eps

        # Learnable operation logits: shape (in_dim, out_dim, K+1)
        # Last slot = summation; slots 0..K-1 = multiplicative groups.
        # K = in_dim // 2 + 1  multiplicative groups by default.
        self.tau    = 1.0
        self.logits = torch.nn.Parameter(torch.rand(in_dim, out_dim, in_dim // 2 + 1)).requires_grad_(True)

        self.to(device)

    def to(self, device):
        """
        Move this layer to *device*.

        Parameters
        ----------
        device : str or torch.device
            Target device.

        Returns
        -------
        LOKANLayer
            ``self`` (for chaining).
        """
        super(LOKANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        """
        Compute the layer output for input batch *x*.

        The computation proceeds in three stages:

        1. **Spline evaluation** — for each (i, j) pair evaluate the B-spline
           plus the residual ``scale_base * base_fun(x_i)``.
        2. **Operation mixing** — decode ``logits`` via softmax to obtain
           per-edge probabilities for the sum path and each multiplicative
           group, then combine the spline outputs accordingly.
        3. **Output aggregation** — sum the sum-path contribution and all
           multiplicative group products across input dimension *i*.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_dim)``.

        Returns
        -------
        y : torch.Tensor
            Output of shape ``(batch, out_dim)``.
        preacts : torch.Tensor
            Pre-activation values broadcast to shape
            ``(batch, out_dim, in_dim)`` — the raw inputs replicated for
            each output neuron.
        postacts : torch.Tensor
            Post-spline activations of shape ``(batch, out_dim, in_dim)``.
        postspline : torch.Tensor
            Raw spline outputs (before base-function blend) of shape
            ``(batch, out_dim, in_dim)``.

        Example
        -------
        >>> layer = LOKANLayer(in_dim=4, out_dim=3, num=5, k=3)
        >>> x = torch.rand(32, 4)
        >>> y, preacts, postacts, postspline = layer(x)
        >>> y.shape
        torch.Size([32, 3])
        """
        batch = x.shape[0]

        # Broadcast raw input for caching: (batch, out_dim, in_dim)
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x)                                              # (batch, in_dim)
        y    = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)  # (batch, in_dim, out_dim)

        postspline = y.clone().permute(0, 2, 1)                              # (batch, out_dim, in_dim)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y                                        # (batch, in_dim, out_dim)

        postacts = y.clone().permute(0, 2, 1)                                # (batch, out_dim, in_dim)

        # Decode operation probabilities
        probs      = torch.softmax(self.logits / self.tau, dim=-1)           # (in_dim, out_dim, K+1)
        prod_probs = probs[:, :, :-1]                                         # (in_dim, out_dim, K)
        sum_probs  = probs[:, :, -1]                                          # (in_dim, out_dim)

        prod_probs = prod_probs[None, :, :, :]                               # (1, in_dim, out_dim, K)
        sum_probs  = sum_probs[None, :, :]                                   # (1, in_dim, out_dim)

        # Multiplicative groups: soft product via (1 - p + p*phi)
        term   = 1 - prod_probs + prod_probs * y.unsqueeze(-1)              # (batch, in_dim, out_dim, K)
        y_prod = torch.prod(term, dim=1)                                      # (batch, out_dim, K)

        # Summation path
        y_sum  = torch.sum(y * sum_probs, dim=1)                             # (batch, out_dim)

        # beta_g = torch.prod(1 - prod_probs, dim=1) # (1, out_dim, K)
        # y = y_sum + torch.sum(y_prod - beta_g, dim=2) # (batch, out_dim)
        
        w_g = 1 - torch.prod(1 - prod_probs, dim=1) # (1, out_dim, K)
        y = y_sum + torch.sum(w_g * y_prod, dim=2)
        
        # y = y_sum + torch.sum(y_prod, dim=2) # (batch, out_dim)v

        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'):
        """
        Refine the B-spline grid to match the empirical distribution of *x*.

        The new grid is a blend between a sample-quantile adaptive grid and a
        uniform grid, controlled by ``grid_eps``.  Spline coefficients are
        re-fitted to the existing spline curve on the new grid.

        Parameters
        ----------
        x : torch.Tensor
            Layer input of shape ``(batch, in_dim)``.
        mode : str
            ``'sample'`` uses the actual data quantiles; ``'grid'`` uses a
            denser intermediate grid for a smoother re-fit.

        Example
        -------
        >>> layer = LOKANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(layer.grid.data[:, 3:6])
        >>> x = torch.linspace(-3, 3, 100)[:, None].expand(-1, 2)
        >>> layer.update_grid_from_samples(x)
        >>> print(layer.grid.data[:, 3:6])
        """
        batch        = x.shape[0]
        x_pos        = torch.sort(x, dim=0)[0]
        y_eval       = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids           = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            margin        = 0.00
            h             = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) / num_interval
            grid_uniform  = grid_adaptive[:, [0]] - margin + h * torch.arange(num_interval + 1)[None, :].to(x.device)
            return self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos       = sample_grid.permute(1, 0)
            y_eval      = coef2curve(x_pos, self.grid, self.coef, self.k)

        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)
