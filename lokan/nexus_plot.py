"""
LOKAN-specific visualization engine.

Replaces the MultKAN-derived plot() with four functions that expose
LOKAN's *learned* operation structure (logit layer) rather than
hard-coded sum/mult topology.

Public API
----------
plot_nexus_network(model, ...)         – full network diagram:
                                          spline curves inline on edges,
                                          edges colored by dominant operation,
                                          Σ/Π annotations at output neurons.

plot_ops_heatmap(model, ...)           – per-layer P(sum) heatmap derived from
                                          softmax(logits / tau_decode).

plot_activation_gallery(model, ...)    – grid of all learned B-spline curves
                                          evaluated on a dense probe grid.

plot_kan_fc_architecture(kan, fc, ...) – unified flow diagram for KAN + FC
                                          hybrid models: per-cell spline
                                          thumbnails on a P(Σ) background for
                                          every KAN layer, then weight-magnitude
                                          heatmaps for every FC Linear layer,
                                          joined by a bridge annotation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from .spline import coef2curve

# ---------------------------------------------------------------------------
# Operation color palette
# ---------------------------------------------------------------------------

OP_COLORS = {
    'sum': '#4878d0',   # blue — summation path (last logit slot)
}
# Mult group colors: tab10 starting from index 1 (skip blue, reserved for sum)
_MULT_PALETTE = [plt.colormaps['tab10'].colors[i] for i in range(1, 10)]


def _op_color(g: int, K: int) -> str:
    """Return hex color for operation slot g (K = number of mult groups)."""
    if g == K:
        return OP_COLORS['sum']
    c = _MULT_PALETTE[g % len(_MULT_PALETTE)]
    return '#%02x%02x%02x' % tuple(int(255 * x) for x in c)


# ---------------------------------------------------------------------------
# Helper: decode logit operation assignment for one layer
# ---------------------------------------------------------------------------

def _decode_ops(layer, tau_decode: float = 0.05):
    """
    Returns:
        dominant : (in_dim, out_dim) int  — argmax operation index per edge
        probs    : (in_dim, out_dim, K+1) float32 — full soft probability
        K        : int  — number of multiplicative groups
    """
    with torch.no_grad():
        probs    = torch.softmax(layer.logits / tau_decode, dim=-1)
        dominant = probs.argmax(dim=-1)
    K = layer.logits.shape[-1] - 1
    return dominant.cpu(), probs.cpu(), K


# ---------------------------------------------------------------------------
# Function 1: full network diagram
# ---------------------------------------------------------------------------

def plot_nexus_network(
    model,
    fc_module           = None,
    folder:      str   = './figures',
    beta:        float = 3.0,
    metric:      str   = 'forward_n',
    scale:       float = 0.5,
    tick:        bool  = False,
    sample:      bool  = False,
    in_vars             = None,
    out_vars            = None,
    title:       str   = None,
    varscale:    float = 1.0,
    tau_decode:  float = 0.05,
    save_path:   str   = None,
):
    """
    Draw the NexusKAN network with logit-colored edges and inline spline plots.

    KAN edges are colored by the dominant operation assignment decoded from
    softmax(logits / tau_decode):
      blue   → summation path
      orange → multiplicative group 0
      green  → multiplicative group 1
      red    → multiplicative group 2

    Each KAN output neuron is annotated with Σ or Π.

    When fc_module is given the FC linear layers are drawn above the KAN
    stack as a dense bipartite graph.  Each FC edge is drawn in gray with
    opacity and line-width proportional to the absolute weight magnitude;
    output neurons of each FC layer are drawn as black dots.  A small
    "FC l (n_in→n_out)" label appears to the right of each FC layer.

    Parameters
    ----------
    model      : NexusKAN instance (must have run a forward pass first)
    fc_module  : optional nn.Module whose child modules include nn.Linear
                 layers (e.g. nn.Sequential); drawn above the KAN stack
    folder     : unused — kept for API compatibility with MultKAN.plot()
    beta       : tanh sharpness for alpha transparency
    metric     : 'forward_n' | 'forward_u' | 'backward'
    scale      : overall figure size multiplier
    tick       : show tick marks on spline insets
    sample     : also scatter data points on spline insets
    in_vars    : list of input variable names / sympy expressions
    out_vars   : list of output variable names / sympy expressions
    title      : figure title
    varscale   : font size multiplier for variable labels
    tau_decode : softmax temperature for decoding operations (low = sharp)
    save_path  : if set, save figure to this path
    """
    import sympy
    from sympy.printing import latex
    import torch.nn as nn

    if not model.save_act:
        print('plot_nexus_network: save_act=False — run model(x) with save_act=True first.')
        return None

    if model.acts is None or len(model.acts) == 0:
        if model.cache_data is None:
            raise RuntimeError('Model has not seen any data. Run model(x) first.')
        model.forward(model.cache_data)

    if metric == 'backward':
        model.attribute()

    # ---- Select edge-importance scores ----
    if metric == 'forward_n':
        scores = model.acts_scale
    elif metric == 'forward_u':
        scores = model.edge_actscale
    elif metric == 'backward':
        scores = model.edge_scores
    else:
        raise ValueError(f'metric={metric!r} not recognized; use forward_n/forward_u/backward')

    if not scores:
        print('plot_nexus_network: no activation scores available. Run model(x) first.')
        return None

    # ---- Collect FC layers ----
    fc_linears = (
        [m for m in fc_module.modules() if isinstance(m, nn.Linear)]
        if fc_module is not None else []
    )
    n_fc = len(fc_linears)

    def score2alpha(s):
        return np.tanh(beta * s)

    alpha = [score2alpha(sc.cpu().detach().numpy()) for sc in scores]

    # ---- Layout ----
    depth        = len(model.width) - 1
    width_in     = np.array(model.width_in)
    width_out    = np.array(model.width_out)

    A            = 1.0
    y0           = 0.3    # vertical span: input → post-spline
    z0           = 0.1    # vertical span: post-spline → next input
    neuron_depth = len(model.width)
    total_depth  = neuron_depth - 1 + n_fc   # total levels of (y0+z0)

    min_spacing  = A / max(int(np.max(width_out)), 5)
    max_neuron   = int(np.max(width_out))
    max_weights  = int(np.max(width_in[:-1] * width_out[1:]))
    y1 = 0.4 / max(max_weights, 5)    # spline inset half-size (data coords)
    y2 = 0.15 / max(max_neuron, 5)    # node annotation size

    fig, ax = plt.subplots(
        figsize=(10 * scale, 10 * scale * total_depth * (y0 + z0))
    )

    DC_to_FC  = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda xy: FC_to_NFC(DC_to_FC(xy))

    # ---- Phase A: collect spline curve data for each edge ----
    spline_data = {}   # (l, i, j) -> (x_arr, y_arr, spline_color, alpha_mask)
    for l in range(depth):
        for i in range(width_in[l]):
            for j in range(width_out[l + 1]):
                rank  = torch.argsort(model.acts[l][:, i])
                x_arr = model.acts[l][:, i][rank].cpu().detach().numpy()
                y_arr = model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy()

                sym_mask = float(model.symbolic_fun[l].mask[j][i])
                num_mask = float(model.act_fun[l].mask[i][j])
                if sym_mask > 0 and num_mask > 0:
                    spline_color, amask = 'purple', 1.0
                elif sym_mask > 0 and num_mask == 0:
                    spline_color, amask = 'red',    1.0
                elif sym_mask == 0 and num_mask > 0:
                    spline_color, amask = 'black',  1.0
                else:
                    spline_color, amask = 'white',  0.0

                spline_data[(l, i, j)] = (x_arr, y_arr, spline_color, amask)

    # ---- Phase B: draw KAN network skeleton with logit-colored edges ----
    for l in range(neuron_depth):
        n = width_in[l]

        # Input neuron scatter
        for i in range(n):
            ax.scatter(
                1 / (2 * n) + i / n,
                l * (y0 + z0),
                s     = min_spacing ** 2 * 10000 * scale ** 2,
                color = 'black',
                zorder = 3,
            )

        # Edges: input i → output j (with logit-based color)
        if l < neuron_depth - 1:
            n_next    = width_out[l + 1]
            N         = n * n_next
            kan_layer = model.act_fun[l]
            K         = kan_layer.logits.shape[-1] - 1

            with torch.no_grad():
                probs_layer = torch.softmax(
                    kan_layer.logits / tau_decode, dim=-1
                )  # (in_dim, out_dim, K+1)
                dominant_layer = probs_layer.argmax(dim=-1).cpu()  # (in_dim, out_dim)

            for i in range(n):
                for j in range(n_next):
                    id_  = i * n_next + j
                    g    = int(dominant_layer[i, j].item())
                    ecol = _op_color(g, K)
                    _, _, _, amask = spline_data[(l, i, j)]
                    a    = float(alpha[l][j][i]) * amask

                    x_from    = 1 / (2 * n)      + i / n
                    x_mid     = 1 / (2 * N)      + id_ / N
                    x_to      = 1 / (2 * n_next) + j / n_next
                    y_bottom  = l * (y0 + z0)
                    y_box_bot = l * (y0 + z0) + y0 / 2 - y1
                    y_box_top = l * (y0 + z0) + y0 / 2 + y1
                    y_top_    = l * (y0 + z0) + y0

                    ax.plot([x_from, x_mid], [y_bottom, y_box_bot],
                            color=ecol, lw=2 * scale, alpha=a)
                    ax.plot([x_mid, x_to], [y_box_top, y_top_],
                            color=ecol, lw=2 * scale, alpha=a)

        # Connections from post-logit to next-layer inputs (straight, 1-to-1)
        if l < neuron_depth - 1:
            n_post = width_out[l + 1]
            n_pre  = width_in[l + 1]
            for i in range(n_post):
                j = i % n_pre
                ax.plot(
                    [1 / (2 * n_post) + i / n_post,
                     1 / (2 * n_pre)  + j / n_pre],
                    [l * (y0 + z0) + y0, (l + 1) * (y0 + z0)],
                    color='black', lw=2 * scale,
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1 * (y0 + z0), (total_depth + 0.1) * (y0 + z0))
    ax.axis('off')

    # ---- KAN operation annotations at output neurons ----
    fs_ann = max(8, int(24 * scale))
    for l in range(depth):
        n_out     = width_out[l + 1]
        kan_layer = model.act_fun[l]
        K         = kan_layer.logits.shape[-1] - 1

        with torch.no_grad():
            dom = torch.softmax(
                kan_layer.logits / tau_decode, dim=-1
            ).argmax(dim=-1).cpu()   # (in_dim, out_dim)

        for j in range(n_out):
            votes    = dom[:, j].tolist()
            n_sum_v  = sum(1 for v in votes if v == K)
            ann, acol = ('Σ', OP_COLORS['sum']) if n_sum_v >= len(votes) - n_sum_v \
                        else ('Π', _op_color(0, 1))
            x_ann = 1 / (2 * n_out) + j / n_out
            y_ann = l * (y0 + z0) + y0
            ax.text(x_ann, y_ann, ann,
                    ha='center', va='center',
                    color=acol, fontsize=fs_ann, fontweight='bold', zorder=5)

    # ---- Phase C: embed spline inset axes ----
    for l in range(depth):
        n      = width_in[l]
        n_next = width_out[l + 1]
        N      = n * n_next

        for i in range(n):
            for j in range(n_next):
                id_ = i * n_next + j
                x_arr, y_arr, spline_color, amask = spline_data[(l, i, j)]

                left   = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                right  = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                bottom = DC_to_NFC([0, l * (y0 + z0) + y0 / 2 - y1])[1]
                up     = DC_to_NFC([0, l * (y0 + z0) + y0 / 2 + y1])[1]

                w = right - left
                h = up    - bottom
                if w <= 0 or h <= 0:
                    continue

                newax = fig.add_axes([left, bottom, w, h])
                ea    = max(float(alpha[l][j][i]) * amask, 0.05 * amask)

                newax.plot(x_arr, y_arr, color=spline_color, lw=2, alpha=ea)
                if sample:
                    newax.scatter(x_arr, y_arr, color=spline_color,
                                  s=100 * scale ** 2, alpha=ea)

                newax.set_xticks([])
                newax.set_yticks([])
                for spine in newax.spines.values():
                    spine.set_color(spline_color if amask > 0 else 'white')
                    spine.set_linewidth(1.5)
                newax.patch.set_edgecolor(spline_color if amask > 0 else 'white')
                newax.patch.set_linewidth(1.5)

    # ---- Phase D: draw FC layers above the KAN stack ----
    # The KAN output nodes are at y = (neuron_depth-1)*(y0+z0).
    # FC layer l_fc connects from y = (neuron_depth-1+l_fc)*(y0+z0)
    #                              to y = (neuron_depth  +l_fc)*(y0+z0).
    # Edges: gray, opacity and width proportional to |weight|.
    # Nodes: black dots at the FC output level.
    FC_EDGE_COLOR = '#777777'
    fs_fc_lbl     = max(6, int(16 * scale))
    for l_fc, fc_layer in enumerate(fc_linears):
        n_in  = fc_layer.in_features
        n_out = fc_layer.out_features
        y_in  = (neuron_depth - 1 + l_fc) * (y0 + z0)
        y_out = (neuron_depth     + l_fc) * (y0 + z0)

        W     = fc_layer.weight.detach().cpu().numpy()  # (n_out, n_in)
        W_abs = np.abs(W)
        W_max = W_abs.max() + 1e-8

        # Edges
        for i in range(n_in):
            for j in range(n_out):
                w_n  = float(W_abs[j, i] / W_max)
                a_e  = float(np.clip(w_n * 0.82 + 0.06, 0.05, 0.92))
                lw_e = float(np.clip(w_n * 2.0 + 0.3, 0.3, 3.0)) * scale
                ax.plot(
                    [1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out],
                    [y_in, y_out],
                    color=FC_EDGE_COLOR, alpha=a_e, lw=lw_e, zorder=1,
                )

        # Output nodes
        fc_node_sz = (A / max(n_out, 5)) ** 2 * 10000 * scale ** 2
        for j in range(n_out):
            ax.scatter(
                1 / (2 * n_out) + j / n_out, y_out,
                s=fc_node_sz, color='black', zorder=3,
            )

        # Layer label (right margin)
        ax.text(
            1.03, (y_in + y_out) / 2,
            f'FC {l_fc}\n({n_in}→{n_out})',
            fontsize=fs_fc_lbl, ha='left', va='center', color='#444',
            transform=ax.transData,
        )

    # ---- Phase E: variable labels ----
    main_ax = fig.get_axes()[0]

    if in_vars is not None:
        n = model.width_in[0]
        for i, var in enumerate(in_vars):
            try:
                import sympy as sp
                if isinstance(var, sp.Expr):
                    txt = f'${latex(var)}$'
                else:
                    txt = str(var)
            except Exception:
                txt = str(var)
            main_ax.text(1 / (2 * n) + i / n, -0.1, txt,
                         fontsize=40 * scale * varscale,
                         ha='center', va='center')

    if out_vars is not None:
        # Place labels above the topmost layer (FC top if present, else KAN top)
        if fc_linears:
            n_top = fc_linears[-1].out_features
            y_top = total_depth * (y0 + z0) + 0.15
        else:
            n_top = model.width_in[-1]
            y_top = (y0 + z0) * (neuron_depth - 1) + 0.15
        for i, var in enumerate(out_vars):
            try:
                import sympy as sp
                if isinstance(var, sp.Expr):
                    txt = f'${latex(var)}$'
                else:
                    txt = str(var)
            except Exception:
                txt = str(var)
            main_ax.text(1 / (2 * n_top) + i / n_top, y_top, txt,
                         fontsize=40 * scale * varscale,
                         ha='center', va='center')

    if title is not None:
        y_t = total_depth * (y0 + z0) + 0.3
        main_ax.text(0.5, y_t, title,
                     fontsize=40 * scale, ha='center', va='center')

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved: {save_path}')

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Function 2: logit operation probability heatmap
# ---------------------------------------------------------------------------

def plot_ops_heatmap(
    model,
    tau_decode: float = 0.05,
    figsize             = None,
    title:      str   = None,
    save_path:  str   = None,
):
    """
    Per-layer heatmap of P(sum) = softmax(logits / tau_decode)[..., -1].

    Blue cells (P_sum ≈ 1) → input is routed to the summation path.
    Red  cells (P_sum ≈ 0) → input is dominated by a multiplicative group.
    Each cell is also annotated with Σ or Πg (dominant operation).

    Parameters
    ----------
    model      : NexusKAN instance
    tau_decode : decoding temperature (low = near-argmax)
    figsize    : (w, h) passed to plt.subplots; auto if None
    title      : overall figure suptitle
    save_path  : save figure to this path if set
    """
    depth = model.depth
    if figsize is None:
        figsize = (max(4, 4 * depth), 4)

    fig, axes = plt.subplots(1, depth, figsize=figsize, squeeze=False)

    for l in range(depth):
        layer  = model.act_fun[l]
        in_dim = layer.in_dim
        out_dim = layer.out_dim
        K      = layer.logits.shape[-1] - 1   # number of mult groups

        with torch.no_grad():
            probs    = torch.softmax(layer.logits / tau_decode, dim=-1).cpu()
            P_sum    = probs[:, :, -1].numpy()     # (in_dim, out_dim)
            dominant = probs.argmax(dim=-1).numpy() # (in_dim, out_dim)

        ax = axes[0, l]
        im = ax.imshow(P_sum, vmin=0, vmax=1, cmap='RdBu',
                       aspect='auto', interpolation='nearest',
                       origin='upper')

        # Cell annotations
        fs_cell = max(7, min(12, int(100 / max(in_dim, out_dim))))
        for i in range(in_dim):
            for j in range(out_dim):
                g   = int(dominant[i, j])
                txt = 'Σ' if g == K else f'Π{g}'
                tcol = 'white' if P_sum[i, j] > 0.6 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=fs_cell, color=tcol)

        ax.set_xlabel('Output  j', fontsize=9)
        ax.set_ylabel('Input  i',  fontsize=9)
        ax.set_title(f'Layer {l} — P(sum)  τ={tau_decode}', fontsize=10)
        ax.set_xticks(range(out_dim))
        ax.set_yticks(range(in_dim))
        ax.set_xticklabels([str(j) for j in range(out_dim)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(in_dim)],  fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved: {save_path}')

    return fig


# ---------------------------------------------------------------------------
# Function 3: activation gallery
# ---------------------------------------------------------------------------

def plot_activation_gallery(
    model,
    n_pts:     int   = 200,
    n_cols:    int   = 8,
    figsize          = None,
    title:     str   = None,
    save_path: str   = None,
):
    """
    Grid of learned B-spline activation curves, one per edge (l, i → j).

    Each curve is evaluated on a dense grid over the layer's spline domain
    using coef2curve directly — no data batch required.

    Curve color matches the dominant operation assignment (blue=Σ, orange=Π₀…).

    Parameters
    ----------
    model     : NexusKAN instance (only needs trained parameters)
    n_pts     : number of probe points per spline
    n_cols    : maximum columns in the grid
    figsize   : (w, h); auto if None
    title     : figure suptitle
    save_path : save figure to this path if set
    """
    device = next(model.parameters()).device
    depth  = model.depth

    edges = []
    for l in range(depth):
        layer = model.act_fun[l]
        for i in range(layer.in_dim):
            for j in range(layer.out_dim):
                edges.append((l, i, j))

    if not edges:
        print('plot_activation_gallery: no edges found.')
        return None

    n_e    = len(edges)
    n_cols = min(n_e, n_cols)
    n_rows = (n_e + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (2.8 * n_cols, 2.2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, (l, i, j) in enumerate(edges):
        row, col = divmod(idx, n_cols)
        ax       = axes[row][col]
        layer    = model.act_fun[l]
        k_spline = layer.k

        # Grid range for input i
        lo = layer.grid[i,  k_spline].item()
        hi = layer.grid[i, -(k_spline + 1)].item()
        if lo >= hi:
            lo, hi = -1.0, 1.0

        x_probe = torch.linspace(lo, hi, n_pts).to(device)

        # Build (n_pts, in_dim) batch — only dimension i varies
        x_batch = torch.zeros(n_pts, layer.in_dim, device=device)
        x_batch[:, i] = x_probe

        with torch.no_grad():
            # coef2curve -> (n_pts, in_dim, out_dim): take slice [i, j]
            spline_vals = coef2curve(x_batch, layer.grid, layer.coef, layer.k)
            y_spline    = spline_vals[:, i, j]          # (n_pts,)
            y_base      = layer.base_fun(x_probe)       # (n_pts,)
            y            = (y_spline * layer.scale_sp[i, j]
                           + y_base  * layer.scale_base[i, j])

        # Decode dominant operation for color
        K = layer.logits.shape[-1] - 1
        with torch.no_grad():
            probs_ij = torch.softmax(layer.logits[i, j, :] / 0.05, dim=0)
        g    = int(probs_ij.argmax().item())
        ecol = _op_color(g, K)

        ax.plot(x_probe.cpu().numpy(), y.cpu().numpy(), color=ecol, lw=1.8)
        ax.set_title(f'φ({l},{i}→{j})', fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(ecol)
        ax.spines['left'].set_color(ecol)

    # Hide unused subplot slots
    for idx in range(n_e, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved: {save_path}')

    return fig


# ---------------------------------------------------------------------------
# Function 4: KAN + FC architecture flow diagram
# ---------------------------------------------------------------------------

def plot_kan_fc_architecture(
    kan_model,
    fc_module,
    in_vars=None,
    out_vars=None,
    title=None,
    scale=1.0,
    tau_decode=0.05,
    n_probe=150,
    save_path=None,
):
    """
    Unified architecture flow diagram for a NexusKAN + FC hybrid model.

    Left section — one panel per KAN layer:
        Each panel is an (in_dim × out_dim) matrix where the cell background
        encodes P(Σ) [blue = summation, red = multiplication] and a small
        B-spline thumbnail is drawn inside every cell, coloured by the
        dominant operation (Σ/Π).

    Right section — one panel per nn.Linear:
        Weight-magnitude heatmap |W_ij| normalised per input column
        (Blues colourmap).

    A bridge annotation between the two sections labels the KAN→FC
    interface dimension with a directional arrow.

    Parameters
    ----------
    kan_model  : NexusKAN instance (only needs trained parameters)
    fc_module  : nn.Module whose child modules include the FC nn.Linear
                 layers in order (e.g. nn.Sequential)
    in_vars    : input variable names for the first KAN layer (optional)
    out_vars   : output variable names for the last FC layer (optional)
    title      : figure suptitle (optional)
    scale      : overall size multiplier; default 1.0
    tau_decode : softmax temperature for decoding KAN operation logits
    n_probe    : number of probe points per spline thumbnail
    save_path  : path to save the figure; skipped if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import torch.nn as nn

    device     = next(kan_model.parameters()).device
    kan_layers = list(kan_model.act_fun)
    fc_linears = [m for m in fc_module.modules() if isinstance(m, nn.Linear)]

    n_kan = len(kan_layers)
    n_fc  = len(fc_linears)

    if n_kan + n_fc == 0:
        print('plot_kan_fc_architecture: no layers found.')
        return None

    # ── Layout constants ─────────────────────────────────────────────
    CELL     = 1.2 * scale   # figure-inches per cell unit
    FC_SCALE = 0.48          # FC panels narrower per output neuron
    THUMB    = 0.50          # fraction of cell used by spline inset

    has_bridge = n_kan > 0 and n_fc > 0

    kan_ratios = [float(l.out_dim) for l in kan_layers]
    fc_ratios  = [float(m.out_features) * FC_SCALE for m in fc_linears]
    gap_ratio  = [0.85] if has_bridge else []
    width_ratios = kan_ratios + gap_ratio + fc_ratios

    fig_w = sum(width_ratios) * CELL + 1.0
    max_rows = max(
        max((l.in_dim  for l in kan_layers), default=0),
        max((m.in_features for m in fc_linears), default=0),
        2,
    )
    fig_h = max_rows * CELL + 1.2 * scale

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        1, len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        left=0.07, right=0.97,
        top=0.87,  bottom=0.14,
        wspace=0.38,
    )

    # ── KAN layer panels ─────────────────────────────────────────────
    for l, layer in enumerate(kan_layers):
        ax    = fig.add_subplot(gs[0, l])
        in_d  = layer.in_dim
        out_d = layer.out_dim
        K     = layer.logits.shape[-1] - 1

        with torch.no_grad():
            probs    = torch.softmax(layer.logits / tau_decode, dim=-1).cpu()
            P_sum    = probs[:, :, -1].numpy()        # (in_dim, out_dim)
            dominant = probs.argmax(dim=-1).numpy()   # (in_dim, out_dim)

        # Cell background: P(Σ) heatmap
        ax.imshow(P_sum, vmin=0, vmax=1, cmap='RdBu',
                   aspect='auto', interpolation='nearest', origin='upper')
        ax.set_xlim(-0.5, out_d - 0.5)
        ax.set_ylim(in_d - 0.5, -0.5)   # origin='upper' → y inverted

        # Operation symbol inside each cell
        fs_ann = max(6, int(13 / max(in_d, out_d, 1) * scale))
        for i in range(in_d):
            for j in range(out_d):
                g   = int(dominant[i, j])
                lbl = 'Σ' if g == K else f'Π{g}'
                ax.text(j, i, lbl, ha='center', va='center',
                         fontsize=fs_ann, color='white', fontweight='bold',
                         zorder=5)

        # Axis ticks
        row_labels = (
            [in_vars[i] if i < len(in_vars) else f'h{i}' for i in range(in_d)]
            if (in_vars is not None and l == 0)
            else [f'h{i}' for i in range(in_d)]
        )
        ax.set_yticks(range(in_d))
        ax.set_yticklabels(row_labels, fontsize=7 * scale)
        ax.set_xticks(range(out_d))
        ax.set_xticklabels([f'h{j}' for j in range(out_d)], fontsize=7 * scale)
        ax.set_title(f'KAN Layer {l}  ({in_d}→{out_d})', fontsize=9 * scale, pad=4)
        if l == 0:
            ax.set_ylabel('input  i', fontsize=7 * scale)
        ax.set_xlabel('output  j', fontsize=7 * scale)

        # P(Σ) colourbar on the first KAN panel only
        if l == 0:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            sm = ScalarMappable(norm=Normalize(0, 1), cmap='RdBu')
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('P(Σ)', fontsize=7 * scale)
            cb.ax.tick_params(labelsize=6 * scale)

        # ── Spline thumbnails ─────────────────────────────────────
        # Cell (i, j) centre in imshow data coords: x=j, y=i.
        # origin='upper' inverts the y-axis: larger data-y = lower on screen.
        # transData maps data→display where display-y increases upward,
        # so data y = i+half → smaller display-y → smaller figure-y.
        DC    = ax.transData.transform
        toNFC = lambda xy: fig.transFigure.inverted().transform(DC(xy))

        for i in range(in_d):
            for j in range(out_d):
                k_sp = layer.k
                lo   = layer.grid[i,  k_sp].item()
                hi   = layer.grid[i, -(k_sp + 1)].item()
                if lo >= hi:
                    lo, hi = -1.0, 1.0

                x_p   = torch.linspace(lo, hi, n_probe).to(device)
                x_bat = torch.zeros(n_probe, in_d, device=device)
                x_bat[:, i] = x_p

                with torch.no_grad():
                    sp_val = coef2curve(x_bat, layer.grid, layer.coef, k_sp)
                    y_sp   = sp_val[:, i, j]
                    y_base = layer.base_fun(x_p)
                    y_out  = (y_sp  * layer.scale_sp[i, j]
                              + y_base * layer.scale_base[i, j])

                g    = int(dominant[i, j])
                col  = _op_color(g, K)
                half = THUMB * 0.5

                # bottom-left corner of inset: data (j-half, i+half) → small fig-y
                bl = toNFC([j - half, i + half])
                # top-right  corner of inset: data (j+half, i-half) → large fig-y
                tr = toNFC([j + half, i - half])
                w  = tr[0] - bl[0]
                h  = tr[1] - bl[1]
                if w < 0.002 or h < 0.002:
                    continue

                inax = fig.add_axes([bl[0], bl[1], w, h])
                inax.plot(x_p.cpu().numpy(), y_out.cpu().numpy(),
                           color=col, lw=1.2)
                inax.set_xticks([])
                inax.set_yticks([])
                inax.patch.set_alpha(0.0)      # transparent → heatmap shows through
                for sp_obj in inax.spines.values():
                    sp_obj.set_visible(False)

    # ── Bridge annotation ────────────────────────────────────────────
    if has_bridge:
        ax_gap = fig.add_subplot(gs[0, n_kan])
        ax_gap.set_xlim(0, 1)
        ax_gap.set_ylim(0, 1)
        ax_gap.axis('off')
        bdim = kan_layers[-1].out_dim
        ax_gap.annotate(
            '', xy=(0.88, 0.5), xytext=(0.12, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#444', lw=2.0 * scale),
        )
        ax_gap.text(0.5, 0.62, f'dim={bdim}',
                     ha='center', va='bottom',
                     fontsize=8 * scale, color='#333', fontweight='bold')
        ax_gap.text(0.5, 0.38, 'KAN→FC',
                     ha='center', va='top',
                     fontsize=7 * scale, color='#666', style='italic')

    # ── FC layer panels ──────────────────────────────────────────────
    for f, layer in enumerate(fc_linears):
        col_idx = n_kan + (1 if has_bridge else 0) + f
        ax  = fig.add_subplot(gs[0, col_idx])
        W   = layer.weight.detach().cpu().numpy()   # (out_features, in_features)
        n_out, n_in = W.shape

        W_abs  = np.abs(W)
        W_norm = W_abs / (W_abs.max(axis=0, keepdims=True) + 1e-8)

        im = ax.imshow(W_norm, cmap='Blues', vmin=0, vmax=1,
                        aspect='auto', interpolation='nearest', origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04)

        fs_fc = max(5, int(11 / max(n_in, n_out, 1) * scale))
        ax.set_xticks(range(n_in))
        ax.set_yticks(range(n_out))
        ax.set_xticklabels([f'h{i}' for i in range(n_in)],
                            fontsize=fs_fc * scale, rotation=45, ha='right')
        y_labels = (
            [out_vars[o] if o < len(out_vars) else f'n{o}' for o in range(n_out)]
            if (out_vars is not None and f == n_fc - 1 and n_out <= len(out_vars))
            else [f'n{o}' for o in range(n_out)]
        )
        ax.set_yticklabels(y_labels, fontsize=fs_fc * scale)
        ax.set_xlabel('Input', fontsize=7 * scale)
        ax.set_ylabel('Output', fontsize=7 * scale)
        ax.set_title(f'FC Layer {f}  ({n_in}→{n_out})', fontsize=9 * scale, pad=4)

    if title:
        fig.suptitle(title, fontsize=12 * scale)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved: {save_path}')

    return fig
