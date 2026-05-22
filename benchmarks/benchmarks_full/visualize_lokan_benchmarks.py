"""
Visualize trained LOKAN models from all three full benchmarks.

Loads saved .pt model files from benchmarks/results/models/ and generates:
  - plot_nexus_network   : full network diagram with logit-colored edges
  - plot_ops_heatmap     : per-layer P(sum) heatmap
  - plot_activation_gallery : B-spline activation gallery

For LOKAN+FC models (classification and kan_fc) additionally:
  - plot_kan_fc_architecture : unified KAN+FC flow diagram

Figures saved to benchmarks/results/figures/.

Requirements
------------
Run at least one of the following first to produce .pt files:
    python benchmarks/benchmarks_full/sr_staged_regression.py
    python benchmarks/benchmarks_full/feature_ext_classification.py
    python benchmarks/benchmarks_full/kan_fc_nonlinear.py

Usage:
    python benchmarks/benchmarks_full/visualize_lokan_benchmarks.py
    python benchmarks/benchmarks_full/visualize_lokan_benchmarks.py --tag sr_stage1
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')   # headless rendering

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)

from lokan import LOKAN
from lokan.nexus_plot import (
    plot_nexus_network,
    plot_ops_heatmap,
    plot_activation_gallery,
    plot_kan_fc_architecture,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR  = os.path.join(ROOT, 'benchmarks', 'results', 'models')
FIGURES_DIR = os.path.join(ROOT, 'benchmarks', 'results', 'figures')

LO, HI = 0.1, 1.0
H_KAN  = 4
H_FC   = 8


# ---------------------------------------------------------------------------
# Data generators — produce a small probe batch for activating save_act
# (must match the ranges used during training for each benchmark)
# ---------------------------------------------------------------------------

def _sample_sr(n_in, n=500):
    """Uniform sample from [0.1, 1.0]^n_in."""
    x = (HI - LO) * torch.rand(n, n_in) + LO
    return x.to(device)


def _sample_clf(n_in, n=500):
    """Uniform sample from [0.1, 1.0]^n_in (same as SR)."""
    x = (HI - LO) * torch.rand(n, n_in) + LO
    return x.to(device)


_KANFC_RANGES = {
    'T1': [[0.5, 2.0], [0.5, 2.0]],
    'T2': [[0.0, 1.0], [0.0, 1.0]],
    'T3': [[0.0, 1.5], [0.0, 1.5]],
    'T4': [[0.0, 1.0], [0.0, 1.5], [0.0, 1.5]],
}


def _sample_kanfc(tag, n=500):
    """Sample from the range used for each KAN+FC task."""
    ranges = _KANFC_RANGES.get(tag.upper())
    if ranges is None:
        return None
    lo = torch.tensor([r[0] for r in ranges])
    hi = torch.tensor([r[1] for r in ranges])
    x  = torch.rand(n, len(ranges)) * (hi - lo) + lo
    return x.to(device)


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def _build_lokan(config):
    """Reconstruct a LOKAN instance from a saved config dict.

    LOKAN.__init__ mutates its width argument in-place (int→[int,0]), so we
    pass a shallow copy to avoid corrupting config.
    SR models use 'width'/'grid'/'k'; CLF and KANFC use 'lokan_*' variants.
    """
    width = list(config.get('lokan_width', config.get('width')))
    grid  = int(config.get('lokan_grid',  config.get('grid')))
    k     = int(config.get('lokan_k',     config.get('k')))
    return LOKAN(
        width=width,
        grid=grid,
        k=k,
        seed=config.get('seed', 0),
        device=device,
        save_act=True,
    )


def _build_clf_fc(config):
    """FC head for classification: Linear(n→1) + Sigmoid."""
    width = config.get('lokan_width', config.get('width'))
    n_in  = int(width[-1]) if not isinstance(width[-1], list) else int(width[-1][0])
    return nn.Sequential(nn.Linear(n_in, 1), nn.Sigmoid()).to(device)


def _build_kanfc_fc(config):
    """FC head for KAN+FC regression: 4-layer MLP."""
    h_fc  = int(config.get('h_fc', H_FC))
    width = config.get('lokan_width', config.get('width'))
    h_in  = int(width[-1]) if not isinstance(width[-1], list) else int(width[-1][0])
    return nn.Sequential(
        nn.Linear(h_in,      h_fc),
        nn.SiLU(),
        nn.Linear(h_fc,      h_fc // 2),
        nn.SiLU(),
        nn.Linear(h_fc // 2, 1),
    ).to(device)


def load_model(pt_path):
    """Load a saved .pt bundle and return (lokan, fc_or_None, config, tag)."""
    bundle = torch.load(pt_path, map_location=device)
    config = bundle['config']
    tag    = config['tag']

    # All models have a LOKAN part
    lokan = _build_lokan(config)
    lokan.load_state_dict(bundle['lokan_state'])
    lokan.to(device)
    lokan.eval()

    fc = None
    if bundle.get('fc_state') is not None:
        if tag.startswith('clf_'):
            fc = _build_clf_fc(config)
        elif tag.startswith('kanfc_'):
            fc = _build_kanfc_fc(config)
        if fc is not None:
            fc.load_state_dict(bundle['fc_state'])
            fc.to(device)
            fc.eval()

    return lokan, fc, config, tag


def _warmup(lokan, fc, probe_x):
    """Run a forward pass to populate save_act data (acts, spline_postacts)."""
    lokan.save_act = True
    with torch.no_grad():
        feat = lokan(probe_x)
        if fc is not None:
            fc(feat)


def _probe_x(tag, config):
    """Return a small tensor to warm up the model's activation cache."""
    width = config.get('lokan_width', config.get('width'))
    n_in  = int(width[0]) if not isinstance(width[0], list) else int(width[0][0])
    if tag.startswith('sr_'):
        return _sample_sr(n_in)
    elif tag.startswith('clf_'):
        return _sample_clf(n_in)
    elif tag.startswith('kanfc_'):
        task_tag = tag.split('_')[-1]   # 'T1', 'T2', ...
        x = _sample_kanfc(task_tag)
        if x is None:
            return _sample_sr(n_in)
        return x
    return _sample_sr(n_in)


# ---------------------------------------------------------------------------
# Input variable names for each benchmark
# ---------------------------------------------------------------------------

def _in_vars(tag, n_in):  # noqa: n_in already extracted as int
    if tag.startswith('sr_'):
        return [f'x{i}' for i in range(n_in)]
    elif tag.startswith('clf_'):
        t = tag.split('_')[-1]
        labels = {'T1': ['x0','x1','x2','x3'],
                  'T2': ['x0','x1','x2','x3'],
                  'T3': ['x0','x1','x2','x3','x4','x5']}
        return labels.get(t, [f'x{i}' for i in range(n_in)])
    elif tag.startswith('kanfc_'):
        return [f'x{i}' for i in range(n_in)]
    return [f'x{i}' for i in range(n_in)]


# ---------------------------------------------------------------------------
# Per-model visualization
# ---------------------------------------------------------------------------

def visualize_one(pt_path, figures_dir):
    print(f'\n  Loading: {os.path.basename(pt_path)}')
    try:
        lokan, fc, config, tag = load_model(pt_path)
    except Exception as e:
        print(f'  ERROR loading model: {e}')
        return

    probe_x = _probe_x(tag, config)
    _warmup(lokan, fc, probe_x)

    width  = config.get('lokan_width', config.get('width'))
    n_in   = int(width[0]) if not isinstance(width[0], list) else int(width[0][0])
    ivars  = _in_vars(tag, n_in)
    prefix = os.path.join(figures_dir, tag)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Network diagram
    try:
        plot_nexus_network(
            lokan,
            fc_module  = fc,
            in_vars    = ivars,
            title      = tag,
            tau_decode = 0.05,
            save_path  = f'{prefix}_network.png',
        )
        print(f'  network diagram → {tag}_network.png')
    except Exception as e:
        print(f'  plot_nexus_network failed: {e}')

    # 2. Ops heatmap
    try:
        plot_ops_heatmap(
            lokan,
            title     = f'{tag} — P(sum)',
            save_path = f'{prefix}_ops_heatmap.png',
        )
        print(f'  ops heatmap     → {tag}_ops_heatmap.png')
    except Exception as e:
        print(f'  plot_ops_heatmap failed: {e}')

    # 3. Activation gallery
    try:
        plot_activation_gallery(
            lokan,
            title     = f'{tag} — activation gallery',
            save_path = f'{prefix}_act_gallery.png',
        )
        print(f'  act gallery     → {tag}_act_gallery.png')
    except Exception as e:
        print(f'  plot_activation_gallery failed: {e}')

    # 4. KAN+FC flow diagram (for hybrid models only)
    if fc is not None:
        try:
            plot_kan_fc_architecture(
                lokan,
                fc,
                in_vars   = ivars,
                title     = f'{tag} — KAN+FC',
                save_path = f'{prefix}_kan_fc_arch.png',
            )
            print(f'  KAN+FC diagram  → {tag}_kan_fc_arch.png')
        except Exception as e:
            print(f'  plot_kan_fc_architecture failed: {e}')

    import matplotlib.pyplot as plt
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tag', nargs='+', default=None,
        help='Only visualize models whose tag matches (e.g. sr_stage1 clf_T1)',
    )
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.isdir(MODELS_DIR):
        print(f'No models directory found at {MODELS_DIR}')
        print('Run at least one benchmark first.')
        return

    pt_files = sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.endswith('.pt')
    )

    if not pt_files:
        print(f'No .pt files found in {MODELS_DIR}')
        print('Run at least one benchmark first.')
        return

    # Filter by tag if specified
    if args.tag:
        pt_files = [
            f for f in pt_files
            if any(t in f for t in args.tag)
        ]
        if not pt_files:
            print(f'No .pt files match tags {args.tag}')
            return

    print(f'Device  : {device}')
    print(f'Models  : {len(pt_files)} .pt files')
    print(f'Output  : {os.path.relpath(FIGURES_DIR, ROOT)}')

    for fname in pt_files:
        visualize_one(os.path.join(MODELS_DIR, fname), FIGURES_DIR)

    print(f'\nDone. Figures saved to {os.path.relpath(FIGURES_DIR, ROOT)}')


if __name__ == '__main__':
    main()
