"""
NexusKAN visualization demo.

Trains four NexusKAN models on representative tasks and produces three
types of visualization for each:
  1. Network diagram  — inline spline curves, edges colored by learned operation
  2. Logit heatmap    — per-layer P(sum) matrix derived from softmax(logits/τ)
  3. Activation gallery — grid of all learned B-spline curves

Scenarios
---------
A  SR Stage 2 : x0/x1 + x2/x3         NexusKAN([4,4,1])
B  Nonlinear  : 1/x0 + 1/x1            NexusKAN([2,4,1])  (inverse pair)
C  Nonlinear  : sin(2π x0)·sin(2π x1)  NexusKAN([2,4,1])  (sine product)
D  Classif.   : x0/x3 + x1/x2 > 2.2   NexusKAN([4,4,1]) + FC head

All figures saved to <project_root>/figures/.

Run:
    python benchmarks/visualize_nexuskan.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display required

from nexuskan import NexusKAN, NexusKANLayer
from nexuskan.nexus_plot import (
    plot_nexus_network,
    plot_ops_heatmap,
    plot_activation_gallery,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS_SR   = 500
EPOCHS_CLS  = 300
LR_START    = 0.1
LR_END      = 1e-3
TAU_START   = 2.0
TAU_END     = 0.1
LAMB_START  = 1e-2
LAMB_END    = 10.0
LAMB_L1_ACTS  = 1e-2   # L1 regularization on acts_scale_spline
LAMB_ENT_ACTS = 1e-2   # entropy regularization on acts_scale_spline
N_TRAIN     = 800
N_TEST      = 200
BATCH       = 64

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _lr(epoch, epochs):
    return LR_START * (LR_END / LR_START) ** (epoch / max(epochs - 1, 1))

def _tau(epoch, epochs):
    return TAU_START * (TAU_END / TAU_START) ** (epoch / max(epochs - 1, 1))

def _lamb(epoch, epochs):
    return LAMB_START * (LAMB_END / LAMB_START) ** (epoch / max(epochs - 1, 1))


def _acts_reg(acts_scale_spline_list):
    """Return (l1_avg, entropy_avg) over all layers in acts_scale_spline."""
    l1_total  = 0.0
    ent_total = 0.0
    n = 0
    for acts in acts_scale_spline_list:
        l1_total += acts.sum()
        p_row = acts / (acts.sum(dim=1, keepdim=True) + 1e-8)
        p_col = acts / (acts.sum(dim=0, keepdim=True) + 1e-8)
        ent_row = -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-8), dim=1))
        ent_col = -torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-8), dim=0))
        ent_total += ent_row + ent_col
        n += 1
    if n == 0:
        return 0.0, 0.0
    return l1_total / n, ent_total / n


# ---------------------------------------------------------------------------
# Generic NexusKAN regression trainer
# ---------------------------------------------------------------------------

def train_nexuskan_regression(model, ds, epochs=EPOCHS_SR):
    optimizer = optim.Adam(model.parameters(), lr=LR_START)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // BATCH, 1)):
            batch_idx = idx[bi * BATCH : (bi + 1) * BATCH]
            bx = ds['train_input'][batch_idx]
            by = ds['train_label'][batch_idx]

            for pg in optimizer.param_groups:
                pg['lr'] = _lr(epoch, epochs)

            new_tau = _tau(epoch, epochs)
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    m.tau = new_tau

            out  = model(bx)
            loss = criterion(out, by)

            lamb_e = _lamb(epoch, epochs)
            ent_ops, n_ops = 0.0, 0
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    probs = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean()
                    n_ops += 1
            ent_avg = ent_ops / n_ops if n_ops > 0 else 0.0

            l1_acts, ent_acts = _acts_reg(model.acts_scale_spline)

            total = (loss
                     + lamb_e          * ent_avg
                     + LAMB_L1_ACTS    * l1_acts
                     + LAMB_ENT_ACTS   * ent_acts)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

    # Final pass with save_act=True so plot can read activations
    model.save_act = True
    with torch.no_grad():
        model(ds['test_input'])


# ---------------------------------------------------------------------------
# Scenario A — SR Stage 2: x0/x1 + x2/x3
# ---------------------------------------------------------------------------

def scenario_a():
    print('\n=== Scenario A: SR Stage 2  (x0/x1 + x2/x3) ===')
    torch.manual_seed(0); np.random.seed(0)

    LO, HI = 0.1, 1.0
    x = (HI - LO) * torch.rand(N_TRAIN + N_TEST, 4) + LO
    y = (x[:, 0] / x[:, 1] + x[:, 2] / x[:, 3]).unsqueeze(1)
    ds = {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].to(device),
    }

    model = NexusKAN(
        width=[4, 4, 1], grid=5, k=3,
        grid_range=[LO, HI],
        seed=0, auto_save=False, device=device,
    )
    train_nexuskan_regression(model, ds, epochs=EPOCHS_SR)

    r2 = 1 - ((model(ds['test_input']) - ds['test_label'])**2).mean().item() / ds['test_label'].var().item()
    print(f'  Test R²: {r2:.4f}')

    stem = 'sr_stage2'
    plot_nexus_network(
        model, metric='forward_n', scale=0.7,
        in_vars=['x₀', 'x₁', 'x₂', 'x₃'], out_vars=['y'],
        title='Stage 2: x₀/x₁ + x₂/x₃',
        save_path=os.path.join(OUT_DIR, f'{stem}_network.png'),
    )
    plot_ops_heatmap(
        model, title='Stage 2 — operation probabilities',
        save_path=os.path.join(OUT_DIR, f'{stem}_ops.png'),
    )
    plot_activation_gallery(
        model, title='Stage 2 — B-spline activations',
        save_path=os.path.join(OUT_DIR, f'{stem}_splines.png'),
    )


# ---------------------------------------------------------------------------
# Scenario B — Nonlinear: 1/x0 + 1/x1  (inverse pair)
# ---------------------------------------------------------------------------

def scenario_b():
    print('\n=== Scenario B: Inverse pair  (1/x0 + 1/x1) ===')
    torch.manual_seed(1); np.random.seed(1)

    LO, HI = 0.5, 2.0
    x = (HI - LO) * torch.rand(N_TRAIN + N_TEST, 2) + LO
    y = (1.0 / x[:, 0] + 1.0 / x[:, 1]).unsqueeze(1)
    ds = {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].to(device),
    }

    model = NexusKAN(
        width=[2, 4, 1], grid=5, k=3,
        grid_range=[LO, HI],
        seed=1, auto_save=False, device=device,
    )
    train_nexuskan_regression(model, ds, epochs=EPOCHS_SR)

    r2 = 1 - ((model(ds['test_input']) - ds['test_label'])**2).mean().item() / ds['test_label'].var().item()
    print(f'  Test R²: {r2:.4f}')

    stem = 'nonlinear_inverse'
    plot_nexus_network(
        model, metric='forward_n', scale=0.7,
        in_vars=['x₀', 'x₁'], out_vars=['y'],
        title='Inverse pair: 1/x₀ + 1/x₁',
        save_path=os.path.join(OUT_DIR, f'{stem}_network.png'),
    )
    plot_ops_heatmap(
        model, title='Inverse pair — operation probabilities',
        save_path=os.path.join(OUT_DIR, f'{stem}_ops.png'),
    )
    plot_activation_gallery(
        model, title='Inverse pair — B-spline activations',
        save_path=os.path.join(OUT_DIR, f'{stem}_splines.png'),
    )


# ---------------------------------------------------------------------------
# Scenario C — Nonlinear: sin(2π x0) · sin(2π x1)  (sine product)
# ---------------------------------------------------------------------------

def scenario_c():
    print('\n=== Scenario C: Sine product  (sin(2π x0)·sin(2π x1)) ===')
    torch.manual_seed(2); np.random.seed(2)

    x = torch.rand(N_TRAIN + N_TEST, 2)
    y = (torch.sin(2 * math.pi * x[:, 0]) * torch.sin(2 * math.pi * x[:, 1])).unsqueeze(1)
    ds = {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].to(device),
    }

    model = NexusKAN(
        width=[2, 4, 1], grid=7, k=3,
        grid_range=[0.0, 1.0],
        seed=2, auto_save=False, device=device,
    )
    train_nexuskan_regression(model, ds, epochs=EPOCHS_SR)

    r2 = 1 - ((model(ds['test_input']) - ds['test_label'])**2).mean().item() / ds['test_label'].var().item()
    print(f'  Test R²: {r2:.4f}')

    stem = 'nonlinear_sine'
    plot_nexus_network(
        model, metric='forward_n', scale=0.7,
        in_vars=['x₀', 'x₁'], out_vars=['y'],
        title='Sine product: sin(2π x₀)·sin(2π x₁)',
        save_path=os.path.join(OUT_DIR, f'{stem}_network.png'),
    )
    plot_ops_heatmap(
        model, title='Sine product — operation probabilities',
        save_path=os.path.join(OUT_DIR, f'{stem}_ops.png'),
    )
    plot_activation_gallery(
        model, title='Sine product — B-spline activations',
        save_path=os.path.join(OUT_DIR, f'{stem}_splines.png'),
    )


# ---------------------------------------------------------------------------
# Scenario D — Classification: x0/x3 + x1/x2 > 2.2  (NexusKAN + FC)
# ---------------------------------------------------------------------------

class _NexusKANWithFC(nn.Module):
    def __init__(self, kan_width, seed=0):
        super().__init__()
        out_dim = kan_width[-1] if isinstance(kan_width[-1], int) else kan_width[-1][0]
        self.kan = NexusKAN(
            width=kan_width, grid=5, k=3,
            grid_range=[0.1, 1.0],
            seed=seed, auto_save=False, device=device,
        )
        self.fc      = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.kan(x)))


def scenario_d():
    print('\n=== Scenario D: Classification  (x0/x3 + x1/x2 > 2.2) ===')
    torch.manual_seed(3); np.random.seed(3)

    LO, HI  = 0.1, 1.0
    N_TR, N_TE = 1000, 200
    x = (HI - LO) * torch.rand(N_TR + N_TE, 4) + LO
    y = ((x[:, 0] / x[:, 3] + x[:, 1] / x[:, 2]) > 2.2).float().unsqueeze(1)
    ds = {
        'train_input': x[:N_TR].to(device),
        'train_label': y[:N_TR].to(device),
        'test_input':  x[N_TR:].to(device),
        'test_label':  y[N_TR:].to(device),
    }

    model = _NexusKANWithFC(kan_width=[4, 4, 1], seed=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_START)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS_CLS):
        idx = torch.randperm(N_TR, device=device)
        for bi in range(max(N_TR // BATCH, 1)):
            batch_idx = idx[bi * BATCH : (bi + 1) * BATCH]
            bx = ds['train_input'][batch_idx]
            by = ds['train_label'][batch_idx]

            for pg in optimizer.param_groups:
                pg['lr'] = _lr(epoch, EPOCHS_CLS)

            new_tau = _tau(epoch, EPOCHS_CLS)
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    m.tau = new_tau

            out  = model(bx)
            loss = criterion(out, by)

            lamb_e = _lamb(epoch, EPOCHS_CLS)
            ent_ops, n_ops = 0.0, 0
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    probs = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean()
                    n_ops += 1
            ent_avg = ent_ops / n_ops if n_ops > 0 else 0.0

            l1_acts, ent_acts = _acts_reg(model.kan.acts_scale_spline)

            total = (loss
                     + lamb_e          * ent_avg
                     + LAMB_L1_ACTS    * l1_acts
                     + LAMB_ENT_ACTS   * ent_acts)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

    with torch.no_grad():
        pred = model(ds['test_input'])
        acc  = ((pred > 0.5).float() == ds['test_label']).float().mean().item()
    print(f'  Test accuracy: {acc:.4f}')

    # Populate KAN activations for visualization
    model.kan.save_act = True
    with torch.no_grad():
        model.kan(ds['test_input'])

    stem = 'cls_division'
    plot_nexus_network(
        model.kan, metric='forward_n', scale=0.7,
        in_vars=['x₀', 'x₁', 'x₂', 'x₃'],
        title='Classification: x₀/x₃ + x₁/x₂ > 2.2',
        save_path=os.path.join(OUT_DIR, f'{stem}_network.png'),
    )
    plot_ops_heatmap(
        model.kan, title='Classification — operation probabilities',
        save_path=os.path.join(OUT_DIR, f'{stem}_ops.png'),
    )
    plot_activation_gallery(
        model.kan, title='Classification — B-spline activations',
        save_path=os.path.join(OUT_DIR, f'{stem}_splines.png'),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Device: {device}')
    print(f'Output directory: {OUT_DIR}')

    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()

    print(f'\nAll figures written to {OUT_DIR}/')
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            print(f'  {f}')
