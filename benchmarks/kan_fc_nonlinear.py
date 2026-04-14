"""
KAN + trainable operations + FC layers: nonlinear feature detection benchmark.

Architecture under test
-----------------------
  NexusKAN([in_dim → H_kan → H_kan])   — KAN feature extractor
      ↓
  Linear(H_kan → H_fc) → SiLU → Linear(H_fc → H_fc//2) → SiLU → Linear(H_fc//2 → 1)

The KAN layers learn interpretable nonlinear activations and discover which
inputs to multiply vs. sum via the logit mechanism.  The FC layers map the
extracted features to the output.

Tasks — each isolates a different type of nonlinearity
------------------------------------------------------
T1  Inverse pair        f = 1/x0 + 1/x1                              [0.5, 2.0]²
T2  Sine product        f = sin(2π x0) · sin(2π x1)                  [0, 1]²
T3  Exp + inverse       f = exp(x0) + 1/(0.5 + x1)                   [0, 1.5]²
T4  Mixed 3-input       f = sin(2π x0) + 1/(0.5 + x1) + x2²          [0,1]×[0,1.5]²

Interpretability check (per task, last run)
-------------------------------------------
For each input of the first KAN layer the learned spline is probed on 200 pts:
  • nonlinearity_score : 1 − R² of a linear fit  (0=linear, 1=fully nonlinear)
  • best_match         : highest Pearson |r| with {sin, 1/x, exp, x², linear}
  • detected           : whether best_match agrees with the ground-truth function

Run:
    python benchmarks/kan_fc_nonlinear.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from nexuskan import NexusKAN, NexusKANLayer
from nexuskan.spline import coef2curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 3
EPOCHS  = 500
N_TRAIN = 800
N_TEST  = 200
H_KAN   = 4    # hidden width of KAN layers
H_FC    = 8   # hidden width of FC layers


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        'name':          'T1 – Inverse pair',
        'description':   'f = 1/x0 + 1/x1',
        'n_inputs':      2,
        'ranges':        [[0.5, 2.0], [0.5, 2.0]],
        'f':             lambda x: 1.0 / x[:, 0] + 1.0 / x[:, 1],
        'ground_truth':  ['1/x', '1/x'],
        'expected_ops':  'summation — two independent inverse terms',
    },
    {
        'name':          'T2 – Sine product',
        'description':   'f = sin(2π x0) · sin(2π x1)',
        'n_inputs':      2,
        'ranges':        [[0.0, 1.0], [0.0, 1.0]],
        'f':             lambda x: (torch.sin(2 * torch.pi * x[:, 0])
                                    * torch.sin(2 * torch.pi * x[:, 1])),
        'ground_truth':  ['sin', 'sin'],
        'expected_ops':  'multiplication — both inputs feed one product group',
    },
    {
        'name':          'T3 – Exp + inverse',
        'description':   'f = exp(x0) + 1/(0.5 + x1)',
        'n_inputs':      2,
        'ranges':        [[0.0, 1.5], [0.0, 1.5]],
        'f':             lambda x: torch.exp(x[:, 0]) + 1.0 / (0.5 + x[:, 1]),
        'ground_truth':  ['exp(x)', '1/x'],
        'expected_ops':  'summation — distinct nonlinear activations per input',
    },
    {
        'name':          'T4 – Mixed 3-input',
        'description':   'f = sin(2π x0) + 1/(0.5 + x1) + x2²',
        'n_inputs':      3,
        'ranges':        [[0.0, 1.0], [0.0, 1.5], [0.0, 1.5]],
        'f':             lambda x: (torch.sin(2 * torch.pi * x[:, 0])
                                    + 1.0 / (0.5 + x[:, 1])
                                    + x[:, 2] ** 2),
        'ground_truth':  ['sin', '1/x', 'x²'],
        'expected_ops':  'summation — three structurally distinct terms',
    },
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(task, seed=0):
    torch.manual_seed(seed)
    n = task['n_inputs']
    ranges = task['ranges']
    lo = torch.tensor([r[0] for r in ranges])
    hi = torch.tensor([r[1] for r in ranges])
    total = N_TRAIN + N_TEST
    x = torch.rand(total, n) * (hi - lo) + lo
    y = task['f'](x)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    return {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].to(device),
    }


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def r2_score(pred, y):
    with torch.no_grad():
        mse = ((pred - y) ** 2).mean().item()
        var = y.var().item()
        return 1.0 - mse / (var + 1e-10)


# ---------------------------------------------------------------------------
# NexusKAN + FC hybrid
# ---------------------------------------------------------------------------

class NexusKANHybrid(nn.Module):
    """NexusKAN feature extractor followed by several FC layers."""

    def __init__(self, n_inputs, h_kan=H_KAN, h_fc=H_FC, seed=0):
        super().__init__()
        self.kan = NexusKAN(
            width=[n_inputs, h_kan, h_kan],
            grid=5, k=3,
            seed=seed, auto_save=False, device=device,
        )
        self.fc = nn.Sequential(
            nn.Linear(h_kan, h_fc),
            nn.SiLU(),
            nn.Linear(h_fc, h_fc // 2),
            nn.SiLU(),
            nn.Linear(h_fc // 2, 1),
        )

    def forward(self, x):
        x = self.kan(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# MLP baseline (comparable capacity)
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    """Fully-connected baseline with the same depth as NexusKANHybrid."""

    def __init__(self, n_inputs, hidden=None):
        super().__init__()
        h = hidden or (H_KAN + H_FC)
        self.net = nn.Sequential(
            nn.Linear(n_inputs, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def _tau(epoch, epochs, start=2.0, end=0.05):
    return start * (end / start) ** (epoch / max(epochs - 1, 1))

def _lr(epoch, epochs, start=1e-2, end=1e-4):
    return start * (end / start) ** (epoch / max(epochs - 1, 1))

def _lamb_ent(epoch, epochs, start=1e-3, end=5.0):
    return start * (end / start) ** (epoch / max(epochs - 1, 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_nexuskan(model: NexusKANHybrid, ds, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    epoch_r2  = []

    # Adapt grid to data distribution before training
    with torch.no_grad():
        model.kan.update_grid(ds['train_input'])

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // 64, 1)):
            batch_idx = idx[bi * 64 : (bi + 1) * 64]
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

            # Entropy regularization on operation logits
            lamb_ent = _lamb_ent(epoch, epochs)
            ent_ops = 0.0
            n_ops = 0
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    probs = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean()
                    n_ops += 1
            ent_avg = ent_ops / n_ops if n_ops > 0 else 0.0

            # Spline activation L1 regularization
            l1_acts = 0.0
            n_acts = 0
            for acts in model.kan.acts_scale_spline:
                l1_acts += acts.abs().sum()
                n_acts += 1
            l1_avg = l1_acts / n_acts if n_acts > 0 else 0.0

            total = loss + lamb_ent * ent_avg + 1e-4 * l1_avg
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(ds['test_input'])
            epoch_r2.append(r2_score(pred, ds['test_label']))

    # Final forward to populate acts_scale_spline
    with torch.no_grad():
        model(ds['test_input'])

    return epoch_r2


def train_mlp(model: MLPRegressor, ds, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    epoch_r2  = []

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // 64, 1)):
            batch_idx = idx[bi * 64 : (bi + 1) * 64]
            bx = ds['train_input'][batch_idx]
            by = ds['train_label'][batch_idx]
            for pg in optimizer.param_groups:
                pg['lr'] = _lr(epoch, epochs)
            out  = model(bx)
            loss = criterion(out, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(ds['test_input'])
            epoch_r2.append(r2_score(pred, ds['test_label']))

    return epoch_r2


# ---------------------------------------------------------------------------
# Interpretability probe
# ---------------------------------------------------------------------------

def _eval_spline(layer: NexusKANLayer, input_idx: int, n_pts: int = 200):
    """
    Evaluate the learned activation for input_idx across its grid range.

    The B-spline for dimension `input_idx` is independent of other dimensions
    in coef2curve, so we can probe it by fixing other inputs to an arbitrary
    value (0 here) and reading off dimension input_idx's slice of the output.

    Returns (x_np, y_np) arrays of shape (n_pts,).
    """
    k   = layer.k
    lo  = layer.grid[input_idx, k].item()
    hi  = layer.grid[input_idx, -(k + 1)].item()

    x_probe = torch.linspace(lo, hi, n_pts).to(device)

    # Build a (n_pts, in_dim) batch — only column input_idx varies
    x_batch = torch.zeros(n_pts, layer.in_dim, device=device)
    x_batch[:, input_idx] = x_probe

    with torch.no_grad():
        spline_all = coef2curve(x_batch, layer.grid, layer.coef, layer.k)
        # shape: (n_pts, in_dim, out_dim) — take dim input_idx, mean over outputs
        spline_i   = spline_all[:, input_idx, :].mean(dim=1)          # (n_pts,)
        base_i     = layer.base_fun(x_probe)                          # (n_pts,)
        scale_sp   = layer.scale_sp[input_idx].mean()
        scale_base = layer.scale_base[input_idx].mean()
        y = spline_i * scale_sp + base_i * scale_base

    return x_probe.cpu().numpy(), y.cpu().numpy()


def nonlinearity_score(x_np, y_np):
    """1 − R² of a linear fit.  0 = linear, 1 = perfectly nonlinear."""
    coeffs = np.polyfit(x_np, y_np, 1)
    y_lin  = np.polyval(coeffs, x_np)
    ss_res = np.sum((y_np - y_lin) ** 2)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2) + 1e-10
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def best_matching_fn(x_np, y_np):
    """
    Return (name, |r|) of the candidate function most correlated with y.
    Candidates span the nonlinearities present in the benchmark tasks.
    """
    def _norm(v):
        std = v.std() + 1e-8
        return (v - v.mean()) / std

    x_c   = x_np - x_np.mean()           # centered x for range-agnostic versions
    x_pos = np.abs(x_np) + 1e-4          # strictly positive for 1/x

    candidates = {
        'sin':    np.sin(2 * np.pi * x_np / (x_np.max() - x_np.min() + 1e-8)),
        '1/x':    1.0 / x_pos,
        'exp(x)': np.exp(x_c),
        'x²':     x_c ** 2,
        'linear': x_np,
    }

    y_n = _norm(y_np)
    best_name, best_r = 'unknown', -1.0
    for name, ref in candidates.items():
        if not np.all(np.isfinite(ref)):
            continue
        r = float(abs(np.dot(y_n, _norm(ref))) / len(y_n))
        if r > best_r:
            best_r, best_name = r, name

    return best_name, best_r


def probe_nonlinearities(model: NexusKANHybrid, task: dict):
    """
    For each input of the first KAN layer, print:
      - nonlinearity score
      - best-matching known function and |r|
      - detection result vs. ground truth
    """
    layer = None
    for m in model.kan.modules():
        if isinstance(m, NexusKANLayer):
            layer = m
            break
    if layer is None:
        print('  [no NexusKANLayer found]')
        return

    ground_truth = task['ground_truth']
    n_detected   = 0
    n_inputs     = task['n_inputs']

    print(f'  Nonlinearity detection (layer 0, {n_inputs} inputs):')
    print(f'  {"Input":<8}{"GT":>8}{"nl-score":>10}{"best-match":>12}{"  |r|":>7}{"detected":>10}')
    print(f'  {"-" * 57}')

    for i in range(n_inputs):
        x_np, y_np = _eval_spline(layer, i)
        nl_score   = nonlinearity_score(x_np, y_np)
        match_name, match_r = best_matching_fn(x_np, y_np)
        gt         = ground_truth[i] if i < len(ground_truth) else '?'
        detected   = (match_name == gt)
        if detected:
            n_detected += 1
        flag = '✓' if detected else '✗'
        print(f'  x{i:<7}{gt:>8}{nl_score:>10.3f}{match_name:>12}{match_r:>7.3f}{flag:>10}')

    print(f'  Detection: {n_detected}/{n_inputs}  |  '
          f'Expected ops: {task["expected_ops"]}')


def probe_operations(model: NexusKANHybrid):
    """Print operation assignment from the logits of each KAN layer."""
    layers = [m for m in model.kan.modules() if isinstance(m, NexusKANLayer)]
    for l_idx, layer in enumerate(layers):
        with torch.no_grad():
            probs = torch.softmax(layer.logits / 0.05, dim=-1)   # sharp decoding
            ent   = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            assign = probs.argmax(dim=-1).cpu()                   # (in, out)
            n_groups = layer.logits.shape[-1] - 1
            n_mult = (assign < n_groups).sum().item()
            n_sum  = (assign == n_groups).sum().item()
        total = n_mult + n_sum
        print(f'  Layer {l_idx}:  ent={ent:.3f} nats  |  '
              f'mult={n_mult}/{total}  sum={n_sum}/{total}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'N_RUNS={N_RUNS}, EPOCHS={EPOCHS}')
    print(f'Architecture: NexusKAN([in → {H_KAN} → {H_KAN}]) + '
          f'FC([{H_KAN} → {H_FC} → {H_FC//2} → 1])\n')

    for task in TASKS:
        n = task['n_inputs']
        print(f'\n{"=" * 65}')
        print(f'{task["name"]}   {task["description"]}   (in_dim={n})')

        results = {
            'NexusKAN+FC': {'r2': [], 'ep95': []},
            'MLP':         {'r2': [], 'ep95': []},
        }
        last_nexuskan = None

        for run in range(N_RUNS):
            torch.manual_seed(run)
            np.random.seed(run)
            ds = make_dataset(task, seed=run)

            # --- NexusKAN + FC ---
            model_kan = NexusKANHybrid(n_inputs=n, seed=run).to(device)
            r2_curve  = train_nexuskan(model_kan, ds)
            final_r2  = r2_curve[-1]
            ep95      = next((i + 1 for i, r in enumerate(r2_curve) if r >= 0.95), None)
            results['NexusKAN+FC']['r2'].append(final_r2)
            results['NexusKAN+FC']['ep95'].append(ep95)
            last_nexuskan = model_kan

            # --- MLP ---
            model_mlp = MLPRegressor(n_inputs=n).to(device)
            r2_curve  = train_mlp(model_mlp, ds)
            final_r2  = r2_curve[-1]
            ep95      = next((i + 1 for i, r in enumerate(r2_curve) if r >= 0.95), None)
            results['MLP']['r2'].append(final_r2)
            results['MLP']['ep95'].append(ep95)

            ep_str = lambda ep: str(ep) if ep else '>EPOCHS'
            print(f'  Run {run + 1}/{N_RUNS}  '
                  f'NexusKAN+FC R²={results["NexusKAN+FC"]["r2"][-1]:.4f} '
                  f'(ep95={ep_str(results["NexusKAN+FC"]["ep95"][-1])})  '
                  f'MLP R²={results["MLP"]["r2"][-1]:.4f} '
                  f'(ep95={ep_str(results["MLP"]["ep95"][-1])})')

        # ---- Interpretability probe on last run ----
        print()
        probe_nonlinearities(last_nexuskan, task)
        probe_operations(last_nexuskan)

        # ---- Per-task summary ----
        print(f'\n  Summary — {task["name"]}')
        print(f'  {"Model":<14}| {"R² mean":>8} | {"R² std":>7} | {"ep→R²≥0.95 mean":>16}')
        print(f'  {"-" * 52}')
        for m_name, res in results.items():
            r2s  = res['r2']
            ep95s = [e for e in res['ep95'] if e is not None]
            mean_ep = float(np.mean(ep95s)) if ep95s else float('nan')
            print(f'  {m_name:<14}| {np.mean(r2s):>8.4f} | {np.std(r2s):>7.4f} | {mean_ep:>16.1f}')
