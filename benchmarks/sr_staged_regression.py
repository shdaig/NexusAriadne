"""
Staged symbolic regression benchmark.

Compares NexusKAN, MultKAN, MLP (and optionally PySR) on four expressions
of increasing complexity, all with inputs sampled from [0.1, 1.0].

  Stage 1: x0 / x1
  Stage 2: x0/x1 + x2/x3
  Stage 3: x0*x1 + x2*x3
  Stage 4: (x0/x1 + x2/x3) / (x4/x5 + x6/x7)

Run:
    python benchmarks/sr_staged_regression.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from nexuskan import NexusKAN, NexusKANLayer, MultKAN
from nexuskan.MLP import MLP

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 3
EPOCHS  = 500   # NexusKAN custom loop iterations
STEPS   = 500   # MultKAN / MLP fit() steps

LO, HI = 0.1, 1.0   # input range (avoids division by zero)
N_TRAIN, N_TEST = 800, 200


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(f, n_inputs, seed=0):
    torch.manual_seed(seed)
    x = (HI - LO) * torch.rand(N_TRAIN + N_TEST, n_inputs) + LO
    y = f(x)
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
    """Coefficient of determination on a tensor pair."""
    mse = ((pred - y) ** 2).mean().item()
    var = y.var().item()
    return 1.0 - mse / (var + 1e-10)


# ---------------------------------------------------------------------------
# Tau / LR schedules (same as existing training scripts)
# ---------------------------------------------------------------------------

def _get_tau(epoch, epochs, tau_start=2.0, tau_end=0.1):
    return tau_start * (tau_end / tau_start) ** (epoch / max(epochs - 1, 1))

def _get_lr(epoch, epochs, lr_start=0.1, lr_end=1e-3):
    return lr_start * (lr_end / lr_start) ** (epoch / max(epochs - 1, 1))

def _get_lamb(epoch, epochs, lamb_start=1e-2, lamb_end=10.0):
    return lamb_start * (lamb_end / lamb_start) ** (epoch / max(epochs - 1, 1))


# ---------------------------------------------------------------------------
# NexusKAN runner
# ---------------------------------------------------------------------------

def run_nexuskan(n_inputs, f, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = make_dataset(f, n_inputs, seed=seed)
    width = [n_inputs, n_inputs, 1]
    model = NexusKAN(
        width=width, grid=5, k=3,
        grid_range=[LO, HI],
        seed=seed, auto_save=False, device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    n_train = N_TRAIN

    for epoch in range(EPOCHS):
        idx = torch.randperm(n_train, device=device)
        for bi in range(max(n_train // 64, 1)):
            batch_idx = idx[bi * 64 : (bi + 1) * 64]
            bx = ds['train_input'][batch_idx]
            by = ds['train_label'][batch_idx]

            for pg in optimizer.param_groups:
                pg['lr'] = _get_lr(epoch, EPOCHS)

            new_tau = _get_tau(epoch, EPOCHS)
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    m.tau = new_tau

            out  = model(bx)
            loss = criterion(out, by)

            lamb_ent = _get_lamb(epoch, EPOCHS)
            ent_ops = 0.0
            n_ops = 0
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    probs = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean()
                    n_ops += 1
            ent_avg = ent_ops / n_ops if n_ops > 0 else 0.0

            total = loss + lamb_ent * ent_avg
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

    with torch.no_grad():
        pred = model(ds['test_input'])
    return r2_score(pred, ds['test_label'])


# ---------------------------------------------------------------------------
# MultKAN runner
# ---------------------------------------------------------------------------

def run_multkan(n_inputs, f, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = make_dataset(f, n_inputs, seed=seed)
    width = [n_inputs, n_inputs, 1]
    model = MultKAN(
        width=width, grid=5, k=3,
        grid_range=[LO, HI],
        seed=seed, auto_save=False, device=device,
    )
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    model.fit(
        ds, opt='Adam', steps=STEPS, log=STEPS + 1,
        lr=0.01, loss_fn=loss_fn, batch=64,
    )
    with torch.no_grad():
        pred = model(ds['test_input'])
    return r2_score(pred, ds['test_label'])


# ---------------------------------------------------------------------------
# MLP runner
# ---------------------------------------------------------------------------

def run_mlp(n_inputs, f, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = make_dataset(f, n_inputs, seed=seed)
    width = [n_inputs, 2 * n_inputs, 1]
    model = MLP(width=width, seed=seed, device=device)
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    model.fit(
        ds, opt='Adam', steps=STEPS, log=STEPS + 1,
        lr=0.01, loss_fn=loss_fn, batch=64,
    )
    with torch.no_grad():
        pred = model(ds['test_input'])
    return r2_score(pred, ds['test_label'])


# ---------------------------------------------------------------------------
# PySR runner
# ---------------------------------------------------------------------------

def run_pysr(n_inputs, f, seed):
    if not HAS_PYSR:
        return None
    np.random.seed(seed)
    ds = make_dataset(f, n_inputs, seed=seed)
    X = ds['train_input'].cpu().numpy()
    y = ds['train_label'].cpu().numpy().ravel()
    model = PySRRegressor(
        niterations=40,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=[],
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)
    X_test = ds['test_input'].cpu().numpy()
    y_test = ds['test_label'].cpu().numpy().ravel()
    pred = model.predict(X_test)
    mse = float(np.mean((pred - y_test) ** 2))
    var = float(np.var(y_test))
    return 1.0 - mse / (var + 1e-10)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(stage_name, results):
    print(f'\n{stage_name}')
    print(f'{"Method":<14}| {"Mean R²":>10} | {"Std R²":>8} | {"Success":>7}')
    print('-' * 48)
    for method, r2s in results.items():
        valid = [r for r in r2s if r is not None]
        if not valid:
            print(f'{method:<14}| {"N/A":>10} | {"N/A":>8} | {"N/A":>7}')
            continue
        mean_r2 = float(np.mean(valid))
        std_r2  = float(np.std(valid))
        success = sum(r > 0.99 for r in valid)
        print(f'{method:<14}| {mean_r2:>10.4f} | {std_r2:>8.4f} | {success:>4}/{len(valid)}')


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

STAGES = [
    (
        'Stage 1: x0 / x1',
        2,
        lambda x: x[:, [0]] / x[:, [1]],
    ),
    (
        'Stage 2: x0/x1 + x2/x3',
        4,
        lambda x: x[:, [0]] / x[:, [1]] + x[:, [2]] / x[:, [3]],
    ),
    (
        'Stage 3: x0*x1 + x2*x3',
        4,
        lambda x: x[:, [0]] * x[:, [1]] + x[:, [2]] * x[:, [3]],
    ),
    (
        'Stage 4: (x0/x1 + x2/x3) / (x4/x5 + x6/x7)',
        8,
        lambda x: (x[:, [0]] / x[:, [1]] + x[:, [2]] / x[:, [3]])
                / (x[:, [4]] / x[:, [5]] + x[:, [6]] / x[:, [7]]),
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'N_RUNS={N_RUNS}, EPOCHS/STEPS={EPOCHS}')
    if not HAS_PYSR:
        print('Note: PySR not installed — PySR column will show N/A')

    for stage_name, n_inputs, f in STAGES:
        print(f'\n{"=" * 55}')
        print(f'Running {stage_name}  (in_dim={n_inputs})')

        results = {'NexusKAN': [], 'MultKAN': [], 'MLP': [], 'PySR': []}

        for run in range(N_RUNS):
            print(f'  Run {run + 1}/{N_RUNS}...', end=' ', flush=True)

            r2_nexus = run_nexuskan(n_inputs, f, seed=run)
            r2_mult  = run_multkan(n_inputs, f, seed=run)
            r2_mlp   = run_mlp(n_inputs, f, seed=run)
            r2_pysr  = run_pysr(n_inputs, f, seed=run)

            results['NexusKAN'].append(r2_nexus)
            results['MultKAN'].append(r2_mult)
            results['MLP'].append(r2_mlp)
            results['PySR'].append(r2_pysr)

            line = (f'NexusKAN={r2_nexus:.4f}, '
                    f'MultKAN={r2_mult:.4f}, '
                    f'MLP={r2_mlp:.4f}')
            if r2_pysr is not None:
                line += f', PySR={r2_pysr:.4f}'
            print(line)

        print_table(stage_name, results)
