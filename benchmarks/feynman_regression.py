"""
Feynman equations benchmark.

Evaluates NexusKAN, MultKAN, MLP (and optionally PySR) on a curated subset
of Feynman physics equations from nexuskan/feynman.py.

Selected equations (varied structure and input dimension):
    I.6.20a  - Gaussian (1-in)
    I.14.4   - Spring energy (2-in)
    I.8.4    - Distance formula (4-in)
    I.11.19  - Dot product (6-in)
    I.12.2   - Coulomb force (4-in, involves division)
    I.12.4   - Electric field (3-in, involves division)

Run:
    python benchmarks/feynman_regression.py
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
from nexuskan.feynman import get_feynman_dataset

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 5
EPOCHS  = 500
STEPS   = 500
N_TRAIN = 1000
N_TEST  = 200


# ---------------------------------------------------------------------------
# Dataset builder from feynman.py metadata
# ---------------------------------------------------------------------------

def make_feynman_dataset(name, seed=0):
    """Build a train/test dataset for a Feynman equation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    symbol, expr, f, ranges = get_feynman_dataset(name)

    # Determine number of inputs from ranges
    if isinstance(ranges[0], list):
        n_inputs = len(ranges)
        lo = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
        hi = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
    else:
        # Single shared range applied to all inputs; detect n_inputs from function
        # by calling with a dummy 1-input tensor and incrementing until it works
        lo_val, hi_val = float(ranges[0]), float(ranges[1])
        for n in range(1, 20):
            try:
                dummy = torch.rand(2, n) * (hi_val - lo_val) + lo_val
                f(dummy)
                n_inputs = n
                break
            except Exception:
                continue
        lo = torch.full((n_inputs,), lo_val)
        hi = torch.full((n_inputs,), hi_val)

    total = N_TRAIN + N_TEST
    x = torch.rand(total, n_inputs) * (hi - lo) + lo
    y = f(x)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    # Sanity check: drop NaN / Inf rows
    valid = torch.isfinite(y).squeeze(1)
    x, y = x[valid], y[valid]
    if x.shape[0] < N_TRAIN + N_TEST:
        # Re-sample generously and filter
        x_extra = torch.rand((total - x.shape[0]) * 5, n_inputs) * (hi - lo) + lo
        y_extra = f(x_extra)
        if y_extra.dim() == 1:
            y_extra = y_extra.unsqueeze(1)
        mask = torch.isfinite(y_extra).squeeze(1)
        x = torch.cat([x, x_extra[mask]], dim=0)[:total]
        y = torch.cat([y, y_extra[mask]], dim=0)[:total]

    return {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].to(device),
        'test_input':  x[N_TRAIN:N_TRAIN + N_TEST].to(device),
        'test_label':  y[N_TRAIN:N_TRAIN + N_TEST].to(device),
    }, n_inputs


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def r2_score(pred, y):
    mse = ((pred - y) ** 2).mean().item()
    var = y.var().item()
    return 1.0 - mse / (var + 1e-10)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def _get_tau(epoch, epochs, tau_start=2.0, tau_end=0.1):
    return tau_start * (tau_end / tau_start) ** (epoch / max(epochs - 1, 1))

def _get_lr(epoch, epochs, lr_start=0.1, lr_end=1e-3):
    return lr_start * (lr_end / lr_start) ** (epoch / max(epochs - 1, 1))

def _get_lamb(epoch, epochs, lamb_start=1e-2, lamb_end=10.0):
    return lamb_start * (lamb_end / lamb_start) ** (epoch / max(epochs - 1, 1))


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_nexuskan(ds, n_inputs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    width = [n_inputs, max(n_inputs, 4), 1]
    model = NexusKAN(
        width=width, grid=5, k=3,
        seed=seed, auto_save=False, device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // 64, 1)):
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


def run_multkan(ds, n_inputs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    width = [n_inputs, max(n_inputs, 4), 1]
    model = MultKAN(width=width, grid=5, k=3, seed=seed, auto_save=False, device=device)
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    model.fit(
        ds, opt='Adam', steps=STEPS, log=STEPS + 1,
        lr=0.01, loss_fn=loss_fn, batch=64,
    )
    with torch.no_grad():
        pred = model(ds['test_input'])
    return r2_score(pred, ds['test_label'])


def run_mlp(ds, n_inputs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    width = [n_inputs, 2 * max(n_inputs, 4), 1]
    model = MLP(width=width, seed=seed, device=device)
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    model.fit(
        ds, opt='Adam', steps=STEPS, log=STEPS + 1,
        lr=0.01, loss_fn=loss_fn, batch=64,
    )
    with torch.no_grad():
        pred = model(ds['test_input'])
    return r2_score(pred, ds['test_label'])


def run_pysr(ds, n_inputs, seed):
    if not HAS_PYSR:
        return None
    np.random.seed(seed)
    X = ds['train_input'].cpu().numpy()
    y = ds['train_label'].cpu().numpy().ravel()
    model = PySRRegressor(
        niterations=40,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin', 'cos', 'exp', 'log', 'sqrt'],
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

def print_table(eq_name, n_inputs, expr, results):
    print(f'\n{eq_name}  (in_dim={n_inputs}, expr={expr})')
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
# Equations to benchmark
# ---------------------------------------------------------------------------

EQUATIONS = [
    'I.6.20a',   # exp(-theta^2/2)/sqrt(2*pi),  1-in
    'I.14.4',    # 0.5*k_s*x^2,                 2-in
    'I.8.4',     # sqrt distance,                4-in
    'I.11.19',   # dot product,                  6-in
    'I.12.2',    # Coulomb / r^2,                4-in (division)
    'I.12.4',    # E field,                      3-in (division)
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'N_RUNS={N_RUNS}, EPOCHS/STEPS={EPOCHS}')
    if not HAS_PYSR:
        print('Note: PySR not installed — PySR column will show N/A')

    summary = {}  # eq_name -> {method -> success_count}

    for eq_name in EQUATIONS:
        print(f'\n{"=" * 55}')
        print(f'Loading {eq_name}...')
        try:
            _, expr, _, _ = get_feynman_dataset(eq_name)
        except Exception as e:
            print(f'  Skipping {eq_name}: {e}')
            continue

        results = {'NexusKAN': [], 'MultKAN': [], 'MLP': [], 'PySR': []}
        eq_summary = {}

        for run in range(N_RUNS):
            print(f'  Run {run + 1}/{N_RUNS}...', end=' ', flush=True)

            ds, n_inputs = make_feynman_dataset(eq_name, seed=run)

            r2_nexus = run_nexuskan(ds, n_inputs, seed=run)
            r2_mult  = run_multkan(ds, n_inputs, seed=run)
            r2_mlp   = run_mlp(ds, n_inputs, seed=run)
            r2_pysr  = run_pysr(ds, n_inputs, seed=run)

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

        print_table(eq_name, n_inputs, expr, results)
        summary[eq_name] = {
            m: sum(r > 0.99 for r in rs if r is not None)
            for m, rs in results.items()
        }

    # ---- Summary table ----
    print(f'\n{"=" * 55}')
    print('SUMMARY: Success rate (R² > 0.99)')
    methods = ['NexusKAN', 'MultKAN', 'MLP', 'PySR']
    header = f'{"Equation":<20}' + ''.join(f'| {m:<12}' for m in methods)
    print(header)
    print('-' * len(header))
    for eq_name, scores in summary.items():
        row = f'{eq_name:<20}'
        for m in methods:
            cnt = scores.get(m, None)
            row += f'| {str(cnt) + "/" + str(N_RUNS):<12}' if cnt is not None else f'| {"N/A":<12}'
        print(row)
