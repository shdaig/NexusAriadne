"""
LOKAN Feynman Benchmark
=======================
Compares LOKAN, MultKAN, MLP, and PySR on 27 Feynman physics equations
following the KAN paper (Section 3.3) methodology.

Metrics: RMSE and R² on held-out test set (N_TEST=200).
N_RUNS=3 random seeds; mean ± std reported.

Run full benchmark:
    python benchmarks/benchmarks_lite/lokan_feynman_benchmark.py

Smoke-test (1 equation, 1 seed, 10 epochs):
    python benchmarks/benchmarks_lite/lokan_feynman_benchmark.py --smoke-test
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)

from lokan import LOKAN
from lokan.feynman import get_feynman_dataset
from nexuskan.MultKAN import MultKAN
from nexuskan.MLP import MLP

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 3
N_TRAIN = 1000
N_TEST  = 200

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
MULTKAN_CKPT = os.path.join(RESULTS_DIR, 'multkan_ckpt')

# 27 equations: 24 from KAN paper Table 2 + 3 replacements for I.6.2, I.6.2b, II.11.7
# Replacements: I.12.1 (μ·N), I.34.8 (qVB/p), II.34.2a (qv/2πr)
EQUATIONS = [
    'I.9.18',    # Gravitational force (9-in)
    'I.12.1',    # μ·N  — replaces I.6.2 (2-in)
    'I.12.11',   # q(E + Bv·sin θ) (5-in)
    'I.13.12',   # KE in CM frame (5-in)
    'I.15.3x',   # Lorentz position (4-in)
    'I.16.6',    # Relativistic velocity addition (3-in)
    'I.18.4',    # Center of mass (3-in)
    'I.26.2',    # Snell refraction angle (2-in)
    'I.27.6',    # Lens formula (3-in)
    'I.29.16',   # Wave interference distance (4-in)
    'I.30.3',    # Diffraction intensity (3-in)
    'I.30.5',    # Diffraction angle (3-in)
    'I.34.8',    # qVB/p (Lorentz)  — replaces I.6.2b (4-in)
    'I.37.4',    # Two-slit intensity (3-in)
    'I.40.1',    # Boltzmann distribution (6-in)
    'I.44.4',    # Entropy change (5-in)
    'I.50.26',   # Driven oscillator (4-in)
    'II.2.42',   # Heat conduction (5-in)
    'II.6.15a',  # Electric susceptibility (6-in)
    'II.11.27',  # Polarization (4-in)
    'II.34.2a',  # Magnetic moment qv/2πr  — replaces II.11.7 (3-in)
    'II.35.18',  # Boltzmann magnetization (5-in)
    'II.36.38',  # Magnetization with field (8-in)
    'II.38.3',   # Electric field / permittivity (4-in)
    'III.9.52',  # Rabi sinc-squared transition (6-in)
    'III.10.19', # Spin magnitude (4-in)
    'III.17.37', # Doppler-shifted intensity (3-in)
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _detect_n_inputs(f, lo_val, hi_val):
    for n in range(1, 20):
        try:
            dummy = torch.rand(2, n) * (hi_val - lo_val) + lo_val
            f(dummy)
            return n
        except Exception:
            continue
    raise RuntimeError('Could not auto-detect n_inputs from feynman lambda')


def make_feynman_dataset(eq_name, n_train=N_TRAIN, n_test=N_TEST, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, _, f, ranges = get_feynman_dataset(eq_name)

    if isinstance(ranges[0], (list, tuple)):
        n_in = len(ranges)
        lo = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
        hi = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
    else:
        lo_val, hi_val = float(ranges[0]), float(ranges[1])
        n_in = _detect_n_inputs(f, lo_val, hi_val)
        lo = torch.full((n_in,), lo_val)
        hi = torch.full((n_in,), hi_val)

    total = n_train + n_test

    def _sample(n):
        x = torch.rand(n, n_in) * (hi - lo) + lo
        y = f(x)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        mask = torch.isfinite(y).squeeze(1)
        return x[mask], y[mask]

    x, y = _sample(total * 5)
    x, y = x[:total], y[:total]

    if x.shape[0] < total:
        raise RuntimeError(f'{eq_name}: could not sample {total} finite points')

    return {
        'train_input': x[:n_train].to(device),
        'train_label': y[:n_train].to(device),
        'test_input':  x[n_train:total].to(device),
        'test_label':  y[n_train:total].to(device),
    }, n_in


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataset):
    model.eval()
    with torch.no_grad():
        pred = model(dataset['test_input']).cpu()
    y = dataset['test_label'].cpu()
    mse    = float(torch.mean((pred - y) ** 2))
    ss_res = float(torch.sum((pred - y) ** 2))
    ss_tot = float(torch.sum((y - y.mean()) ** 2))
    rmse   = math.sqrt(mse)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    return rmse, r2


# ---------------------------------------------------------------------------
# LOKAN runner
# ---------------------------------------------------------------------------

def run_lokan(dataset, n_in, seed=0, smoke=False):
    torch.manual_seed(seed)
    model = LOKAN(width=[n_in, n_in + 2, 1], grid=5, k=3, seed=seed, device=device)

    epochs          = 5 if smoke else 120
    refine_schedule = None # if smoke else [(20, 5)]

    model.fit(
        dataset=dataset,
        lr_start=0.2, lr_end=1e-3,
        batch_size=64,
        epochs=epochs,
        tau_start=2.0, tau_end=0.1,
        lamb_ent_ops_start=1e-2, lamb_ent_ops_end=1.0,
        lamb_l1_acts=1e-3, lamb_ent_acts=1e-6,
        update_grid=True,
        # update_grid=False, 
        grid_update_num=10,
        start_grid_update_epoch=0, stop_grid_update_epoch=None,
        refine_schedule=refine_schedule,
        verbose=epochs + 1,  # suppress per-epoch output
    )
    return model


# ---------------------------------------------------------------------------
# MultKAN runner  (staged grid extension: 3 → 5 → 10, 600 steps each)
# ---------------------------------------------------------------------------

def run_multkan(dataset, n_in, seed=0, smoke=False):
    torch.manual_seed(seed)
    n_mult = max(1, n_in // 2)
    width  = [[n_in, 0], [n_in, n_mult], [1, 0]]
    train_x = dataset['train_input']
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)

    steps_per_stage = 5 if smoke else 600

    def _fit_stage(model, steps, lr):
        # MultKAN computes grid_update_freq = int(stop / num); ensure stop >= num to avoid % 0
        grid_update_num  = 10
        stop_grid_update = max(grid_update_num, steps // 2)
        model.fit(
            dataset, opt='Adam', steps=steps, lr=lr, batch=64,
            log=steps + 1,
            loss_fn=loss_fn,
            update_grid=True, grid_update_num=grid_update_num,
            start_grid_update_step=0,
            stop_grid_update_step=stop_grid_update,
            lamb=1e-3, lamb_l1=1.0, lamb_entropy=2.0,
        )

    # Stage 1: grid = 3
    model = MultKAN(
        width=width, grid=3, k=3, mult_arity=2, seed=seed,
        auto_save=False, ckpt_path=MULTKAN_CKPT, device=device,
    )
    _fit_stage(model, 320, lr=0.01)

    if not smoke:
        # Stage 2: grid = 5
        model.update_grid_from_samples(train_x)
        model = model.refine(5)
        model.update_grid_from_samples(train_x)
        _fit_stage(model, 880, lr=0.005)

        # Stage 3: grid = 10
        model.update_grid_from_samples(train_x)
        model = model.refine(5)
        model.update_grid_from_samples(train_x)
        _fit_stage(model, steps_per_stage, lr=0.001)

    return model


# ---------------------------------------------------------------------------
# MLP runner
# ---------------------------------------------------------------------------

def run_mlp(dataset, n_in, seed=0, smoke=False):
    torch.manual_seed(seed)
    model   = MLP(width=[n_in, 64, 64, 1], seed=seed, device=device)
    steps   = 15 if smoke else 1800
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    model.fit(
        dataset, opt='Adam', steps=steps, lr=0.001, batch=64,
        log=steps + 1,
        loss_fn=loss_fn,
        lamb=1e-4,
    )
    return model


# ---------------------------------------------------------------------------
# PySR runner
# ---------------------------------------------------------------------------

def run_pysr(dataset, n_in, timeout_secs=120):
    if not HAS_PYSR:
        return float('nan'), float('nan')
    X = dataset['train_input'].cpu().numpy()
    y = dataset['train_label'].cpu().numpy().ravel()
    model = PySRRegressor(
        niterations=100,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin', 'cos', 'exp', 'log', 'sqrt'],
        maxsize=30,
        populations=50,
        timeout_in_seconds=timeout_secs,
        verbosity=0,
    )
    model.fit(X, y)
    X_test = dataset['test_input'].cpu().numpy()
    y_test = dataset['test_label'].cpu().numpy().ravel()
    pred   = model.predict(X_test)
    valid  = np.isfinite(pred) & np.isfinite(y_test)
    if valid.sum() < 2:
        return float('nan'), float('nan')
    rmse = float(np.sqrt(np.mean((pred[valid] - y_test[valid]) ** 2)))
    ss_res = float(np.sum((pred[valid] - y_test[valid]) ** 2))
    ss_tot = float(np.sum((y_test[valid] - y_test[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    return rmse, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick 1-equation / 1-seed / few-epoch check')
    parser.add_argument('--no-pysr', action='store_true',
                        help='Skip PySR even if installed')
    parser.add_argument('--equations', nargs='+', default=None,
                        help='Run only the listed equation IDs')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MULTKAN_CKPT, exist_ok=True)

    run_pysr_flag = HAS_PYSR and not args.no_pysr and not args.smoke_test
    equations = [EQUATIONS[0]] if args.smoke_test else (args.equations or EQUATIONS)
    n_runs    = 1 if args.smoke_test else N_RUNS

    print(f'Device  : {device}')
    print(f'Equations: {len(equations)}, Seeds: {n_runs}')
    print(f'PySR    : {"yes" if run_pysr_flag else "no"}')
    print()

    records = []

    for eq_id in equations:
        print(f'{"=" * 60}')
        print(f'Equation: {eq_id}')
        t0 = time.time()

        try:
            _, expr, _, _ = get_feynman_dataset(eq_id)
            print("eq_id ---------------------", eq_id)
        except Exception as e:
            print(f'  SKIP: {e}')
            continue

        rmse_lokan, r2_lokan     = [], []
        rmse_multkan, r2_multkan = [], []
        rmse_mlp, r2_mlp         = [], []

        n_in         = None
        dataset_last = None

        for seed in range(n_runs):
            print(f'  seed={seed}', end=' ', flush=True)
            try:
                dataset, n_in = make_feynman_dataset(eq_id, seed=seed)
            except Exception as e:
                print(f'dataset error: {e}')
                continue
            dataset_last = dataset

            # LOKAN
            try:
                m = run_lokan(dataset, n_in, seed=seed, smoke=args.smoke_test)
                rm, r2 = evaluate(m, dataset)
                print(f'LOKAN={r2:.3f}', end=' ', flush=True)
            except Exception as e:
                rm, r2 = float('nan'), float('nan')
                print(f'LOKAN=ERR({e})', end=' ', flush=True)
            rmse_lokan.append(rm); r2_lokan.append(r2)

            # MultKAN
            try:
                m = run_multkan(dataset, n_in, seed=seed, smoke=args.smoke_test)
                rm, r2 = evaluate(m, dataset)
                print(f'MultKAN={r2:.3f}', end=' ', flush=True)
            except Exception as e:
                rm, r2 = float('nan'), float('nan')
                print(f'MultKAN=ERR({e})', end=' ', flush=True)
            rmse_multkan.append(rm); r2_multkan.append(r2)

            # MLP
            try:
                m = run_mlp(dataset, n_in, seed=seed, smoke=args.smoke_test)
                rm, r2 = evaluate(m, dataset)
                print(f'MLP={r2:.3f}', end=' ', flush=True)
            except Exception as e:
                rm, r2 = float('nan'), float('nan')
                print(f'MLP=ERR({e})', end=' ', flush=True)
            rmse_mlp.append(rm); r2_mlp.append(r2)

            print()

        # PySR: run once on the last seed's dataset (stochastic internally)
        pysr_rmse, pysr_r2 = float('nan'), float('nan')
        if run_pysr_flag and dataset_last is not None and n_in is not None:
            print('  PySR...', end=' ', flush=True)
            try:
                pysr_rmse, pysr_r2 = run_pysr(dataset_last, n_in)
                print(f'R²={pysr_r2:.3f}')
            except Exception as e:
                print(f'ERR({e})')

        elapsed = time.time() - t0

        def _safe_mean(lst):
            vals = [v for v in lst if math.isfinite(v)]
            return float(np.mean(vals)) if vals else float('nan')

        def _safe_std(lst):
            vals = [v for v in lst if math.isfinite(v)]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        rec = {
            'eq':     eq_id,
            'n_in':   n_in if n_in is not None else '?',
            'lokan_rmse_mean':   _safe_mean(rmse_lokan),
            'lokan_rmse_std':    _safe_std(rmse_lokan),
            'lokan_r2_mean':     _safe_mean(r2_lokan),
            'lokan_r2_std':      _safe_std(r2_lokan),
            'multkan_rmse_mean': _safe_mean(rmse_multkan),
            'multkan_rmse_std':  _safe_std(rmse_multkan),
            'multkan_r2_mean':   _safe_mean(r2_multkan),
            'multkan_r2_std':    _safe_std(r2_multkan),
            'mlp_rmse_mean':     _safe_mean(rmse_mlp),
            'mlp_rmse_std':      _safe_std(rmse_mlp),
            'mlp_r2_mean':       _safe_mean(r2_mlp),
            'mlp_r2_std':        _safe_std(r2_mlp),
            'pysr_rmse':         pysr_rmse,
            'pysr_r2':           pysr_r2,
            'elapsed_s':         round(elapsed, 1),
        }
        records.append(rec)

        print(f'  [{elapsed:.0f}s] '
              f'LOKAN R²={rec["lokan_r2_mean"]:.4f}±{rec["lokan_r2_std"]:.4f} | '
              f'MultKAN={rec["multkan_r2_mean"]:.4f}±{rec["multkan_r2_std"]:.4f} | '
              f'MLP={rec["mlp_r2_mean"]:.4f}±{rec["mlp_r2_std"]:.4f}')

    if not records:
        print('No results collected.')
        return

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, 'lokan_feynman_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved → {csv_path}')

    # Console summary
    print(f'\n{"=" * 90}')
    print(f'{"Equation":<14} {"N":>3} | '
          f'{"LOKAN R²":>10} {"±":>3} | '
          f'{"MultKAN R²":>10} {"±":>3} | '
          f'{"MLP R²":>10} {"±":>3} | '
          f'{"PySR R²":>10}')
    print('-' * 90)
    for rec in records:
        print(f'{rec["eq"]:<14} {rec["n_in"]:>3} | '
              f'{rec["lokan_r2_mean"]:>10.4f} {rec["lokan_r2_std"]:>4.4f} | '
              f'{rec["multkan_r2_mean"]:>10.4f} {rec["multkan_r2_std"]:>4.4f} | '
              f'{rec["mlp_r2_mean"]:>10.4f} {rec["mlp_r2_std"]:>4.4f} | '
              f'{rec["pysr_r2"]:>10.4f}')

    # Markdown table for paper
    md_path = os.path.join(RESULTS_DIR, 'lokan_feynman_results.md')
    with open(md_path, 'w') as fh:
        fh.write('| Equation | N | LOKAN R² | MultKAN R² | MLP R² | PySR R² |\n')
        fh.write('|----------|---|----------|------------|--------|----------|\n')
        for rec in records:
            fh.write(
                f'| {rec["eq"]} | {rec["n_in"]} '
                f'| {rec["lokan_r2_mean"]:.4f}±{rec["lokan_r2_std"]:.4f} '
                f'| {rec["multkan_r2_mean"]:.4f}±{rec["multkan_r2_std"]:.4f} '
                f'| {rec["mlp_r2_mean"]:.4f}±{rec["mlp_r2_std"]:.4f} '
                f'| {rec["pysr_r2"]:.4f} |\n'
            )
    print(f'Markdown  → {md_path}')


if __name__ == '__main__':
    main()
