"""
Staged symbolic regression benchmark — full version.

Compares LOKAN, MultKAN, MLP (and optionally PySR) on four algebraic
expressions of increasing complexity with inputs from [0.1, 1.0].

  Stage 1: x0 / x1                                  (2-in)
  Stage 2: x0/x1 + x2/x3                            (4-in)
  Stage 3: x0*x1 + x2*x3                            (4-in)
  Stage 4: (x0/x1 + x2/x3) / (x4/x5 + x6/x7)       (8-in)

Training budget (matched across models):
  LOKAN   : 120 epochs × ⌈800/64⌉ = ~1875 gradient steps
  MultKAN : staged 3→5→10, totaling 1800 steps
  MLP     : 1800 steps

Results saved to benchmarks/results/sr_staged_{results,models}.

Run full benchmark:
    python benchmarks/benchmarks_full/sr_staged_regression.py

Smoke-test (1 stage, 1 seed, 5 epochs):
    python benchmarks/benchmarks_full/sr_staged_regression.py --smoke-test
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)

from lokan import LOKAN
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
LO, HI  = 0.1, 1.0

RESULTS_DIR  = os.path.join(ROOT, 'benchmarks', 'results')
MODELS_DIR   = os.path.join(RESULTS_DIR, 'models')
MULTKAN_CKPT = os.path.join(RESULTS_DIR, 'multkan_ckpt')


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

STAGES = [
    {
        'name':        'Stage 1: x0/x1',
        'n_inputs':    2,
        'f':           lambda x: x[:, [0]] / x[:, [1]],
    },
    {
        'name':        'Stage 2: x0/x1 + x2/x3',
        'n_inputs':    4,
        'f':           lambda x: x[:, [0]] / x[:, [1]] + x[:, [2]] / x[:, [3]],
    },
    {
        'name':        'Stage 3: x0*x1 + x2*x3',
        'n_inputs':    4,
        'f':           lambda x: x[:, [0]] * x[:, [1]] + x[:, [2]] * x[:, [3]],
    },
    {
        'name':        'Stage 4: (x0/x1+x2/x3)/(x4/x5+x6/x7)',
        'n_inputs':    8,
        'f':           lambda x: (
                           x[:, [0]] / x[:, [1]] + x[:, [2]] / x[:, [3]]
                       ) / (
                           x[:, [4]] / x[:, [5]] + x[:, [6]] / x[:, [7]]
                       ),
    },
]


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
    mse = ((pred - y) ** 2).mean().item()
    var = y.var().item()
    return 1.0 - mse / (var + 1e-10)


# ---------------------------------------------------------------------------
# LOKAN runner
# ---------------------------------------------------------------------------

def run_lokan(dataset, n_in, seed=0, smoke=False):
    torch.manual_seed(seed)
    model = LOKAN(
        width=[n_in, n_in + 2, 1], grid=5, k=3,
        grid_range=[LO, HI],
        seed=seed, device=device,
    )
    epochs          = 5 if smoke else 120
    refine_schedule = None
    model.fit(
        dataset=dataset,
        lr_start=0.1, lr_end=1e-3,
        batch_size=64,
        epochs=epochs,
        tau_start=2.0, tau_end=0.1,
        lamb_ent_ops_start=1e-2, lamb_ent_ops_end=1.0,
        lamb_l1_acts=1e-3, lamb_ent_acts=1e-6,
        update_grid=True, grid_update_num=10,
        refine_schedule=refine_schedule,
        verbose=epochs + 1,
    )
    with torch.no_grad():
        pred = model(dataset['test_input'])
    return r2_score(pred, dataset['test_label']), model


# ---------------------------------------------------------------------------
# MultKAN runner  (staged: grid 3 → 5 → 10)
# ---------------------------------------------------------------------------

def run_multkan(dataset, n_in, seed=0, smoke=False):
    torch.manual_seed(seed)
    n_mult = max(1, n_in // 2)
    width  = [[n_in, 0], [n_in, n_mult], [1, 0]]
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)

    steps_per_stage = 5 if smoke else 600

    def _fit_stage(model, steps, lr):
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

    model = MultKAN(
        width=width, grid=5, k=3, mult_arity=2, seed=seed,
        auto_save=False, ckpt_path=MULTKAN_CKPT, device=device,
    )
    _fit_stage(model, steps_per_stage, lr=0.01)

    if not smoke:
        train_x = dataset['train_input']
        model.update_grid_from_samples(train_x)
        model = model.refine(5)
        model.update_grid_from_samples(train_x)
        _fit_stage(model, steps_per_stage, lr=0.005)

        model.update_grid_from_samples(train_x)
        model = model.refine(5)
        model.update_grid_from_samples(train_x)
        _fit_stage(model, steps_per_stage, lr=0.001)

    with torch.no_grad():
        pred = model(dataset['test_input'])
    return r2_score(pred, dataset['test_label'])


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
    with torch.no_grad():
        pred = model(dataset['test_input'])
    return r2_score(pred, dataset['test_label'])


# ---------------------------------------------------------------------------
# PySR runner
# ---------------------------------------------------------------------------

def run_pysr(dataset, n_in, seed=0, smoke=False):
    if not HAS_PYSR:
        return None
    np.random.seed(seed)
    X = dataset['train_input'].cpu().numpy()
    y = dataset['train_label'].cpu().numpy().ravel()
    model = PySRRegressor(
        niterations=5 if smoke else 100,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=[],
        timeout_in_seconds=30 if smoke else 120,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)
    X_test = dataset['test_input'].cpu().numpy()
    y_test = dataset['test_label'].cpu().numpy().ravel()
    pred   = model.predict(X_test)
    valid  = np.isfinite(pred) & np.isfinite(y_test)
    if valid.sum() < 2:
        return None
    mse = float(np.mean((pred[valid] - y_test[valid]) ** 2))
    var = float(np.var(y_test[valid]))
    return 1.0 - mse / (var + 1e-10)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(stage_name, results, n_runs):
    print(f'\n  Summary — {stage_name}')
    print(f'  {"Model":<14}| {"R² mean":>8} | {"R² std":>7} | {"R²>0.99":>8}')
    print(f'  {"-" * 46}')
    for method, r2s in results.items():
        valid = [r for r in r2s if r is not None]
        if not valid:
            print(f'  {method:<14}| {"N/A":>8} | {"N/A":>7} | {"N/A":>8}')
            continue
        mean_r2   = float(np.mean(valid))
        std_r2    = float(np.std(valid))
        n_success = sum(r > 0.99 for r in valid)
        print(f'  {method:<14}| {mean_r2:>8.4f} | {std_r2:>7.4f} | {n_success:>4}/{len(valid)}')


def save_model(lokan_model, stage_idx, seed, n_in):
    os.makedirs(MODELS_DIR, exist_ok=True)
    tag     = f'sr_stage{stage_idx + 1}'
    pt_path = os.path.join(MODELS_DIR, f'{tag}_lokan.pt')
    meta    = {
        'tag':      tag,
        'n_in':     n_in,
        'width':    [n_in, n_in, 1],
        'grid':     lokan_model.grid,
        'k':        lokan_model.k,
        'seed':     seed,
    }
    torch.save(
        {'lokan_state': lokan_model.state_dict(), 'fc_state': None, 'config': meta},
        pt_path,
    )
    json_path = os.path.join(MODELS_DIR, f'{tag}_meta.json')
    with open(json_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    return pt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--no-pysr',   action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MULTKAN_CKPT, exist_ok=True)

    run_pysr_flag = HAS_PYSR and not args.no_pysr and not args.smoke_test
    stages = STAGES[:1] if args.smoke_test else STAGES
    n_runs = 1        if args.smoke_test else N_RUNS

    print(f'Device  : {device}')
    print(f'N_RUNS  : {n_runs}')
    print(f'N_TRAIN : {N_TRAIN}  N_TEST : {N_TEST}')
    print(f'PySR    : {"yes" if run_pysr_flag else "no"}')
    print()

    all_records = []

    for s_idx, stage in enumerate(stages):
        name    = stage['name']
        n_in    = stage['n_inputs']
        f       = stage['f']

        print(f'{"=" * 65}')
        print(f'{name}  (in_dim={n_in})')

        results = {'LOKAN': [], 'MultKAN': [], 'MLP': [], 'PySR': []}
        last_lokan = None

        for seed in range(n_runs):
            t0 = time.time()
            print(f'  seed={seed}', end='  ', flush=True)
            ds = make_dataset(f, n_in, seed=seed)

            r2_lokan, lokan_model = run_lokan(ds, n_in, seed=seed, smoke=args.smoke_test)
            last_lokan = lokan_model
            print(f'LOKAN={r2_lokan:.4f}', end='  ', flush=True)

            r2_mult = run_multkan(ds, n_in, seed=seed, smoke=args.smoke_test)
            print(f'MultKAN={r2_mult:.4f}', end='  ', flush=True)

            r2_mlp = run_mlp(ds, n_in, seed=seed, smoke=args.smoke_test)
            print(f'MLP={r2_mlp:.4f}', end='  ', flush=True)

            r2_pysr = run_pysr(ds, n_in, seed=seed, smoke=args.smoke_test) if run_pysr_flag else None
            if r2_pysr is not None:
                print(f'PySR={r2_pysr:.4f}', end='  ')

            print(f'({time.time() - t0:.0f}s)')

            results['LOKAN'].append(r2_lokan)
            results['MultKAN'].append(r2_mult)
            results['MLP'].append(r2_mlp)
            results['PySR'].append(r2_pysr)

            all_records.append({
                'stage': name, 'n_in': n_in, 'seed': seed,
                'r2_lokan': r2_lokan, 'r2_multkan': r2_mult,
                'r2_mlp': r2_mlp, 'r2_pysr': r2_pysr,
            })

        # Save last seed LOKAN model
        if last_lokan is not None:
            pt = save_model(last_lokan, s_idx, seed=n_runs - 1, n_in=n_in)
            print(f'  Saved LOKAN model → {os.path.relpath(pt, ROOT)}')

        print_table(name, results, n_runs)

    # ---- CSV output ----
    if not args.smoke_test:
        import csv
        csv_path = os.path.join(RESULTS_DIR, 'sr_staged_results.csv')
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f'\nCSV saved → {os.path.relpath(csv_path, ROOT)}')

        # ---- Markdown summary ----
        md_path = os.path.join(RESULTS_DIR, 'sr_staged_results.md')
        with open(md_path, 'w') as fh:
            fh.write('# Staged SR Benchmark\n\n')
            for stage in STAGES:
                fh.write(f'## {stage["name"]}\n\n')
                fh.write('| Model | R² mean | R² std | R²>0.99 |\n')
                fh.write('|-------|---------|--------|--------|\n')
                stage_recs = [r for r in all_records if r['stage'] == stage['name']]
                for col, key in [('LOKAN', 'r2_lokan'), ('MultKAN', 'r2_multkan'),
                                  ('MLP', 'r2_mlp'), ('PySR', 'r2_pysr')]:
                    vals = [r[key] for r in stage_recs if r[key] is not None]
                    if not vals:
                        fh.write(f'| {col} | N/A | N/A | N/A |\n')
                        continue
                    mean_r2 = float(np.mean(vals))
                    std_r2  = float(np.std(vals))
                    n_ok    = sum(v > 0.99 for v in vals)
                    fh.write(f'| {col} | {mean_r2:.4f} | {std_r2:.4f} | {n_ok}/{len(vals)} |\n')
                fh.write('\n')
        print(f'Markdown saved → {os.path.relpath(md_path, ROOT)}')


if __name__ == '__main__':
    main()
