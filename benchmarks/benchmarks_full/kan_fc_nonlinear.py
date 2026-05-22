"""
KAN + FC nonlinear regression benchmark — full version.

Compares LOKAN+FC against MLP on four nonlinear regression tasks, each
isolating a different type of nonlinearity.

Tasks
-----
T1 – Inverse pair   f = 1/x0 + 1/x1            [0.5, 2.0]²
T2 – Sine product   f = sin(2π x0)·sin(2π x1)  [0, 1]²
T3 – Exp + inverse  f = exp(x0) + 1/(0.5+x1)   [0, 1.5]²
T4 – Mixed 3-input  f = sin(2π x0) + 1/(0.5+x1) + x2²  mixed

Architecture
------------
LOKAN+FC : LOKAN([n, 4, 4]) → Sequential(Linear→SiLU→Linear→SiLU→Linear)
MLP      : Sequential([n, 12, 12, 6, 1]) — comparable parameter count

Training (LOKAN+FC) mirrors LOKAN.fit() with MSE loss:
  - Staged grid: 3 → 5 (epoch 40) → 10 (epoch 80), total 120 epochs
  - Exponential schedules for lr, tau, lambda
  - Entropy + sparsity regularization on LOKAN parts only

Interpretability (per task, last seed):
  - nonlinearity_score: 1 - R² of linear fit to learned spline
  - best_matching_fn: highest Pearson |r| among {sin, 1/x, exp, x², linear}

Results saved to benchmarks/results/{kanfc_results.csv, models/}.

Run full benchmark:
    python benchmarks/benchmarks_full/kan_fc_nonlinear.py

Smoke-test (1 task, 1 seed, 5 epochs):
    python benchmarks/benchmarks_full/kan_fc_nonlinear.py --smoke-test
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT)

from lokan import LOKAN
from lokan.LOKANLayer import LOKANLayer
from lokan.spline import coef2curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 3
N_TRAIN = 1000
N_TEST  = 200
H_KAN   = 4
H_FC    = 4

RESULTS_DIR = os.path.join(ROOT, 'benchmarks', 'results')
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        'tag':          'T1',
        'name':         'T1 – Inverse pair',
        'description':  'f = 1/x0 + 1/x1',
        'n_inputs':     2,
        'ranges':       [[0.5, 2.0], [0.5, 2.0]],
        'f':            lambda x: 1.0 / x[:, 0] + 1.0 / x[:, 1],
        'ground_truth': ['1/x', '1/x'],
        'expected_ops': 'summation — two independent inverse terms',
    },
    {
        'tag':          'T2',
        'name':         'T2 – Sine product',
        'description':  'f = sin(2π x0)·sin(2π x1)',
        'n_inputs':     2,
        'ranges':       [[0.0, 1.0], [0.0, 1.0]],
        'f':            lambda x: (torch.sin(2 * torch.pi * x[:, 0])
                                   * torch.sin(2 * torch.pi * x[:, 1])),
        'ground_truth': ['sin', 'sin'],
        'expected_ops': 'multiplication — both inputs feed one product group',
    },
    {
        'tag':          'T3',
        'name':         'T3 – Exp + inverse',
        'description':  'f = exp(x0) + 1/(0.5+x1)',
        'n_inputs':     2,
        'ranges':       [[0.0, 1.5], [0.0, 1.5]],
        'f':            lambda x: torch.exp(x[:, 0]) + 1.0 / (0.5 + x[:, 1]),
        'ground_truth': ['exp(x)', '1/x'],
        'expected_ops': 'summation — distinct nonlinear activations per input',
    },
    {
        'tag':          'T4',
        'name':         'T4 – Mixed 3-input',
        'description':  'f = sin(2π x0) + 1/(0.5+x1) + x2²',
        'n_inputs':     3,
        'ranges':       [[0.0, 1.0], [0.0, 1.5], [0.0, 1.5]],
        'f':            lambda x: (torch.sin(2 * torch.pi * x[:, 0])
                                   + 1.0 / (0.5 + x[:, 1])
                                   + x[:, 2] ** 2),
        'ground_truth': ['sin', '1/x', 'x²'],
        'expected_ops': 'summation — three structurally distinct terms',
    },
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(task, seed=0):
    torch.manual_seed(seed)
    n      = task['n_inputs']
    ranges = task['ranges']
    lo     = torch.tensor([r[0] for r in ranges])
    hi     = torch.tensor([r[1] for r in ranges])
    total  = N_TRAIN + N_TEST
    x      = torch.rand(total, n) * (hi - lo) + lo
    y      = task['f'](x)
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
# fit_lokan_fc — standalone training loop for LOKAN + FC hybrid
# (identical implementation to feature_ext_classification.py)
# ---------------------------------------------------------------------------

def _update_epoch_set(seg_start, seg_stop, n):
    window = max(seg_stop - seg_start, 1)
    freq   = max(window // n, 1)
    return {seg_start + k * freq
            for k in range(n)
            if seg_start + k * freq < seg_stop}


def fit_lokan_fc(
    lokan,
    fc,
    dataset,
    loss_fn,
    epochs=120,
    lr_start=0.2,
    lr_end=1e-3,
    batch_size=64,
    tau_start=2.0,
    tau_end=0.1,
    lamb_ent_ops_start=1e-2,
    lamb_ent_ops_end=1.0,
    lamb_l1_acts=1e-3,
    lamb_ent_acts=1e-6,
    update_grid=True,
    grid_update_num=10,
    refine_schedule=None,
    verbose=0,
):
    """Train LOKAN feature extractor + arbitrary FC head jointly.

    Returns epoch_r2: list[float] — R² on the test set after each epoch.
    """
    refine_dict = dict(refine_schedule) if refine_schedule else {}

    if update_grid:
        if refine_dict:
            boundaries     = [0] + sorted(refine_dict) + [epochs]
            _update_epochs = set()
            for seg_s, seg_e in zip(boundaries, boundaries[1:]):
                half = seg_s + (seg_e - seg_s) // 2
                _update_epochs |= _update_epoch_set(seg_s, half, grid_update_num)
        else:
            _update_epochs = _update_epoch_set(0, epochs // 2, grid_update_num)
    else:
        _update_epochs = set()

    with torch.no_grad():
        lokan.update_grid(dataset['train_input'])

    all_params = list(lokan.parameters()) + list(fc.parameters())
    optimizer  = optim.Adam(all_params, lr=lr_start)
    epoch_r2   = []

    for epoch in range(epochs):
        t = epoch / max(epochs - 1, 1)

        if epoch in refine_dict:
            new_g     = refine_dict[epoch]
            model_new = LOKAN(
                width=[list(w) for w in lokan.width],
                grid=new_g, k=lokan.k,
                base_fun=lokan.base_fun_name,
                symbolic_enabled=lokan.symbolic_enabled,
                affine_trainable=lokan.affine_trainable,
                grid_eps=lokan.grid_eps,
                grid_range=lokan.grid_range,
                sp_trainable=lokan.sp_trainable,
                sb_trainable=lokan.sb_trainable,
                save_act=lokan.save_act,
                device=lokan.device,
            )
            model_new.initialize_from_another_model(lokan, lokan.cache_data)
            lokan.act_fun = model_new.act_fun
            lokan.grid    = new_g
            cur_lr        = lr_start * (lr_end / lr_start) ** t
            all_params    = list(lokan.parameters()) + list(fc.parameters())
            optimizer     = optim.Adam(all_params, lr=cur_lr)
            if verbose:
                print(f'  [refine] epoch {epoch}: grid → {new_g}')

        new_lr   = lr_start           * (lr_end           / lr_start)           ** t
        new_tau  = tau_start          * (tau_end          / tau_start)          ** t
        new_lamb = lamb_ent_ops_start * (lamb_ent_ops_end / lamb_ent_ops_start) ** t

        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        for m in lokan.modules():
            if isinstance(m, LOKANLayer):
                m.tau = new_tau

        if epoch in _update_epochs:
            with torch.no_grad():
                lokan.update_grid(dataset['train_input'])

        n_train = dataset['train_input'].shape[0]
        perm    = torch.randperm(n_train)

        for start in range(0, n_train, batch_size):
            idx     = perm[start:start + batch_size]
            batch_x = dataset['train_input'][idx]
            batch_y = dataset['train_label'][idx]

            out       = fc(lokan(batch_x))
            task_loss = loss_fn(out, batch_y)

            ent_ops, n_ops = 0.0, 0
            for m in lokan.modules():
                if isinstance(m, LOKANLayer):
                    probs    = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    n_ops   += 1
            ent_ops_avg = ent_ops / n_ops if n_ops else 0.0

            l1_acts, ent_acts, n_al = 0.0, 0.0, 0
            for acts in lokan.acts_scale_spline:
                l1_acts += torch.sum(acts)
                p_row    = acts / (acts.sum(dim=1, keepdim=True) + 1e-8)
                p_col    = acts / (acts.sum(dim=0, keepdim=True) + 1e-8)
                ent_acts += (
                    -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-8), dim=1))
                    - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-8), dim=0))
                )
                n_al += 1
            l1_acts  = l1_acts  / n_al if n_al else 0.0
            ent_acts = ent_acts / n_al if n_al else 0.0

            total = (task_loss
                     + new_lamb      * ent_ops_avg
                     + lamb_l1_acts  * l1_acts
                     + lamb_ent_acts * ent_acts)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        with torch.no_grad():
            pred = fc(lokan(dataset['test_input']))
            epoch_r2.append(r2_score(pred, dataset['test_label']))

    with torch.no_grad():
        lokan(dataset['train_input'])

    return epoch_r2


# ---------------------------------------------------------------------------
# MLP regressor (comparable capacity to LOKAN+FC)
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, n_inputs, hidden=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Parameter / step counter
# ---------------------------------------------------------------------------

def count_params(*models):
    """Total trainable parameter count across one or more nn.Module objects."""
    return sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)


def find_mlp_hidden(n_inputs, target_params):
    """Smallest hidden size h for [n→h→h→1] MLP with params ≥ target_params."""
    for h in range(1, 2000):
        p = n_inputs * h + h + h * h + h + h + 1
        if p >= target_params:
            return h
    return 2000


def _model_info_table(n, epochs, batch_size=64):
    """Build dummy models, match MLP params to LOKAN+FC, print info table.

    Returns the MLP hidden size chosen so that params(MLP) ≥ params(LOKAN+FC).
    """
    import math
    _lokan = LOKAN(width=[n, H_KAN, H_KAN], grid=5, k=3, seed=0, device='cpu')
    _fc    = nn.Sequential(
        nn.Linear(H_KAN, H_FC), nn.SiLU(),
        nn.Linear(H_FC, H_FC // 2), nn.SiLU(),
        nn.Linear(H_FC // 2, 1),
    )
    n_lokan_fc = count_params(_lokan, _fc)

    h_mlp = find_mlp_hidden(n, n_lokan_fc)
    _mlp  = MLPRegressor(n_inputs=n, hidden=h_mlp)
    n_mlp = count_params(_mlp)

    steps_lokan = int(epochs * N_TRAIN / batch_size)
    steps_mlp   = epochs * (N_TRAIN // batch_size)

    print(f'  {"Model":<14}| {"Params":>8} | {"Steps":>7} | {"Hidden"}')
    print(f'  {"-" * 48}')
    print(f'  {"LOKAN+FC":<14}| {n_lokan_fc:>8,} | {steps_lokan:>7,} | {H_KAN}→{H_FC}→{H_FC//2}→1')
    print(f'  {"MLP [n,h,h,1]":<14}| {n_mlp:>8,} | {steps_mlp:>7,} | h={h_mlp}')

    return h_mlp


def train_mlp(model, dataset, epochs=120, smoke=False):
    _epochs  = 5 if smoke else epochs
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    epoch_r2  = []

    for epoch in range(_epochs):
        t      = epoch / max(_epochs - 1, 1)
        new_lr = 0.01 * (1e-4 / 0.01) ** t
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // 64, 1)):
            bx = dataset['train_input'][idx[bi * 64:(bi + 1) * 64]]
            by = dataset['train_label'][idx[bi * 64:(bi + 1) * 64]]
            out  = model(bx)
            loss = criterion(out, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(dataset['test_input'])
            epoch_r2.append(r2_score(pred, dataset['test_label']))

    return epoch_r2


# ---------------------------------------------------------------------------
# Interpretability probe
# ---------------------------------------------------------------------------

def _eval_spline(layer: LOKANLayer, input_idx: int, n_pts: int = 200):
    k   = layer.k
    lo  = layer.grid[input_idx, k].item()
    hi  = layer.grid[input_idx, -(k + 1)].item()

    x_probe = torch.linspace(lo, hi, n_pts).to(device)
    x_batch = torch.zeros(n_pts, layer.in_dim, device=device)
    x_batch[:, input_idx] = x_probe

    with torch.no_grad():
        spline_all = coef2curve(x_batch, layer.grid, layer.coef, layer.k)
        spline_i   = spline_all[:, input_idx, :].mean(dim=1)
        base_i     = layer.base_fun(x_probe)
        scale_sp   = layer.scale_sp[input_idx].mean()
        scale_base = layer.scale_base[input_idx].mean()
        y = spline_i * scale_sp + base_i * scale_base

    return x_probe.cpu().numpy(), y.cpu().numpy()


def nonlinearity_score(x_np, y_np):
    coeffs = np.polyfit(x_np, y_np, 1)
    y_lin  = np.polyval(coeffs, x_np)
    ss_res = np.sum((y_np - y_lin) ** 2)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2) + 1e-10
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def best_matching_fn(x_np, y_np):
    def _norm(v):
        return (v - v.mean()) / (v.std() + 1e-8)
    x_c   = x_np - x_np.mean()
    x_pos = np.abs(x_np) + 1e-4
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


def probe_nonlinearities(lokan_model, task):
    layer = next((m for m in lokan_model.modules() if isinstance(m, LOKANLayer)), None)
    if layer is None:
        return
    ground_truth = task['ground_truth']
    n_inputs     = task['n_inputs']
    n_detected   = 0
    print(f'  Nonlinearity detection (layer 0, {n_inputs} inputs):')
    print(f'  {"Input":<8}{"GT":>8}{"nl-score":>10}{"best-match":>12}{"  |r|":>7}{"detected":>10}')
    print(f'  {"-" * 57}')
    for i in range(n_inputs):
        x_np, y_np  = _eval_spline(layer, i)
        nl_score    = nonlinearity_score(x_np, y_np)
        match_name, match_r = best_matching_fn(x_np, y_np)
        gt          = ground_truth[i] if i < len(ground_truth) else '?'
        detected    = (match_name == gt)
        if detected:
            n_detected += 1
        flag = 'OK' if detected else 'X'
        print(f'  x{i:<7}{gt:>8}{nl_score:>10.3f}{match_name:>12}{match_r:>7.3f}{flag:>10}')
    print(f'  Detection: {n_detected}/{n_inputs}  |  Expected: {task["expected_ops"]}')


def probe_operations(lokan_model):
    layers = [m for m in lokan_model.modules() if isinstance(m, LOKANLayer)]
    for l_idx, layer in enumerate(layers):
        with torch.no_grad():
            probs    = torch.softmax(layer.logits / 0.05, dim=-1)
            ent      = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            assign   = probs.argmax(dim=-1).cpu()
            n_groups = layer.logits.shape[-1] - 1
            n_mult   = (assign < n_groups).sum().item()
            n_sum    = (assign == n_groups).sum().item()
        total = n_mult + n_sum
        print(f'  Layer {l_idx}:  ent={ent:.3f} nats  |  '
              f'mult={n_mult}/{total}  sum={n_sum}/{total}')


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_lokan_fc(lokan, fc, task, seed):
    os.makedirs(MODELS_DIR, exist_ok=True)
    n_in   = task['n_inputs']
    tag    = f'kanfc_{task["tag"]}'
    config = {
        'tag':          tag,
        'task':         task['name'],
        'n_in':         n_in,
        'lokan_width':  [n_in, H_KAN, H_KAN],
        'lokan_grid':   lokan.grid,
        'lokan_k':      lokan.k,
        'h_fc':         H_FC,
        'seed':         seed,
    }
    pt_path = os.path.join(MODELS_DIR, f'{tag}_lokan_fc.pt')
    torch.save(
        {
            'lokan_state': lokan.state_dict(),
            'fc_state':    fc.state_dict(),
            'config':      config,
        },
        pt_path,
    )
    json_path = os.path.join(MODELS_DIR, f'{tag}_meta.json')
    with open(json_path, 'w') as fh:
        json.dump(config, fh, indent=2)
    return pt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tasks  = TASKS[:1] if args.smoke_test else TASKS
    n_runs = 1         if args.smoke_test else N_RUNS
    epochs = 5         if args.smoke_test else 120

    print(f'Device  : {device}')
    print(f'N_RUNS  : {n_runs}  EPOCHS: {epochs}')
    print(f'Architecture: LOKAN([n→{H_KAN}→{H_KAN}]) + FC([{H_KAN}→{H_FC}→{H_FC//2}→1])')
    print(f'MLP baseline: [n, h, h, 1]  (h chosen per task to match LOKAN+FC param count)')
    print()

    all_records = []
    mse_fn = nn.MSELoss()

    for task in tasks:
        n = task['n_inputs']
        print(f'{"=" * 65}')
        print(f'{task["name"]}   {task["description"]}   (in_dim={n})')
        h_mlp = _model_info_table(n, epochs)

        results = {
            'LOKAN+FC': {'r2': [], 'ep95': []},
            'MLP':      {'r2': [], 'ep95': []},
        }
        last_lokan = None
        last_fc    = None

        for seed in range(n_runs):
            t0 = time.time()
            print(f'  seed={seed}', end='  ', flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            ds = make_dataset(task, seed=seed)

            # ---- LOKAN + FC ----
            lokan = LOKAN(
                width=[n, H_KAN, H_KAN], grid=5, k=3,
                seed=seed, device=device,
            )
            fc = nn.Sequential(
                nn.Linear(H_KAN, H_FC),
                nn.SiLU(),
                nn.Linear(H_FC, H_FC // 2),
                nn.SiLU(),
                nn.Linear(H_FC // 2, 1),
            ).to(device)
            refine = None if args.smoke_test else [(40, 5), (80, 10)]
            r2_curve = fit_lokan_fc(
                lokan, fc, ds,
                loss_fn=lambda out, y: mse_fn(out, y),
                epochs=epochs,
                refine_schedule=refine,
                update_grid=True, grid_update_num=10,
            )
            final_r2 = r2_curve[-1]
            ep95     = next((i + 1 for i, r in enumerate(r2_curve) if r >= 0.95), None)
            results['LOKAN+FC']['r2'].append(final_r2)
            results['LOKAN+FC']['ep95'].append(ep95)
            last_lokan, last_fc = lokan, fc
            print(f'LOKAN+FC R²={final_r2:.4f}', end='  ', flush=True)

            # ---- MLP ----
            mlp = MLPRegressor(n_inputs=n, hidden=h_mlp).to(device)
            r2_mlp = train_mlp(mlp, ds, epochs=epochs, smoke=args.smoke_test)
            final_mlp = r2_mlp[-1]
            ep95_mlp  = next((i + 1 for i, r in enumerate(r2_mlp) if r >= 0.95), None)
            results['MLP']['r2'].append(final_mlp)
            results['MLP']['ep95'].append(ep95_mlp)
            print(f'MLP R²={final_mlp:.4f}  ({time.time() - t0:.0f}s)')

            all_records.append({
                'task': task['name'], 'n_in': n, 'seed': seed,
                'r2_lokan_fc': final_r2,
                'r2_mlp':      final_mlp,
                'ep95_lokan_fc': ep95,
                'ep95_mlp':     ep95_mlp,
            })

        # Interpretability on last seed
        if last_lokan is not None and not args.smoke_test:
            print()
            probe_nonlinearities(last_lokan, task)
            probe_operations(last_lokan)

        # Save last seed model
        if last_lokan is not None:
            pt = save_lokan_fc(last_lokan, last_fc, task, seed=n_runs - 1)
            print(f'  Saved LOKAN+FC model → {os.path.relpath(pt, ROOT)}')

        # Per-task summary
        print(f'\n  Summary — {task["name"]}')
        print(f'  {"Model":<14}| {"R² mean":>8} | {"R² std":>7} | {"ep→R²≥0.95 mean":>16}')
        print(f'  {"-" * 54}')
        for m_name, res in results.items():
            r2s   = res['r2']
            ep95s = [e for e in res['ep95'] if e is not None]
            mean_ep = float(np.mean(ep95s)) if ep95s else float('nan')
            print(f'  {m_name:<14}| {np.mean(r2s):>8.4f} | {np.std(r2s):>7.4f} | {mean_ep:>16.1f}')

    # ---- CSV output ----
    if not args.smoke_test and all_records:
        csv_path = os.path.join(RESULTS_DIR, 'kanfc_results.csv')
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f'\nCSV saved → {os.path.relpath(csv_path, ROOT)}')

        md_path = os.path.join(RESULTS_DIR, 'kanfc_results.md')
        with open(md_path, 'w') as fh:
            fh.write('# KAN+FC Nonlinear Benchmark\n\n')
            for task in TASKS:
                fh.write(f'## {task["name"]}\n\n')
                fh.write('| Model | R² mean | R² std | ep→R²≥0.95 |\n')
                fh.write('|-------|---------|--------|------------|\n')
                recs = [r for r in all_records if r['task'] == task['name']]
                for col, r2_key, ep_key in [
                    ('LOKAN+FC', 'r2_lokan_fc', 'ep95_lokan_fc'),
                    ('MLP',      'r2_mlp',      'ep95_mlp'),
                ]:
                    r2s = [r[r2_key] for r in recs]
                    eps = [r[ep_key]  for r in recs if r[ep_key] is not None]
                    mean_ep = f'{np.mean(eps):.1f}' if eps else 'N/A'
                    fh.write(f'| {col} | {np.mean(r2s):.4f} | {np.std(r2s):.4f} | {mean_ep} |\n')
                fh.write('\n')
        print(f'Markdown saved → {os.path.relpath(md_path, ROOT)}')


if __name__ == '__main__':
    main()
