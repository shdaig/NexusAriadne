"""
Feature extraction classification benchmark — full version.

Compares LOKAN+FC (feature extractor with a linear classification head) against
MLP baselines on three binary classification tasks.

Tasks
-----
T1 – Additive-ratio:     x0/x1 + x2/x3 > 2.2              (4 inputs)
T2 – Product-of-sums:    (x0 + x1) * (x2 + x3) > 0.49     (4 inputs)
T3 – Two-product-groups: x0*x1*x2 + x3*x4*x5 > 0.18       (6 inputs)

Architecture
------------
LOKAN+FC : LOKAN([n, n, n]) → Linear(n→1) → Sigmoid
MLP-small: Linear chain [n, n, 1] + Sigmoid
MLP-large: Linear chain [n, 4n, 2n, 1] + Sigmoid

Training (LOKAN+FC) mirrors LOKAN.fit() with BC-loss and FC included:
  - Staged grid: 3 → 5 (epoch 40) → 10 (epoch 80), total 120 epochs
  - Exponential schedules for lr, tau, lambda
  - Entropy regularization on LOKAN operation logits
  - L1 + entropy regularization on LOKAN spline activation scales

Interpretability (per task, last seed)
---------------------------------------
  decode_operations() : argmax of logits at τ=0.05
  feature_importance() : acts_scale_spline[0].abs().mean(dim=1)

Results saved to benchmarks/results/{clf_results.csv, models/}.

Run full benchmark:
    python benchmarks/benchmarks_full/feature_ext_classification.py

Smoke-test (1 task, 1 seed, 5 epochs):
    python benchmarks/benchmarks_full/feature_ext_classification.py --smoke-test
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 3
N_TRAIN = 1000
N_TEST  = 200
LO, HI  = 0.1, 1.0

RESULTS_DIR = os.path.join(ROOT, 'benchmarks', 'results')
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        'tag':         'T1',
        'name':        'T1 – Additive-ratio',
        'description': 'x0/x1 + x2/x3 > 2.2',
        'n_inputs':    4,
        'f':           lambda x: x[:, 0] / x[:, 1] + x[:, 2] / x[:, 3],
        'threshold':   2.2,
        'expected_ops': 'summation of ratio terms',
    },
    {
        'tag':         'T2',
        'name':        'T2 – Product-of-sums',
        'description': '(x0+x1)*(x2+x3) > 0.49',
        'n_inputs':    4,
        'f':           lambda x: (x[:, 0] + x[:, 1]) * (x[:, 2] + x[:, 3]),
        'threshold':   0.49,
        'expected_ops': 'multiplication of summed pairs',
    },
    {
        'tag':         'T3',
        'name':        'T3 – Two-product-groups',
        'description': 'x0*x1*x2 + x3*x4*x5 > 0.18',
        'n_inputs':    6,
        'f':           lambda x: x[:, 0] * x[:, 1] * x[:, 2] + x[:, 3] * x[:, 4] * x[:, 5],
        'threshold':   0.18,
        'expected_ops': 'two independent product groups summed',
    },
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(task, seed=0):
    torch.manual_seed(seed)
    n     = task['n_inputs']
    x     = (HI - LO) * torch.rand(N_TRAIN + N_TEST, n) + LO
    y_raw = task['f'](x)
    y     = (y_raw > task['threshold']).float()
    return {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].unsqueeze(1).to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].unsqueeze(1).to(device),
    }


# ---------------------------------------------------------------------------
# fit_lokan_fc — standalone training loop for LOKAN + FC hybrid
#
# Mirrors LOKAN.fit() with three extensions:
#   1. optimizer covers both LOKAN and FC parameters
#   2. loss_fn is passed in (allows BCE instead of MSE)
#   3. refine_schedule triggers inline grid refinement (same as LOKAN.fit)
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

    The training loop is identical to LOKAN.fit() except:
      - The optimizer includes both lokan.parameters() and fc.parameters().
      - The task loss comes from loss_fn(fc(lokan(x)), y) instead of MSE.
      - Regularization (entropy on logits, L1+entropy on spline scales) is
        computed only from LOKAN layers — FC is trained only via task loss.

    Returns epoch_metric: list[float], one value per epoch (evaluated on
    the full test set using the task loss, converted to accuracy when the
    output is binary classification).
    """
    refine_dict = dict(refine_schedule) if refine_schedule else {}

    if update_grid:
        if refine_dict:
            boundaries   = [0] + sorted(refine_dict) + [epochs]
            _update_epochs = set()
            for seg_s, seg_e in zip(boundaries, boundaries[1:]):
                half = seg_s + (seg_e - seg_s) // 2
                _update_epochs |= _update_epoch_set(seg_s, half, grid_update_num)
        else:
            _update_epochs = _update_epoch_set(0, epochs // 2, grid_update_num)
    else:
        _update_epochs = set()

    # Pre-fit grid to data distribution
    with torch.no_grad():
        lokan.update_grid(dataset['train_input'])

    all_params = list(lokan.parameters()) + list(fc.parameters())
    optimizer  = optim.Adam(all_params, lr=lr_start)
    epoch_acc  = []

    for epoch in range(epochs):
        t = epoch / max(epochs - 1, 1)

        # Inline grid refinement
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

        # Exponential schedules
        new_lr   = lr_start           * (lr_end           / lr_start)           ** t
        new_tau  = tau_start          * (tau_end          / tau_start)          ** t
        new_lamb = lamb_ent_ops_start * (lamb_ent_ops_end / lamb_ent_ops_start) ** t

        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        for m in lokan.modules():
            if isinstance(m, LOKANLayer):
                m.tau = new_tau

        # Adaptive grid update
        if epoch in _update_epochs:
            with torch.no_grad():
                lokan.update_grid(dataset['train_input'])

        # Mini-batch loop
        n_train = dataset['train_input'].shape[0]
        perm    = torch.randperm(n_train)

        for start in range(0, n_train, batch_size):
            idx     = perm[start:start + batch_size]
            batch_x = dataset['train_input'][idx]
            batch_y = dataset['train_label'][idx]

            out        = fc(lokan(batch_x))
            task_loss  = loss_fn(out, batch_y)

            # Entropy regularization on operation logits (LOKAN only)
            ent_ops, n_ops = 0.0, 0
            for m in lokan.modules():
                if isinstance(m, LOKANLayer):
                    probs    = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    n_ops   += 1
            ent_ops_avg = ent_ops / n_ops if n_ops else 0.0

            # Sparsity regularization on spline activations (LOKAN only)
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

        # Per-epoch accuracy on test set
        with torch.no_grad():
            pred = fc(lokan(dataset['test_input']))
            acc  = ((pred > 0.5).float() == dataset['test_label']).float().mean().item()
            epoch_acc.append(acc)

    # Populate acts_scale_spline for interpretability probe
    with torch.no_grad():
        lokan(dataset['train_input'])

    return epoch_acc


# ---------------------------------------------------------------------------
# MLP classifier (plain FC, 2 hidden layers)
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, n_inputs, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Parameter / step counter and MLP size matcher
# ---------------------------------------------------------------------------

def count_params(*models):
    """Total trainable parameter count across one or more nn.Module objects."""
    return sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)


def find_mlp_hidden(n_inputs, target_params):
    """Smallest hidden size h for [n→h→h→1]+Sigmoid with params ≥ target_params."""
    for h in range(1, 2000):
        p = n_inputs * h + h + h * h + h + h + 1
        if p >= target_params:
            return h
    return 2000


def _model_info_table(n, epochs, batch_size=64):
    """Build dummy models, match MLP params to LOKAN+FC, print info table.

    Returns the MLP hidden size chosen so that params(MLP) ≥ params(LOKAN+FC).
    Metric: Accuracy (binary classification).
    """
    import math
    _lokan = LOKAN(width=[n, n, n], grid=5, k=3, seed=0, device='cpu')
    _fc    = nn.Sequential(nn.Linear(n, 1), nn.Sigmoid())
    n_lokan_fc = count_params(_lokan, _fc)

    h_mlp = find_mlp_hidden(n, n_lokan_fc)
    _mlp  = MLPClassifier(n_inputs=n, hidden=h_mlp)
    n_mlp = count_params(_mlp)

    steps_lokan = epochs * math.ceil(N_TRAIN / batch_size)
    steps_mlp   = epochs * (N_TRAIN // batch_size)

    print(f'  Metric : Accuracy  |  N_TRAIN={N_TRAIN}  N_TEST={N_TEST}')
    print(f'  {"Model":<14}| {"Params":>8} | {"Steps":>7} | {"Hidden"}')
    print(f'  {"-" * 48}')
    print(f'  {"LOKAN+FC":<14}| {n_lokan_fc:>8,} | {steps_lokan:>7,} | LOKAN([n,n,n])+Linear(n→1)')
    print(f'  {"MLP [n,h,h,1]":<14}| {n_mlp:>8,} | {steps_mlp:>7,} | h={h_mlp}')

    return h_mlp


def train_mlp(model, dataset, epochs=120, smoke=False):
    _epochs = 5 if smoke else epochs
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    epoch_acc = []

    for epoch in range(_epochs):
        t = epoch / max(_epochs - 1, 1)
        new_lr = 0.1 * (1e-3 / 0.1) ** t
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
            acc  = ((pred > 0.5).float() == dataset['test_label']).float().mean().item()
            epoch_acc.append(acc)

    return epoch_acc


# ---------------------------------------------------------------------------
# Interpretability probe (adapted from NexusKAN version for LOKAN)
# ---------------------------------------------------------------------------

def decode_operations(layer: LOKANLayer):
    with torch.no_grad():
        probs       = torch.softmax(layer.logits / 0.05, dim=-1)
        assignments = probs.argmax(dim=-1)
        ent         = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        ent_mean    = ent.mean().item()
        n_groups    = layer.logits.shape[-1] - 1
        n_mult      = (assignments < n_groups).sum().item()
        n_sum       = (assignments == n_groups).sum().item()
    return assignments.cpu(), ent_mean, n_mult, n_sum


def feature_importance(lokan_model):
    if not lokan_model.acts_scale_spline:
        return None
    return lokan_model.acts_scale_spline[0].abs().mean(dim=1).cpu()


def print_interpretability(lokan_model, task):
    print(f'\n  -- Interpretability probe --')
    print(f'  Expected structure: {task["expected_ops"]}')
    layers = [m for m in lokan_model.modules() if isinstance(m, LOKANLayer)]
    for l_idx, layer in enumerate(layers):
        assignments, ent_mean, n_mult, n_sum = decode_operations(layer)
        total = n_mult + n_sum
        print(f'  Layer {l_idx}:  logit entropy={ent_mean:.3f} nats  '
              f'|  mult={n_mult}/{total}  sum={n_sum}/{total}')
        in_d, out_d = assignments.shape
        for j in range(out_d):
            col     = assignments[:, j].tolist()
            n_groups = layer.logits.shape[-1] - 1
            ops     = ['M' + str(c) if c < n_groups else 'S' for c in col]
            print(f'    out[{j}]: inputs → [{", ".join(ops)}]')
    scores = feature_importance(lokan_model)
    if scores is not None:
        n_in   = task['n_inputs']
        ranked = scores.argsort(descending=True)
        parts  = [f'x{ranked[k].item()}={scores[ranked[k]].item():.3f}'
                  for k in range(min(n_in, scores.numel()))]
        print(f'  Feature importance (layer 0, ranked): {" | ".join(parts)}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def epochs_to_threshold(epoch_accs, threshold=0.90):
    for i, acc in enumerate(epoch_accs):
        if acc >= threshold:
            return i + 1
    return None


def save_lokan_fc(lokan, fc, task, seed):
    os.makedirs(MODELS_DIR, exist_ok=True)
    n_in = task['n_inputs']
    tag  = f'clf_{task["tag"]}'
    config = {
        'tag':    tag,
        'task':   task['name'],
        'n_in':   n_in,
        'lokan_width': [n_in, n_in, n_in],
        'lokan_grid':  lokan.grid,
        'lokan_k':     lokan.k,
        'fc_out':      1,
        'seed':        seed,
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
    print(f'N_TRAIN : {N_TRAIN}  N_TEST: {N_TEST}')
    print()

    all_records = []

    for task in tasks:
        n = task['n_inputs']
        print(f'{"=" * 65}')
        print(f'{task["name"]}   {task["description"]}   (in_dim={n})')
        h_mlp = _model_info_table(n, epochs)

        results = {
            'LOKAN+FC': {'acc': [], 'ep90': []},
            'MLP':      {'acc': [], 'ep90': []},
        }
        last_lokan = None
        last_fc    = None

        for seed in range(n_runs):
            t0 = time.time()
            print(f'  seed={seed}', end='  ', flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            ds = make_dataset(task, seed=seed)
            bce = nn.BCELoss()

            # ---- LOKAN + FC ----
            lokan = LOKAN(
                width=[n, n, n], grid=5, k=3,
                grid_range=[LO, HI],
                seed=seed, device=device,
            )
            fc = nn.Sequential(nn.Linear(n, 1), nn.Sigmoid()).to(device)
            refine = None if args.smoke_test else [(40, 5), (80, 10)]
            acc_curve = fit_lokan_fc(
                lokan, fc, ds,
                loss_fn=lambda out, y: bce(out, y),
                epochs=epochs,
                refine_schedule=refine,
                update_grid=True, grid_update_num=10,
            )
            final_acc = acc_curve[-1]
            ep90      = epochs_to_threshold(acc_curve)
            results['LOKAN+FC']['acc'].append(final_acc)
            results['LOKAN+FC']['ep90'].append(ep90)
            last_lokan, last_fc = lokan, fc
            print(f'LOKAN+FC={final_acc:.4f}', end='  ', flush=True)

            # ---- MLP (matched params) ----
            mlp = MLPClassifier(n_inputs=n, hidden=h_mlp).to(device)
            acc_m = train_mlp(mlp, ds, epochs=epochs, smoke=args.smoke_test)
            results['MLP']['acc'].append(acc_m[-1])
            results['MLP']['ep90'].append(epochs_to_threshold(acc_m))
            print(f'MLP={acc_m[-1]:.4f}  ({time.time() - t0:.0f}s)')

            all_records.append({
                'task': task['name'], 'n_in': n, 'seed': seed,
                'acc_lokan_fc': final_acc,
                'acc_mlp':      acc_m[-1],
                'ep90_lokan_fc': ep90,
                'ep90_mlp':     epochs_to_threshold(acc_m),
            })

        # Interpretability on last seed
        if last_lokan is not None and not args.smoke_test:
            print_interpretability(last_lokan, task)

        # Save last seed model
        if last_lokan is not None:
            pt = save_lokan_fc(last_lokan, last_fc, task, seed=n_runs - 1)
            print(f'  Saved LOKAN+FC model → {os.path.relpath(pt, ROOT)}')

        # Per-task summary
        print(f'\n  Summary — {task["name"]}  (metric: Accuracy)')
        print(f'  {"Model":<14}| {"Acc mean":>9} | {"Acc std":>8} | {"Ep→90% mean":>13}')
        print(f'  {"-" * 53}')
        for m_name, res in results.items():
            accs  = res['acc']
            ep90s = [e for e in res['ep90'] if e is not None]
            mean_ep = float(np.mean(ep90s)) if ep90s else float('nan')
            print(f'  {m_name:<14}| {np.mean(accs):>9.4f} | {np.std(accs):>8.4f} | {mean_ep:>13.1f}')

    # ---- CSV output ----
    if not args.smoke_test and all_records:
        csv_path = os.path.join(RESULTS_DIR, 'clf_results.csv')
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f'\nCSV saved → {os.path.relpath(csv_path, ROOT)}')

        md_path = os.path.join(RESULTS_DIR, 'clf_results.md')
        with open(md_path, 'w') as fh:
            fh.write('# Classification Benchmark  (metric: Accuracy)\n\n')
            for task in TASKS:
                fh.write(f'## {task["name"]}\n\n')
                fh.write('| Model | Acc mean | Acc std | Ep→90% |\n')
                fh.write('|-------|----------|---------|--------|\n')
                recs = [r for r in all_records if r['task'] == task['name']]
                for col, acc_key, ep_key in [
                    ('LOKAN+FC', 'acc_lokan_fc', 'ep90_lokan_fc'),
                    ('MLP',      'acc_mlp',      'ep90_mlp'),
                ]:
                    accs = [r[acc_key] for r in recs]
                    eps  = [r[ep_key]  for r in recs if r[ep_key] is not None]
                    mean_ep = f'{np.mean(eps):.1f}' if eps else 'N/A'
                    fh.write(f'| {col} | {np.mean(accs):.4f} | {np.std(accs):.4f} | {mean_ep} |\n')
                fh.write('\n')
        print(f'Markdown saved → {os.path.relpath(md_path, ROOT)}')


if __name__ == '__main__':
    main()
