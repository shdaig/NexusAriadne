"""
Feature extraction classification benchmark.

Evaluates NexusKAN (feature extractor + FC head) against MLP baselines on
three binary classification tasks that differ in their algebraic dependency
structure.  After training, the script probes the learned NexusKAN layers for
interpretability: operation assignments derived from logit argmax and
feature-importance scores from acts_scale_spline.

Tasks
-----
T1 – Additive-ratio:    x0/x1 + x2/x3 > 2.2             (4 inputs)
T2 – Product-of-sums:   (x0 + x1) * (x2 + x3) > 0.49    (4 inputs)
T3 – Two-product-groups: x0*x1*x2 + x3*x4*x5 > 0.18     (6 inputs)

Run:
    python benchmarks/feature_ext_classification.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from nexuskan import NexusKAN, NexusKANLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_RUNS  = 3
EPOCHS  = 300
LO, HI  = 0.1, 1.0
N_TRAIN, N_TEST = 1000, 200


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        'name':        'T1 – Additive-ratio',
        'description': 'x0/x1 + x2/x3 > 2.2',
        'n_inputs':    4,
        'f':           lambda x: x[:, 0] / x[:, 1] + x[:, 2] / x[:, 3],
        'threshold':   2.2,
        'expected_ops': 'summation of ratio terms',
    },
    {
        'name':        'T2 – Product-of-sums',
        'description': '(x0 + x1) * (x2 + x3) > 0.49',
        'n_inputs':    4,
        'f':           lambda x: (x[:, 0] + x[:, 1]) * (x[:, 2] + x[:, 3]),
        'threshold':   0.49,
        'expected_ops': 'multiplication of summed pairs',
    },
    {
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
    n = task['n_inputs']
    x = (HI - LO) * torch.rand(N_TRAIN + N_TEST, n) + LO
    y_raw = task['f'](x)
    y = (y_raw > task['threshold']).float()
    return {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].unsqueeze(1).to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].unsqueeze(1).to(device),
    }


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def _tau(epoch, epochs, start=2.0, end=0.1):
    return start * (end / start) ** (epoch / max(epochs - 1, 1))

def _lr(epoch, epochs, start=0.1, end=1e-3):
    return start * (end / start) ** (epoch / max(epochs - 1, 1))

def _lamb(epoch, epochs, start=1e-2, end=10.0):
    return start * (end / start) ** (epoch / max(epochs - 1, 1))


# ---------------------------------------------------------------------------
# NexusKAN + FC model
# ---------------------------------------------------------------------------

class NexusKANWithFC(nn.Module):
    def __init__(self, kan_width, seed=0):
        super().__init__()
        # Capture output dimension before NexusKAN mutates the width list in-place
        out_dim = kan_width[-1] if isinstance(kan_width[-1], int) else kan_width[-1][0]
        self.kan = NexusKAN(
            width=kan_width, grid=5, k=3,
            grid_range=[LO, HI],
            seed=seed, auto_save=False, device=device,
        )
        self.fc      = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.kan(x)
        return self.sigmoid(self.fc(x))


def train_nexuskan_fc(model, dataset, epochs=EPOCHS, lr=0.1, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    epoch_test_acc = []

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // batch_size, 1)):
            batch_idx = idx[bi * batch_size : (bi + 1) * batch_size]
            bx = dataset['train_input'][batch_idx]
            by = dataset['train_label'][batch_idx]

            for pg in optimizer.param_groups:
                pg['lr'] = _lr(epoch, epochs)

            new_tau = _tau(epoch, epochs)
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    m.tau = new_tau

            out  = model(bx)
            loss = criterion(out, by)

            # Activation regularization (L1 + entropy of spline scales)
            l1_acts = ent_acts = 0.0
            n_acts = 0
            for acts in model.kan.acts_scale_spline:
                l1_acts += acts.sum()
                p_row = acts / (acts.sum(dim=1, keepdim=True) + 1e-8)
                p_col = acts / (acts.sum(dim=0, keepdim=True) + 1e-8)
                ent_acts += (
                    -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-8), dim=1))
                    - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-8), dim=0))
                )
                n_acts += 1
            if n_acts > 0:
                l1_acts /= n_acts
                ent_acts /= n_acts

            # Entropy regularization on logit operation probabilities
            lamb_ent_ops = _lamb(epoch, epochs)
            ent_ops = 0.0
            n_ops = 0
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    probs = torch.softmax(m.logits / m.tau, dim=-1)
                    ent_ops += (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean()
                    n_ops += 1
            ent_ops_avg = ent_ops / n_ops if n_ops > 0 else 0.0

            total = (loss
                     + lamb_ent_ops * ent_ops_avg
                     + 1e-2 * l1_acts
                     + 1e-2 * ent_acts)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(dataset['test_input'])
            acc = ((pred > 0.5).float() == dataset['test_label']).float().mean().item()
            epoch_test_acc.append(acc)

    return epoch_test_acc


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, width):
        super().__init__()
        layers = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                layers.append(nn.SiLU())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(model, dataset, epochs=EPOCHS, lr=0.1, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    epoch_test_acc = []

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // batch_size, 1)):
            batch_idx = idx[bi * batch_size : (bi + 1) * batch_size]
            bx = dataset['train_input'][batch_idx]
            by = dataset['train_label'][batch_idx]
            for pg in optimizer.param_groups:
                pg['lr'] = _lr(epoch, epochs)
            out  = model(bx)
            loss = criterion(out, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(dataset['test_input'])
            acc = ((pred > 0.5).float() == dataset['test_label']).float().mean().item()
            epoch_test_acc.append(acc)

    return epoch_test_acc


# ---------------------------------------------------------------------------
# Interpretability probe
# ---------------------------------------------------------------------------

def decode_operations(layer: NexusKANLayer):
    """
    Decode learned operation assignments from logits at final (low) temperature.

    For each (input i, output j) pair, softmax over the groups+1 dimension gives
    the probability of being assigned to mult-group 0, ..., K-1, or summation (last).
    Argmax gives the dominant assignment.

    Returns:
        assignments : (in_dim, out_dim) int tensor — argmax operation index
        entropy_mean: float — mean entropy over all (i,j) pairs (nats)
        n_mult      : int — number of (i,j) pairs assigned to a mult group
        n_sum       : int — number of (i,j) pairs assigned to summation
    """
    with torch.no_grad():
        tau_eval = 0.05   # near-zero temperature for crisp decoding
        probs = torch.softmax(layer.logits / tau_eval, dim=-1)  # (in, out, K+1)
        assignments = probs.argmax(dim=-1)                       # (in, out)

        eps = 1e-8
        ent = -(probs * torch.log(probs + eps)).sum(dim=-1)      # (in, out)
        entropy_mean = ent.mean().item()

        n_groups = layer.logits.shape[-1] - 1  # number of mult groups
        n_mult = (assignments < n_groups).sum().item()
        n_sum  = (assignments == n_groups).sum().item()

    return assignments.cpu(), entropy_mean, n_mult, n_sum


def feature_importance(kan_model):
    """
    Return per-input feature importance scores from the first layer's
    acts_scale_spline (absolute mean activation scale per input edge).
    Returns a 1-D tensor of length in_dim_layer0.
    """
    if not kan_model.acts_scale_spline:
        return None
    # acts_scale_spline[0] shape: (in_dim, out_dim)
    scores = kan_model.acts_scale_spline[0].abs().mean(dim=1)  # mean over outputs
    return scores.cpu()


def print_interpretability(model: NexusKANWithFC, task):
    """Print operation structure and feature importance for a trained NexusKANWithFC."""
    print(f'\n  -- Interpretability probe --')
    print(f'  Expected structure: {task["expected_ops"]}')

    layers = [m for m in model.kan.modules() if isinstance(m, NexusKANLayer)]
    for l_idx, layer in enumerate(layers):
        assignments, ent_mean, n_mult, n_sum = decode_operations(layer)
        total = n_mult + n_sum
        print(f'  Layer {l_idx}:  logit entropy={ent_mean:.3f} nats'
              f'  |  mult={n_mult}/{total}  sum={n_sum}/{total}')

        # Show per-output dominant assignment (compact view)
        in_d, out_d = assignments.shape
        for j in range(out_d):
            col = assignments[:, j].tolist()
            n_groups = layer.logits.shape[-1] - 1
            ops = ['M' + str(c) if c < n_groups else 'S' for c in col]
            print(f'    out[{j}]: inputs → [{", ".join(ops)}]')

    # Feature importance from first layer
    scores = feature_importance(model.kan)
    if scores is not None:
        top_k = min(task['n_inputs'], scores.numel())
        ranked = scores.argsort(descending=True)
        rank_str = '  '.join(f'x{ranked[k].item()}={scores[ranked[k]].item():.3f}'
                              for k in range(top_k))
        print(f'  Feature importance (layer 0, ranked): {rank_str}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def epochs_to_threshold(epoch_accs, threshold=0.90):
    for i, acc in enumerate(epoch_accs):
        if acc >= threshold:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'N_RUNS={N_RUNS}, EPOCHS={EPOCHS}')

    for task in TASKS:
        n = task['n_inputs']
        print(f'\n{"=" * 65}')
        print(f'{task["name"]}   {task["description"]}   (in_dim={n})')

        models_cfg = [
            ('NexusKAN-FC', 'nexuskan', [n, n, 1]),
            ('MLP-small',   'mlp',      [n, n, 1]),
            ('MLP-large',   'mlp',      [n, 4 * n, 2 * n, 1]),
        ]

        all_results = {name: {'acc': [], 'ep90': []} for name, _, _ in models_cfg}
        # Keep last NexusKAN model per run for interpretability probe
        last_nexuskan_model = None

        for run in range(N_RUNS):
            print(f'\n  Run {run + 1}/{N_RUNS}', flush=True)
            torch.manual_seed(run)
            np.random.seed(run)
            ds = make_dataset(task, seed=run)

            for model_name, model_type, width in models_cfg:
                if model_type == 'nexuskan':
                    model = NexusKANWithFC(kan_width=width, seed=run).to(device)
                    epoch_accs = train_nexuskan_fc(model, ds)
                    # Run forward pass once to populate acts_scale_spline
                    with torch.no_grad():
                        model.kan(ds['test_input'])
                    last_nexuskan_model = model
                else:
                    model = MLPClassifier(width).to(device)
                    epoch_accs = train_mlp(model, ds)

                final_acc = epoch_accs[-1]
                ep90      = epochs_to_threshold(epoch_accs, threshold=0.90)
                all_results[model_name]['acc'].append(final_acc)
                all_results[model_name]['ep90'].append(ep90)

                ep90_str = str(ep90) if ep90 is not None else '>EPOCHS'
                print(f'    {model_name:<16}: acc={final_acc:.4f}  epochs→90%={ep90_str}')

            # Interpretability probe on the NexusKAN model from this run
            if last_nexuskan_model is not None:
                print_interpretability(last_nexuskan_model, task)

        # ---- Per-task summary ----
        print(f'\n  Summary — {task["name"]}')
        print(f'  {"Model":<16}| {"Acc mean":>9} | {"Acc std":>8} | {"Ep→90% mean":>12}')
        print(f'  {"-" * 52}')
        for model_name, _, _ in models_cfg:
            accs  = all_results[model_name]['acc']
            ep90s = [e for e in all_results[model_name]['ep90'] if e is not None]
            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs))
            mean_ep  = float(np.mean(ep90s)) if ep90s else float('nan')
            print(f'  {model_name:<16}| {mean_acc:>9.4f} | {std_acc:>8.4f} | {mean_ep:>12.1f}')
