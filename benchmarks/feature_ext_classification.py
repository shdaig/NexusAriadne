"""
Feature extraction classification benchmark.

Compares NexusKAN (as a feature extractor with a linear head) against
small and large MLP baselines on a binary classification task whose
decision boundary is structured around division:

    label = 1  iff  x0/x3 + x1/x2 > 2.2

This mirrors the existing lan_lo_fc_division_learning scripts but runs
multiple seeds and reports test accuracy statistics.

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
THRESHOLD = 2.2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(seed=0):
    torch.manual_seed(seed)
    x = (HI - LO) * torch.rand(N_TRAIN + N_TEST, 4) + LO
    y_raw = x[:, 0] / x[:, 3] + x[:, 1] / x[:, 2]
    y = torch.where(y_raw > THRESHOLD, 1.0, 0.0)
    return {
        'train_input': x[:N_TRAIN].to(device),
        'train_label': y[:N_TRAIN].unsqueeze(1).to(device),
        'test_input':  x[N_TRAIN:].to(device),
        'test_label':  y[N_TRAIN:].unsqueeze(1).to(device),
    }


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
# NexusKAN-FC model
# ---------------------------------------------------------------------------

class NexusKANWithFC(nn.Module):
    def __init__(self, kan_width, seed=0):
        super().__init__()
        # Read out_dim before NexusKAN mutates the width list in-place
        out_dim = kan_width[-1] if isinstance(kan_width[-1], int) else kan_width[-1][0]
        self.kan = NexusKAN(
            width=kan_width, grid=5, k=3,
            grid_range=[LO, HI],
            seed=seed, auto_save=False, device=device,
        )
        self.fc = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.kan(x)
        x = self.fc(x)
        return self.sigmoid(x)


def train_nexuskan_fc(model, dataset, epochs=EPOCHS, lr=0.1, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Track per-epoch test accuracy for "epochs to 90%" metric
    epoch_test_acc = []

    for epoch in range(epochs):
        idx = torch.randperm(N_TRAIN, device=device)
        for bi in range(max(N_TRAIN // batch_size, 1)):
            batch_idx = idx[bi * batch_size : (bi + 1) * batch_size]
            bx = dataset['train_input'][batch_idx]
            by = dataset['train_label'][batch_idx]

            for pg in optimizer.param_groups:
                pg['lr'] = _get_lr(epoch, epochs, lr_start=lr, lr_end=1e-3)

            new_tau = _get_tau(epoch, epochs)
            for m in model.modules():
                if isinstance(m, NexusKANLayer):
                    m.tau = new_tau

            out  = model(bx)
            loss = criterion(out, by)

            # Activation regularization (L1 + entropy of scale_spline)
            lamb_l1  = 1e-2
            lamb_ent = 1e-2
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

            # Entropy regularization for logits
            lamb_ent_ops = _get_lamb(epoch, epochs)
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
                     + lamb_l1  * l1_acts
                     + lamb_ent * ent_acts)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(dataset['test_input'])
            acc = ((pred > 0.5).float() == dataset['test_label']).float().mean().item()
            epoch_test_acc.append(acc)

    return epoch_test_acc


# ---------------------------------------------------------------------------
# MLP runners (using standard torch training, not MLP.fit, to use BCE)
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


def train_mlp_classifier(model, dataset, epochs=EPOCHS, lr=0.1, batch_size=64):
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
                pg['lr'] = _get_lr(epoch, epochs, lr_start=lr, lr_end=1e-3)

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
# Helpers
# ---------------------------------------------------------------------------

def epochs_to_threshold(epoch_accs, threshold=0.90):
    """Return first epoch index (1-based) where accuracy exceeds threshold, or None."""
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
    print(f'Task: x0/x3 + x1/x2 > {THRESHOLD}  (binary classification, 4 inputs)\n')

    models_cfg = [
        ('NexusKAN-FC', 'nexuskan', [4, 8, 1]),
        ('MLP-small',   'mlp',      [4, 8, 1]),
        ('MLP-large',   'mlp',      [4, 32, 16, 1]),
    ]

    all_results = {name: {'acc': [], 'epochs_to_90': []} for name, _, _ in models_cfg}

    for run in range(N_RUNS):
        print(f'Run {run + 1}/{N_RUNS}', flush=True)
        torch.manual_seed(run)
        np.random.seed(run)
        ds = make_dataset(seed=run)

        for model_name, model_type, width in models_cfg:
            if model_type == 'nexuskan':
                model = NexusKANWithFC(kan_width=width, seed=run).to(device)
                epoch_accs = train_nexuskan_fc(model, ds)
            else:
                model = MLPClassifier(width).to(device)
                epoch_accs = train_mlp_classifier(model, ds)

            final_acc = epoch_accs[-1]
            ep90 = epochs_to_threshold(epoch_accs, threshold=0.90)
            all_results[model_name]['acc'].append(final_acc)
            all_results[model_name]['epochs_to_90'].append(ep90)

            ep90_str = str(ep90) if ep90 is not None else 'N/A'
            print(f'  {model_name:<16}: acc={final_acc:.4f}, epochs_to_90%={ep90_str}')

    # ---- Summary table ----
    print(f'\n{"=" * 65}')
    print(f'Feature Extraction Classification  (x0/x3 + x1/x2 > {THRESHOLD})')
    print(f'{"Method":<16}| {"Acc Mean":>10} | {"Acc Std":>8} | {"Ep→90% Mean":>12} | {"Ep→90% Std":>11}')
    print('-' * 65)
    for model_name, _, _ in models_cfg:
        accs = all_results[model_name]['acc']
        ep90s = [e for e in all_results[model_name]['epochs_to_90'] if e is not None]

        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))

        if ep90s:
            mean_ep = float(np.mean(ep90s))
            std_ep  = float(np.std(ep90s))
            ep_str  = f'{mean_ep:>12.1f}'
            std_ep_str = f'{std_ep:>11.1f}'
        else:
            ep_str = f'{"N/A":>12}'
            std_ep_str = f'{"N/A":>11}'

        print(f'{model_name:<16}| {mean_acc:>10.4f} | {std_acc:>8.4f} | {ep_str} | {std_ep_str}')
