"""
Gated EML Tree — Symbolic Regression of ln(x)
==============================================

Binary tree of gated nodes:

    node(u, v) = sigmoid(alpha / tau) * (u * v)
               + (1 - sigmoid(alpha / tau)) * eml(u, v)

    eml(u, v)  = exp(u) - ln(v)

Each node additionally has a swap gate (gamma) applied to its inputs
before the alpha gate, so the tree can orient (u, v) freely at every
level — not just at leaves.  This is necessary because eml is
non-commutative: only eml(1, M) = ln(x), not eml(M, 1).

All logits (alpha, gamma) init to 0  →  sigmoid(0 / tau) = 0.5:
no bias toward either choice.  Temperature tau anneals from tau_start
to tau_end across epochs, sharpening decisions.  Lambda on the binary
entropy of each gate grows across epochs, mimicking the NexusKAN
entropy schedule that drives gates toward hard {0, 1} decisions.

Target: ln(x).   Known EML form: eml(1, eml(eml(1, x), 1)).
"""

from __future__ import annotations
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# EML primitive
# ---------------------------------------------------------------------------

def safe_eml(u: torch.Tensor, v: torch.Tensor,
             clip_real: float = 5.5, clip_imag: float = 20.0) -> torch.Tensor:
    """
    eml(u, v) = exp(u) - ln(v), numerically stabilized.

    clip_real = 5.5: covers the deepest correct intermediate value for the
    ln(x) formula on x ∈ [0.2, 5.0]  (max = e − ln(0.2) ≈ 4.33 < 5.5).
    Wrong-path explosions are bounded by exp(5.5) ≈ 245.

    clip_imag = 20.0: prevents wild phase oscillations in the imaginary part
    while still allowing enough range for complex logarithm arithmetic.
    """
    ur = u.real
    ui = u.imag
    # Differentiable clamp (gradient = 1 inside, 0 outside)
    ur_c = ur - torch.clamp(ur - clip_real, min=0.0) + torch.clamp(-clip_real - ur, min=0.0)
    ui_c = ui - torch.clamp(ui - clip_imag, min=0.0) + torch.clamp(-clip_imag - ui, min=0.0)
    u_c = torch.complex(ur_c, ui_c)
    eps = 1e-30
    v_reg = v + eps * (v.abs() < eps).to(v.dtype)
    return torch.exp(u_c) - torch.log(v_reg)


# ---------------------------------------------------------------------------
# Schedules  (exponential annealing, same pattern as NexusKAN)
# ---------------------------------------------------------------------------

def get_tau(epoch: int, epochs: int, tau_start: float = 5.0, tau_end: float = 0.1) -> float:
    return tau_start * (tau_end / tau_start) ** (epoch / max(epochs - 1, 1))


def get_lr(epoch: int, epochs: int, lr_start: float = 3e-2, lr_end: float = 1e-4) -> float:
    return lr_start * (lr_end / lr_start) ** (epoch / max(epochs - 1, 1))


def get_lam_ent(epoch: int, epochs: int, lam_start: float = 1e-4, lam_end: float = 5e-2) -> float:
    return lam_start * (lam_end / lam_start) ** (epoch / max(epochs - 1, 1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GatedEMLTree(nn.Module):
    """
    Full binary tree of depth D.
    Leaves: pair 0 = (x, 1), all other pairs = (1, 1).

    At every level each node has:
      - gamma logit: soft swap of (U, V) before the gate
      - alpha logit: mult vs eml gate on the (possibly swapped) inputs

    All logits init to 0 → sigmoid(0/tau) = 0.5 (unbiased).
    """

    def __init__(self, depth: int = 3, init_noise: float = 1.0):
        super().__init__()
        self.depth = depth

        # Level l ∈ [1..depth] has 2^(depth-l) nodes.
        # Logits are initialized with zero-mean Gaussian noise (unbiased on
        # average, but breaks the saddle at exactly 0.5 for every gate).
        self.alphas = nn.ParameterList([   # eml vs mult gate
            nn.Parameter(torch.randn(2 ** (depth - l)) * init_noise)
            for l in range(1, depth + 1)
        ])
        self.gammas = nn.ParameterList([   # swap gate (per node, every level)
            nn.Parameter(torch.randn(2 ** (depth - l)) * init_noise)
            for l in range(1, depth + 1)
        ])

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        x : complex128 tensor, shape (B,)
        Returns complex128 tensor, shape (B,)
        """
        B = x.shape[0]
        n = 2 ** (self.depth - 1)
        dev = x.device

        U = torch.ones(B, n, dtype=torch.complex128, device=dev)
        V = torch.ones(B, n, dtype=torch.complex128, device=dev)
        U[:, 0] = x

        for level_idx in range(self.depth):
            # Soft swap of (U, V) inputs at each node
            g = torch.sigmoid(self.gammas[level_idx] / tau).to(torch.complex128).unsqueeze(0)
            U_s = g * U + (1.0 - g) * V
            V_s = (1.0 - g) * U + g * V

            # Gated eml vs mult
            a = torch.sigmoid(self.alphas[level_idx] / tau).to(torch.complex128).unsqueeze(0)
            O = a * (U_s * V_s) + (1.0 - a) * safe_eml(U_s, V_s)

            if level_idx < self.depth - 1:
                O = O.view(B, O.shape[1] // 2, 2)
                U, V = O[..., 0], O[..., 1]
            else:
                return O.squeeze(-1)

    # ------------------------------------------------------------------

    def gate_entropy(self, tau: float) -> torch.Tensor:
        """Sum of binary entropies of all gate probabilities."""
        eps = 1e-8

        def bce(p: torch.Tensor) -> torch.Tensor:
            return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

        total = torch.zeros(1)
        for a, g in zip(self.alphas, self.gammas):
            total = total + bce(torch.sigmoid(a / tau)).mean()
            total = total + bce(torch.sigmoid(g / tau)).mean()
        return total

    def gate_probs(self, tau: float) -> dict[str, list]:
        """Readable gate probabilities for display."""
        out = {}
        for i, (a, g) in enumerate(zip(self.alphas, self.gammas)):
            pa = [f"{v:.3f}" for v in torch.sigmoid(a / tau).detach().cpu().tolist()]
            pg = [f"{v:.3f}" for v in torch.sigmoid(g / tau).detach().cpu().tolist()]
            out[f"level {i + 1} alpha (mult)"] = pa
            out[f"level {i + 1} gamma (keep)"] = pg
        return out


# ---------------------------------------------------------------------------
# Symbolic snap
# ---------------------------------------------------------------------------

def snap_tree(model: GatedEMLTree, threshold: float = 0.5) -> str:
    """Discretize all gates to {0,1} and return the symbolic expression."""
    depth = model.depth

    def hard(param: nn.Parameter) -> list[int]:
        return (torch.sigmoid(param).detach().cpu() > threshold).int().tolist()

    alphas_h = [hard(a) for a in model.alphas]
    gammas_h = [hard(g) for g in model.gammas]

    n = 2 ** (depth - 1)
    # Leaves: pair 0 = (x, 1), rest = (1, 1)
    U_syms = ["x"] + ["1"] * (n - 1)
    V_syms = ["1"] * n

    # Level-1 gamma applies first (leaf-pair level)
    symbols = []
    for i in range(n):
        u, v = U_syms[i], V_syms[i]
        if gammas_h[0][i] == 0:    # gamma=0 means swap
            u, v = v, u
        if alphas_h[0][i] == 1:    # mult
            if u == "1":   expr = v
            elif v == "1": expr = u
            else:          expr = f"({u} * {v})"
        else:                       # eml
            expr = f"eml({u}, {v})"
        symbols.append(expr)

    # Levels 2..depth
    for l in range(1, depth):
        # Pair up adjacent symbols into (U, V) for this level
        pairs = [(symbols[i], symbols[i + 1]) for i in range(0, len(symbols), 2)]
        new_syms = []
        for i, (u, v) in enumerate(pairs):
            if gammas_h[l][i] == 0:   # swap
                u, v = v, u
            if alphas_h[l][i] == 1:   # mult
                if u == "1":   expr = v
                elif v == "1": expr = u
                else:          expr = f"({u} * {v})"
            else:                      # eml
                expr = f"eml({u}, {v})"
            new_syms.append(expr)
        symbols = new_syms

    return symbols[0]


def evaluate_snapped(model: GatedEMLTree, x: torch.Tensor, y: torch.Tensor,
                     threshold: float = 0.5) -> float:
    """Hard-gate copy of model, measure MSE."""
    with torch.no_grad():
        hard = GatedEMLTree(depth=model.depth)
        for i in range(model.depth):
            for src_list, dst_list in [
                (model.alphas, hard.alphas),
                (model.gammas, hard.gammas),
            ]:
                v = (torch.sigmoid(src_list[i]) > threshold).float()
                dst_list[i].data = torch.where(v > 0.5,
                                               torch.full_like(v, 20.0),
                                               torch.full_like(v, -20.0))
        y_pred = hard(x, tau=0.01)
        return ((y_pred.real - y.real) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_data(n: int = 300, x_lo: float = 0.2, x_hi: float = 5.0):
    x_r = torch.rand(n, dtype=torch.float64) * (x_hi - x_lo) + x_lo
    x = x_r.to(torch.complex128)
    y = torch.log(x)
    return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    depth: int = 3,
    # Phase 1: MSE exploration — soft gates, high tau, no entropy pressure
    phase1_epochs: int = 500,
    tau1_start: float = 2.0,
    tau1_end: float = 1.0,
    lr1_start: float = 1.0,
    lr1_end: float = 1e-3,
    # Phase 2: MSE + entropy sharpening, tau falls to 0.05
    phase2_epochs: int = 700,
    tau2_start: float = 1.0,
    tau2_end: float = 0.05,
    lr2_start: float = 3e-3,
    lr2_end: float = 5e-5,
    lam_ent_start: float = 1e-4,
    lam_ent_end: float = 1e-1,
    log_every: int = 50,
) -> None:
    """
    Two-phase training, loss mapped across all epochs.

    Phase 1 — MSE only, tau anneals 5 → 1: free exploration without
               any gate commitment pressure.
    Phase 2 — MSE + growing entropy penalty (lam 1e-4 → 1e-1), tau
               falls to 0.05 so sigmoid(alpha/tau) ≈ step(alpha).
               Gates sharpen toward hard {0,1} decisions while the MSE
               gradient keeps guiding which way they snap.
    """
    x, y = make_data()
    model = GatedEMLTree(depth=depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1_start)

    history = {"mse": [], "loss": [], "tau": [], "lam": [], "expr": []}
    total_epochs = phase1_epochs + phase2_epochs

    def _step(epoch_global: int, tau: float, lr: float,
               mse_val: float, lam: float) -> None:
        expr = snap_tree(model)
        history["expr"].append((epoch_global + 1, expr))
        print(
            f"epoch {epoch_global+1:5d}/{total_epochs}"
            f"  tau={tau:.4f}"
            f"  lr={lr:.2e}"
            f"  mse={mse_val:.3e}"
            f"  lam={lam:.2e}"
            f"  snap: {expr}"
        )

    # ---- Phase 1 ----
    print(f"Phase 1 ({phase1_epochs} epochs)  MSE only, tau {tau1_start} → {tau1_end}")
    for epoch in range(phase1_epochs):
        tau = get_tau(epoch, phase1_epochs, tau1_start, tau1_end)
        lr  = get_lr(epoch, phase1_epochs, lr1_start, lr1_end)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        y_pred = model(x, tau=tau)
        mse = ((y_pred.real - y.real) ** 2).mean() + 0.01 * (y_pred.imag ** 2).mean()

        if torch.isfinite(mse):
            mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        mse_val = mse.item() if torch.isfinite(mse) else float("nan")
        history["mse"].append(mse_val)
        history["loss"].append(mse_val)
        history["tau"].append(tau)
        history["lam"].append(0.0)

        if (epoch + 1) % log_every == 0 or epoch == phase1_epochs - 1:
            _step(epoch, tau, lr, mse_val, 0.0)

    # ---- Phase 2 ----
    print(f"\nPhase 2 ({phase2_epochs} epochs)  tau {tau2_start} → {tau2_end}  "
          f"lam_ent {lam_ent_start:.0e} → {lam_ent_end:.0e}")
    for epoch in range(phase2_epochs):
        g = phase1_epochs + epoch
        tau = get_tau(epoch, phase2_epochs, tau2_start, tau2_end)
        lr  = get_lr(epoch, phase2_epochs, lr2_start, lr2_end)
        lam = get_lam_ent(epoch, phase2_epochs, lam_ent_start, lam_ent_end)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        y_pred = model(x, tau=tau)
        mse = ((y_pred.real - y.real) ** 2).mean() + 0.01 * (y_pred.imag ** 2).mean()
        ent = model.gate_entropy(tau)
        loss = mse + lam * ent

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        mse_val  = mse.item() if torch.isfinite(mse) else float("nan")
        loss_val = loss.item() if torch.isfinite(loss) else float("nan")
        history["mse"].append(mse_val)
        history["loss"].append(loss_val)
        history["tau"].append(tau)
        history["lam"].append(lam)

        if (epoch + 1) % log_every == 0 or epoch == phase2_epochs - 1:
            _step(g, tau, lr, mse_val, lam)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    final_expr  = snap_tree(model)
    snapped_mse = evaluate_snapped(model, x, y)
    print(f"Final snapped expression : {final_expr}")
    print(f"Snapped MSE              : {snapped_mse:.4e}")
    print(f"Soft MSE (last epoch)    : {history['mse'][-1]:.4e}")
    print()
    print("Gate probabilities (final tau):")
    for key, vals in model.gate_probs(tau=tau2_end).items():
        print(f"  {key}: {vals}")
    print("=" * 70)

    print("\nExpression evolution (unique formulas):")
    seen: set[str] = set()
    for ep, expr in history["expr"]:
        if expr not in seen:
            print(f"  epoch {ep:5d}: {expr}")
            seen.add(expr)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    train()
