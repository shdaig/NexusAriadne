"""
Gated EML Tree — Symbolic Regression of ln(x)
==============================================

Binary tree of gated nodes:

    node(u, v) = sigmoid(alpha / tau) * (u * v)
               + (1 - sigmoid(alpha / tau)) * eml(u, v)

    eml(u, v)  = exp(u) - ln(v)

Leaves: pair 0 = (1, x), all other pairs = (1, 1).
x is fixed as the second (V) operand of the first leaf — the network
learns which internal node should receive it via the alpha gates alone.
There are no swap gates: eml is non-commutative, so fixing x in the V
slot biases the tree toward the correct orientation for ln(x).

All alpha logits init with small Gaussian noise → sigmoid ≈ 0.5:
no bias toward either choice.  Temperature tau anneals from tau_start
to tau_end across epochs, sharpening decisions.  Lambda on the binary
entropy of each alpha grows across epochs, driving gates toward hard
{0, 1} decisions.

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
    Full binary tree of depth D with 2^(depth-1) leaf pairs.

    Leaf layout (interleaved, flat index k = 0 .. 2n-1):
        k even  →  U-slot of pair k//2
        k odd   →  V-slot of pair k//2

    phi (size 2n = 2^depth): softmax over all 2n leaf cells.
    The cell with the highest weight receives x; all others receive 1.
    x can therefore land in any U or V slot of any pair.

    Each internal node has one alpha logit: sigmoid→1 means multiply,
    sigmoid→0 means eml.
    """

    def __init__(self, depth: int = 3, init_noise: float = 1.0):
        super().__init__()
        self.depth = depth
        n = 2 ** (depth - 1)

        # Level l ∈ [1..depth] has 2^(depth-l) nodes.
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.randn(2 ** (depth - l)) * init_noise)
            for l in range(1, depth + 1)
        ])
        # Softmax over all 2n leaf cells (U and V slots of every pair)
        self.phi = nn.Parameter(torch.randn(2 * n) * init_noise)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        x : complex128 tensor, shape (B,)
        Returns complex128 tensor, shape (B,)
        """
        B = x.shape[0]

        # soft_pos[k]: weight for flat leaf cell k (even→U slot, odd→V slot).
        # leaf_k = soft_pos[k]*x + (1 - soft_pos[k])*1  →  x where weight≈1, else 1.
        soft_pos = torch.softmax(self.phi / tau, dim=0).to(torch.complex128)  # (2n,)
        leaves = soft_pos.unsqueeze(0) * x.unsqueeze(1) + (1.0 - soft_pos.unsqueeze(0))
        U = leaves[:, 0::2]  # even indices → U slots, shape (B, n)
        V = leaves[:, 1::2]  # odd  indices → V slots, shape (B, n)

        for level_idx in range(self.depth):
            # Gated eml vs mult
            a = torch.sigmoid(self.alphas[level_idx] / tau).to(torch.complex128).unsqueeze(0)
            O = a * (U * V) + (1.0 - a) * safe_eml(U, V)

            if level_idx < self.depth - 1:
                O = O.view(B, O.shape[1] // 2, 2)
                U, V = O[..., 0], O[..., 1]
            else:
                return O.squeeze(-1)

    # ------------------------------------------------------------------

    def gate_entropy(self, tau: float) -> torch.Tensor:
        """Sum of binary entropies of alpha gates plus categorical entropy of phi."""
        eps = 1e-8

        def bce(p: torch.Tensor) -> torch.Tensor:
            return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

        total = torch.zeros(1)
        for a in self.alphas:
            total = total + bce(torch.sigmoid(a / tau)).mean()
        # Categorical entropy of the x-position softmax (minimize → concentrate on one slot)
        p_pos = torch.softmax(self.phi / tau, dim=0)
        total = total + -(p_pos * torch.log(p_pos + eps)).sum()
        return total

    def gate_probs(self, tau: float) -> dict[str, list]:
        """Readable gate probabilities for display."""
        out = {}
        for i, a in enumerate(self.alphas):
            pa = [f"{v:.3f}" for v in torch.sigmoid(a / tau).detach().cpu().tolist()]
            out[f"level {i + 1} alpha (mult)"] = pa
        pp = [f"{v:.3f}" for v in torch.softmax(self.phi / tau, dim=0).detach().cpu().tolist()]
        out["phi (x position)"] = pp
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

    n = 2 ** (depth - 1)
    # phi has 2n entries (even→U, odd→V); argmax picks the leaf that receives x
    x_cell = int(model.phi.detach().cpu().argmax().item())
    U_syms = ["1"] * n
    V_syms = ["1"] * n
    if x_cell % 2 == 0:
        U_syms[x_cell // 2] = "x"
    else:
        V_syms[x_cell // 2] = "x"

    # Level 1: leaf pairs
    symbols = []
    for i in range(n):
        u, v = U_syms[i], V_syms[i]
        if alphas_h[0][i] == 1:    # mult
            if u == "1":   expr = v
            elif v == "1": expr = u
            else:          expr = f"({u} * {v})"
        else:                       # eml
            expr = f"eml({u}, {v})"
        symbols.append(expr)

    # Levels 2..depth
    for l in range(1, depth):
        pairs = [(symbols[i], symbols[i + 1]) for i in range(0, len(symbols), 2)]
        new_syms = []
        for i, (u, v) in enumerate(pairs):
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
            v = (torch.sigmoid(model.alphas[i]) > threshold).float()
            hard.alphas[i].data = torch.where(v > 0.5,
                                              torch.full_like(v, 20.0),
                                              torch.full_like(v, -20.0))
        # Snap phi: one-hot at argmax cell
        x_cell = int(model.phi.detach().cpu().argmax().item())
        hard_phi = torch.full_like(model.phi, -100.0)
        hard_phi[x_cell] = 100.0
        hard.phi.data = hard_phi
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
    epochs: int = 1200,
    n_data: int = 1000,
    batch_size: int = 64,
    tau_start: float = 2.0,
    tau_end: float = 2.0,
    lr_start: float = 1e-1,
    lr_end: float = 1e-1,
    lam_ent_start: float = 1e-10,
    lam_ent_end: float = 1e-1,
    log_every: int = 50,
) -> None:
    x, y = make_data(n=n_data)
    model = GatedEMLTree(depth=depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    history: dict[str, list] = {"mse": [], "loss": [], "expr": []}

    for epoch in range(epochs):
        tau = get_tau(epoch, epochs, tau_start, tau_end)
        lr  = get_lr(epoch, epochs, lr_start, lr_end)
        lam = 0  # get_lam_ent(epoch, epochs, lam_ent_start, lam_ent_end)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        perm = torch.randperm(len(x))
        epoch_mse = epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]

            optimizer.zero_grad()
            y_pred = model(xb, tau=tau)
            mse  = ((y_pred.real - yb.real) ** 2).mean() + 0.01 * (y_pred.imag ** 2).mean()
            loss = mse + lam * model.gate_entropy(tau)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

            epoch_mse  += mse.item()  if torch.isfinite(mse)  else float("nan")
            epoch_loss += loss.item() if torch.isfinite(loss) else float("nan")
            n_batches  += 1

        mse_val  = epoch_mse  / n_batches
        loss_val = epoch_loss / n_batches
        history["mse"].append(mse_val)
        history["loss"].append(loss_val)

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            expr = snap_tree(model)
            history["expr"].append((epoch + 1, expr))
            print(
                f"epoch {epoch+1:5d}/{epochs}"
                f"  tau={tau:.4f}"
                f"  lr={lr:.2e}"
                f"  mse={mse_val:.3e}"
                f"  lam={lam:.2e}"
                f"  snap: {expr}"
            )

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    final_expr  = snap_tree(model)
    snapped_mse = evaluate_snapped(model, x, y)
    print(f"Final snapped expression : {final_expr}")
    print(f"Snapped MSE              : {snapped_mse:.4e}")
    print(f"Soft MSE (last epoch)    : {history['mse'][-1]:.4e}")
    print()
    print("Gate probabilities (final tau):")
    for key, vals in model.gate_probs(tau=tau_end).items():
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
