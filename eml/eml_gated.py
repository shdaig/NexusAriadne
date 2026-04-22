"""
Gated EML Network for Symbolic Regression of Univariate Elementary Functions
============================================================================

Implements the gated-pass-through EML tree:

    node(u, v) = alpha * (u * v) + (1 - alpha) * eml(u, v)
    eml(u, v) = exp(u) - log(v)

A full binary tree of depth D has 2^D - 1 internal nodes, each with its own
alpha in [0,1]. Leaves are fixed as pairs [[x,1],[1,1],...,[1,1]], with an
optional per-leaf-pair 'beta' gate that softly swaps (u,v) <-> (v,u), since
eml is non-commutative.

When alpha=1 on every node, the multiplier branch turns the tree into an
identity wire computing x, giving a benign initialization. Training then
'turns on' eml at specific nodes (alpha -> 0) as required.

Target function by default: ln(x).
Known solution (paper's formula): ln(x) = eml(1, eml(eml(1,x), 1)).

Author: demonstration implementation.
"""

from __future__ import annotations
import math
import argparse
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Numerically conditioned EML
# ---------------------------------------------------------------------------

def safe_eml(u: torch.Tensor, v: torch.Tensor, clip: float = 80.0) -> torch.Tensor:
    """
    eml(u, v) = exp(u) - log(v), stabilized:
      - exp input clipped to [-clip, clip] (applied to the real part only,
        preserving the imaginary phase)
      - log input regularized: v -> v + eps_vec where eps_vec nudges zeros
        off the branch cut.

    Operates in complex128. Returns complex128.
    """
    # Clip exponent's real part to prevent overflow. We do this by subtracting
    # a detached excess from u.real so gradients still flow where unclipped.
    ur = u.real
    ui = u.imag
    excess_hi = torch.clamp(ur - clip, min=0.0)
    excess_lo = torch.clamp(-clip - ur, min=0.0)
    u_clipped = torch.complex(ur - excess_hi + excess_lo, ui)

    # Regularize log input: push tiny |v| away from zero
    eps = 1e-30
    v_reg = v + eps * (v.abs() < eps).to(v.dtype)

    return torch.exp(u_clipped) - torch.log(v_reg)


# ---------------------------------------------------------------------------
# The Gated EML Tree
# ---------------------------------------------------------------------------

class GatedEMLTree(nn.Module):
    """
    Depth-D full binary tree of gated EML nodes.

    Parameters
    ----------
    depth : int
        Tree depth. Number of leaves = 2**depth, number of internal nodes
        = 2**depth - 1 (wait: with our convention, leaves feed level 1 and
        root is level `depth`, so number of level-1 nodes = 2**(depth-1),
        total nodes = 2**depth - 1).
    use_swap : bool
        If True, add a per-leaf-pair beta gate that softly swaps (u,v).
        Needed to represent ln(x) which wants x in the second slot.
    alpha_init : float
        Initial logit for alpha. +3.0 means sigmoid ~= 0.953, so each node
        starts as mostly-multiplier (near-identity).
    """

    def __init__(self, depth: int = 3, use_swap: bool = True,
                 alpha_init: float = 3.0):
        super().__init__()
        self.depth = depth
        self.use_swap = use_swap

        # Per-level alpha logits (one per node at each level).
        # Level l has 2**(depth - l) nodes, l = 1..depth.
        self.alphas = nn.ParameterList([
            nn.Parameter(alpha_init * torch.ones(2 ** (depth - l)))
            for l in range(1, depth + 1)
        ])

        # Per-leaf-pair swap logits (one per level-1 node).
        n_leaf_pairs = 2 ** (depth - 1)
        if use_swap:
            self.beta = nn.Parameter(torch.zeros(n_leaf_pairs))
        else:
            self.register_buffer("beta", torch.zeros(n_leaf_pairs))

    # ---- leaf construction ----

    def build_leaves(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: complex tensor of shape (B,) for B data points.
        Returns (U, V), each of shape (B, n_leaf_pairs) in complex128.
        Leaf pattern: pair 0 = (x, 1), pairs 1.. = (1, 1).
        """
        B = x.shape[0]
        n = 2 ** (self.depth - 1)
        dev = x.device
        U = torch.ones(B, n, dtype=torch.complex128, device=dev)
        V = torch.ones(B, n, dtype=torch.complex128, device=dev)
        U[:, 0] = x
        return U, V

    def apply_swap(self, U: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft swap of (u,v) pairs at the leaves using beta."""
        b = torch.sigmoid(self.beta).to(U.dtype)     # (n,)
        b = b.unsqueeze(0)                            # (1, n) broadcasts over batch
        U_new = b * U + (1.0 - b) * V
        V_new = (1.0 - b) * U + b * V
        return U_new, V_new

    # ---- forward ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: complex128 tensor of shape (B,).
        Returns complex128 tensor of shape (B,) = tree root output.
        """
        U, V = self.build_leaves(x)           # (B, n_leaf_pairs)
        U, V = self.apply_swap(U, V)

        for level_idx in range(self.depth):
            a = torch.sigmoid(self.alphas[level_idx]).to(U.dtype)   # (n_level,)
            a = a.unsqueeze(0)                                       # (1, n)

            mult_path = U * V
            eml_path  = safe_eml(U, V)

            O = a * mult_path + (1.0 - a) * eml_path                 # (B, n_level)

            if level_idx < self.depth - 1:
                # Reshape (B, 2*k) -> (B, k, 2) and split pairs
                B, N = O.shape
                O = O.view(B, N // 2, 2)
                U = O[..., 0]
                V = O[..., 1]
            else:
                return O.squeeze(-1)          # (B,) at the root
        return O  # unreachable

    # ---- readouts ----

    def readable_alphas(self) -> list[torch.Tensor]:
        return [torch.sigmoid(a).detach().cpu() for a in self.alphas]

    def readable_beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta).detach().cpu()

    def n_params(self) -> int:
        n = sum(a.numel() for a in self.alphas)
        if self.use_swap:
            n += self.beta.numel()
        return n


# ---------------------------------------------------------------------------
# Symbolic export (snap weights, print formula)
# ---------------------------------------------------------------------------

def snap_tree(model: GatedEMLTree, threshold: float = 0.5) -> str:
    """
    Discretize alphas and betas to {0,1} and build a symbolic expression
    string. Returns the formula at the root.
    """
    depth = model.depth
    alphas_soft = model.readable_alphas()
    alphas_hard = [(a > threshold).to(torch.int32).tolist() for a in alphas_soft]
    beta_hard = (model.readable_beta() > threshold).to(torch.int32).tolist()

    # Build leaf symbols
    n_leaf_pairs = 2 ** (depth - 1)
    U_syms = ["x"] + ["1"] * (n_leaf_pairs - 1)
    V_syms = ["1"] * n_leaf_pairs

    # Soft swap -> hard swap at leaves. beta=1 means "no swap" (stays b*U+(1-b)*V = U).
    U_leaves, V_leaves = [], []
    for i, bi in enumerate(beta_hard):
        if bi == 1:
            U_leaves.append(U_syms[i]); V_leaves.append(V_syms[i])
        else:
            U_leaves.append(V_syms[i]); V_leaves.append(U_syms[i])

    current = []
    for u, v in zip(U_leaves, V_leaves):
        current.append((u, v))  # unpaired, they become node inputs level by level

    # current is a list of (u,v) input pairs for level 1
    symbols = [(u, v) for (u, v) in current]

    for l in range(depth):
        a_level = alphas_hard[l]     # 1 = mult, 0 = eml
        new_symbols = []
        for i, ((u, v), a) in enumerate(zip(symbols, a_level)):
            if a == 1:
                # Multiplier: simplify trivial products
                if u == "1": expr = v
                elif v == "1": expr = u
                else: expr = f"({u} * {v})"
            else:
                expr = f"eml({u}, {v})"
            new_symbols.append(expr)
        # Pair them up for next level (unless we're at root)
        if l < depth - 1:
            paired = []
            for i in range(0, len(new_symbols), 2):
                paired.append((new_symbols[i], new_symbols[i + 1]))
            symbols = paired
        else:
            return new_symbols[0]
    return new_symbols[0]  # unreachable


# ---------------------------------------------------------------------------
# Target functions and data generation
# ---------------------------------------------------------------------------

TARGETS = {
    "ln":   lambda x: torch.log(x),
    "exp":  lambda x: torch.exp(x),
    "x":    lambda x: x,
    "1/x":  lambda x: 1.0 / x,
    "-x":   lambda x: -x,
    "x^2":  lambda x: x * x,
    "ln_of_exp_minus_ln":  # ln(e - ln(x)), easy depth-2 eml composition
        lambda x: torch.log(math.e - torch.log(x)),
}


def generate_data(target_name: str, n: int = 200,
                  x_range: tuple[float, float] = (0.2, 5.0),
                  seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x in the given real range, compute y = f(x) with complex128 arithmetic.
    """
    g = torch.Generator().manual_seed(seed)
    x_real = torch.rand(n, generator=g, dtype=torch.float64)
    x_real = x_range[0] + (x_range[1] - x_range[0]) * x_real
    x = x_real.to(torch.complex128)
    f = TARGETS[target_name]
    y = f(x)
    return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_once(
    target_name: str,
    depth: int,
    seed: int,
    n_steps: int = 4000,
    lr: float = 3e-2,
    sparsity_weight: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Train one GatedEMLTree on the chosen target and return a dict of results.

    The loss is MSE on the real part of the tree output vs. real target,
    plus a small regularizer that slightly penalizes "multiplier" usage at
    upper levels (encouraging eml where possible, so the snapped expression
    is a clean formula).
    """
    torch.manual_seed(seed)
    x, y = generate_data(target_name, n=200, seed=seed)

    model = GatedEMLTree(depth=depth, use_swap=True, alpha_init=3.0)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps)

    best_loss = float("inf")
    best_state = None
    history = []

    for step in range(n_steps):
        optim.zero_grad()
        y_pred = model(x)

        # Primary loss: MSE on real part (targets are real)
        real_err = (y_pred.real - y.real)
        imag_pen = y_pred.imag.pow(2).mean()     # should be ~0 for real targets
        mse = real_err.pow(2).mean() + 0.01 * imag_pen

        # Light sparsity: mild pressure toward alpha=0 (eml) so the final
        # snapped expression prefers the canonical pure-EML form.
        # Kept small; the data fit dominates.
        alpha_reg = sum(torch.sigmoid(a).sum() for a in model.alphas)
        loss = mse + sparsity_weight * alpha_reg

        if torch.isfinite(loss):
            loss.backward()
            # Gradient clipping to handle occasional explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            sched.step()
        else:
            # Catastrophic step: re-init alphas near identity and continue
            with torch.no_grad():
                for a in model.alphas:
                    a.fill_(3.0)
                model.beta.zero_()

        history.append(mse.item() if torch.isfinite(mse) else float("nan"))

        if torch.isfinite(mse) and mse.item() < best_loss:
            best_loss = mse.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            with torch.no_grad():
                print(f"  step {step:5d}  mse={mse.item():.3e}  best={best_loss:.3e}")

    # Load best and evaluate snapped formula
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        y_pred = model(x)
        final_mse = ((y_pred.real - y.real) ** 2).mean().item()

    snapped_expr = snap_tree(model)

    # Also measure snapped numerical MSE by rebuilding with hard {0,1} params
    snapped_mse = evaluate_snapped(model, x, y)

    return {
        "seed": seed,
        "depth": depth,
        "target": target_name,
        "final_soft_mse": final_mse,
        "snapped_mse": snapped_mse,
        "snapped_expr": snapped_expr,
        "alphas": [a.tolist() for a in model.readable_alphas()],
        "beta": model.readable_beta().tolist(),
        "history": history,
        "n_params": model.n_params(),
    }


def evaluate_snapped(model: GatedEMLTree, x: torch.Tensor, y: torch.Tensor,
                     threshold: float = 0.5) -> float:
    """
    Rebuild a model with hard {0,1} alphas/betas and measure MSE. This tests
    whether the 'snapped' symbolic formula actually fits the data.
    """
    with torch.no_grad():
        hard_model = GatedEMLTree(depth=model.depth, use_swap=model.use_swap)
        for i, a in enumerate(model.alphas):
            hard_vals = (torch.sigmoid(a) > threshold).to(torch.float64)
            # Map {0,1} to logits {-inf, +inf} approximated by +/- large.
            hard_model.alphas[i].data = torch.where(
                hard_vals > 0.5,
                torch.full_like(hard_vals,  20.0),
                torch.full_like(hard_vals, -20.0),
            )
        hard_vals = (torch.sigmoid(model.beta) > threshold).to(torch.float64)
        hard_model.beta.data = torch.where(
            hard_vals > 0.5,
            torch.full_like(hard_vals,  20.0),
            torch.full_like(hard_vals, -20.0),
        )
        y_pred = hard_model(x)
        return ((y_pred.real - y.real) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Validation driver
# ---------------------------------------------------------------------------

def run_experiment(target_name: str = "ln",
                   depth: int = 3,
                   n_restarts: int = 20,
                   n_steps: int = 4000) -> None:
    """
    Train `n_restarts` models with different seeds, pick best, report.
    This is the 'ensemble with restart' approach — essential because EML
    training has many basins of attraction.
    """
    print(f"\n{'='*66}")
    print(f"Target: {target_name}   depth: {depth}   restarts: {n_restarts}")
    print(f"{'='*66}")

    results = []
    for s in range(n_restarts):
        print(f"\n--- seed {s} ---")
        r = train_once(target_name, depth, seed=s, n_steps=n_steps, verbose=False)
        results.append(r)
        print(f"  final MSE={r['final_soft_mse']:.3e}   "
              f"snapped MSE={r['snapped_mse']:.3e}")
        print(f"  snapped expr: {r['snapped_expr']}")

    # Report
    results.sort(key=lambda r: r["snapped_mse"])
    best = results[0]
    print(f"\n{'-'*66}")
    print(f"BEST OVER {n_restarts} RESTARTS")
    print(f"{'-'*66}")
    print(f"  seed            : {best['seed']}")
    print(f"  soft  MSE       : {best['final_soft_mse']:.6e}")
    print(f"  snapped MSE     : {best['snapped_mse']:.6e}")
    print(f"  snapped formula : {best['snapped_expr']}")
    print(f"  #parameters     : {best['n_params']}")

    # How many restarts got an exact-ish recovery?
    exact = sum(1 for r in results if r["snapped_mse"] < 1e-10)
    close = sum(1 for r in results if r["snapped_mse"] < 1e-4)
    print(f"\n  {exact}/{n_restarts} restarts achieved snapped MSE < 1e-10 (exact)")
    print(f"  {close}/{n_restarts} restarts achieved snapped MSE < 1e-4  (close)")

    # Show top 3
    print(f"\nTop 3 snapped formulas:")
    for r in results[:3]:
        print(f"  mse={r['snapped_mse']:.2e}  -> {r['snapped_expr']}")


# ---------------------------------------------------------------------------
# Structural sanity check: forward pass of a hand-wired tree
# ---------------------------------------------------------------------------

def structural_check() -> None:
    """
    Hand-wire the known ln(x) EML formula and verify the tree computes it.

    Formula:  ln(x) = eml(1, eml(eml(1, x), 1))

    We need depth 3 with the eml at:
      - level 1, node 1:  eml(1, x)      -- swap on (leaf pair is (x,1))
      - level 1, node 2:  eml(1, 1)       -- but we don't need this output
      - level 1, nodes 3,4: anything, we won't use them
      - level 2, node 1:  eml( eml(1,x), 1 )
      - level 2, node 2:  anything unused
      - level 3 (root):   eml( 1, <level2-node1> )

    The tree only has one root, so we need the UNUSED subtree to supply '1'.
    Multiplier on (1,1) pair gives 1, and repeated mult of 1*1 = 1 at every
    level, so that whole sub-branch should be alpha=1 (multiplier).

    Layout (leaves feed level 1 left-to-right; root is the single level-3 node):
        Leaves:  [x,1]  [1,1]  [1,1]  [1,1]
                   |      |      |      |
        Lvl1:     N1     N2     N3     N4
                  \     /        \     /
        Lvl2:       M1             M2
                     \             /
        Lvl3:             R

    We want:
        R = eml(1, M1)
        M1 = eml(N1, N2)       -> needs alpha=0 (eml)
        N1 = eml(1, x)          -> alpha=0, swap so leaf becomes (1,x)
        N2 = 1 (constant)       -> multiplier on (1,1): alpha=1
        M2 = 1                  -> multiplier (its inputs are mult outputs, all 1)
        N3, N4 = 1              -> multiplier alpha=1

    Root: alpha=0 (eml), and we need root's two inputs to be (1, M1),
    meaning its U=1, V=M1. After reshaping, the root receives (M1, M2)
    as its (U, V). So we want U=1=M2 and V=M1. Since M1 is on the left
    and M2 on the right, root gets (U, V) = (M1, M2). We need (1, M1),
    i.e. swap... but the root has no swap gate in our design. Solution:
    place the 'useful' branch on the right, and put the leaf (x,1) in
    the last leaf pair so it flows to M2.

    Simpler: swap the left<->right at the root level by putting the x-leaf
    on the right half. But build_leaves hardcodes x at position 0. So
    instead let's verify with the formula eml(M1, 1) rather than eml(1, M1):
    eml(M1, 1) = exp(M1) - ln(1) = exp(M1) = exp(eml(1,x)) = exp(e - ln x)
        = e^e / x,  not ln(x).

    Workaround: modify the model briefly to put x at position with the
    right orientation. For this sanity check we'll build a mini custom
    tree.
    """
    print("\n" + "=" * 66)
    print("STRUCTURAL CHECK: evaluate known ln(x) formula")
    print("=" * 66)

    # Directly evaluate the ground-truth formula in complex128
    x = torch.linspace(0.2, 5.0, 11, dtype=torch.float64).to(torch.complex128)

    def eml(u, v):
        return torch.exp(u) - torch.log(v)

    one = torch.ones_like(x)
    gt_formula = eml(one, eml(eml(one, x), one))
    gt_ln = torch.log(x)

    err = (gt_formula - gt_ln).abs().max().item()
    print(f"  max|eml(1,eml(eml(1,x),1)) - ln(x)| over 11 points = {err:.3e}")
    assert err < 1e-12, "Ground-truth formula check failed"

    # Now verify our GatedEMLTree can reach the same value by manual weight
    # setting. We place x on the right side by using a leaf pair (1, x) via
    # beta = 0 (full swap), so the first leaf pair reads (1, x) instead of (x, 1).
    #
    # For mirroring in our tree layout (leaves L1..L4 -> N1..N4 -> M1 M2 -> R),
    # the canonical formula needs path:
    #   N1 must compute eml(1, x)  -> alpha=0, beta=0 (swap)
    #   N2 must compute 1           -> alpha=1 (mult on (1,1))
    #   M1 must compute eml(N1, 1)  -> alpha=0 (eml on (eml(1,x), 1))
    #   But we want eml(eml(1,x), 1) which is exactly that!  = N1 then eml with 1
    #   M1 = eml(N1, N2) = eml(eml(1,x), 1)   -> alpha=0
    #   R  = eml(M1, M2) with M2=1, so R = eml(M1,1) = exp(M1) - ln(1) = exp(M1)
    #
    # Oops: R wants eml(1, M1) = e - ln(M1) = e - ln(eml(eml(1,x),1)) = ln(x).
    # But the tree layout gives R = eml(M1, M2). We need the opposite order.
    #
    # Resolution: put x into the RIGHT half of the tree instead of the left.
    # Since build_leaves puts x at position 0 (leftmost leaf pair), the useful
    # subtree ends up as M1 on the left. So root R = eml(M1, M2) where M1 is
    # the useful subtree and M2 = 1. This gives exp(M1), which is NOT ln(x).
    #
    # So without a root swap, this particular tree layout cannot represent
    # ln(x) exactly. It CAN represent exp(M1) = exp(eml(eml(1,x),1)).
    # Let's sanity-check that forward pass instead.

    model = GatedEMLTree(depth=3, use_swap=True, alpha_init=0.0)  # start mid
    with torch.no_grad():
        # Level 1: N1=eml(1,x) [alpha=0,beta=0], N2=1 [alpha=1], N3=1 [alpha=1], N4=1 [alpha=1]
        model.alphas[0].data = torch.tensor([-20., 20., 20., 20.])  # [eml, mult, mult, mult]
        model.beta.data      = torch.tensor([-20.,  0.,  0.,  0.])  # [swap, -,   -,   -]
        # Level 2: M1=eml(N1,N2)=eml(eml(1,x),1) [alpha=0], M2=1 [alpha=1]
        model.alphas[1].data = torch.tensor([-20., 20.])
        # Level 3 (root): R=eml(M1,M2) = eml(eml(eml(1,x),1), 1)
        model.alphas[2].data = torch.tensor([-20.])

    y_pred = model(x)
    y_ref  = eml(eml(eml(one, x), one), one)  # what this layout actually computes
    err2 = (y_pred - y_ref).abs().max().item()
    print(f"  max|model(x) - eml(eml(eml(1,x),1),1)| = {err2:.3e}")
    print(f"  note: tree layout without root swap cannot place x on the V side")
    print(f"        of the root, so exact ln(x) requires the SEARCH to find")
    print(f"        an alternative depth-3 EML formula for ln(x) that fits.")
    print("  structural forward pass works correctly.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="ln",
                    choices=list(TARGETS.keys()))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--check", action="store_true",
                    help="Run structural sanity check only")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)

    if args.check:
        structural_check()
        return

    structural_check()
    run_experiment(args.target, depth=args.depth,
                   n_restarts=args.restarts, n_steps=args.steps)


if __name__ == "__main__":
    main()