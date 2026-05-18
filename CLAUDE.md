# NexusAriadne — LOKAN Project

## Goal

Research project developing **LOKAN** (Learnable Operations KAN), a modification of Kolmogorov–Arnold Networks (KAN) for discovering multiplicative and additive dependencies in symbolic regression and automated feature search (when fully conected layers head is used). The modification replaces the fixed additive aggregation of KAN with a *learnable soft routing* of spline outputs into multiplication groups and one summation group, controlled by a trainable logit tensor decoded via temperature-softmax. This preserves KAN's interpretability (visualizable B-spline activations per edge) while adding automatic detection of which inputs should be multiplied vs. summed.

The paper describing the architecture is `paper.pdf` (Russian). The main demo notebook is `lokan_prod_group.ipynb` (with PDF companion `lokan_prod_group.pdf`).

---

## Module Map

### `lokan/` — primary module, the LOKAN architecture

| File | Purpose |
| --- | --- |
| `LOKAN.py` | Full LOKAN model (`nn.Module`): forward pass, symbolic regression API, attribution, visualization |
| `LOKANLayer.py` | Single LOKAN layer: B-spline activations + learnable operation logits |
| `nexus_plot.py` | Visualization engine: network diagram, operation heatmaps, activation gallery, KAN+FC hybrid diagram |
| `spline.py` | B-spline utilities: `extend_grid`, `coef2curve`, `curve2coef` |
| `Symbolic_KANLayer.py` | Symbolic front-end per edge: fits affine `c*f(ax+b)+d` forms |
| `utils.py` | `SYMBOLIC_LIB` (symbolic function library), `sparse_mask`, singularity-protection wrappers |
| `__init__.py` | Re-exports everything from `LOKAN.py` |

### `nexuskan/` — original KAN + NexusKAN predecessor

| File | Purpose |
| --- | --- |
| `MultKAN.py` | Original MultKAN model (upstream KAN 2.0 implementation) |
| `KANLayer.py` | Original KAN layer |
| `NexusKAN.py` | NexusKAN: predecessor to LOKAN with the same logit structure but extra legacy methods |
| `NexusKANLayer.py` | NexusKAN layer (logit structure identical to LOKANLayer) |
| `MLP.py` | MLP baseline |
| `nexus_plot.py` | Older visualization for NexusKAN |
| `compiler.py` / `hypothesis.py` | Expression tree utilities |
| `LBFGS.py` | LBFGS optimizer |
| `feynman.py` | Feynman dataset loader |
| `experiment.py` | Experiment helpers |

> **NexusKAN vs LOKAN**: NexusKAN is the predecessor. The logit tensor and forward computation are identical, but NexusKAN contains extra methods (checkpoint saving, fit loop, etc.) that are not needed for the current research stage. LOKAN is the clean, current architecture.

### `benchmarks/benchmarks_lite/` — benchmark scripts

| File | What it benchmarks |
| --- | --- |
| `sr_staged_regression.py` | NexusKAN vs MultKAN vs MLP vs PySR on 4 staged symbolic regression tasks (x0/x1, x0/x1+x2/x3, x0*x1+x2*x3, nested division) |
| `feynman_regression.py` | Feynman symbolic regression dataset |
| `feature_ext_classification.py` | Feature extraction + classification benchmark |
| `kan_fc_nonlinear.py` | KAN + fully-connected hybrid benchmark |
| `visualize_nexuskan.py` | Visualization demo for NexusKAN |

---

## Architecture: LOKAN Layer

**Logit tensor** shape: `(in_dim, out_dim, G)` where `G = in_dim // 2 + 1`.

- Slots `0 .. G-2` → **multiplicative groups**
- Slot `G-1` (last) → **summation group**

**Forward computation** for each output neuron `o`:

1. Compute scaled spline output for each input `i`: `ỹ_{i,o}` (B-spline + SiLU residual)
2. Decode probabilities: `p = softmax(logits / τ, dim=-1)` — shape `(in_dim, out_dim, G)`
3. Multiplicative groups `g = 0..G-2`:
   `term_{i,o,g} = 1 - p_{i,o,g} + p_{i,o,g} * ỹ_{i,o}`
   `prod_{o,g} = ∏_i term_{i,o,g}`
4. Summation group: `sum_o = Σ_i p_{i,o,G-1} * ỹ_{i,o}`
5. Output: `z_o = sum_o + Σ_g w_{o,g} * prod_{o,g}`
   where `w_{o,g} = 1 - ∏_i (1 - p_{i,o,g})` — probability that group `g` has at least one active input.

When `p_{i,o,g} = 0`: term = 1 (neutral for product). When `p_{i,o,g} = 1`: term = ỹ_{i,o}.

**Note on empty-group bias fix:** The original aggregation `z_o = sum_o + Σ_g prod_{o,g}` had a constant bias of `+K` (where `K = G-1` is the number of multiplicative groups): an empty group contributes `∏_i 1 = 1` regardless of inputs, so with large `in_dim` the bias grows as `in_dim // 2`. The weight `w_{o,g}` suppresses empty groups to zero — when all `p_{i,o,g} = 0`, `w_{o,g} = 0` and the group contributes nothing.

---

## Training Optimizations for Discrete Structure Discovery

The hard combinatorial landscape of operation assignments requires several techniques applied together:

| Technique | Parameters | Purpose |
| --- | --- | --- |
| **Temperature annealing** | `τ_start=2.0 → τ_end=0.1` over T epochs, exponential | Softmax transitions from soft distribution to near-discrete assignment |
| **Entropy regularization on operations** | `λ_start=1e-2 → λ_end=1.0`, exponential | Penalizes high-entropy (ambiguous) operation assignments, promotes sharp grouping |
| **Learning rate schedule** | `lr_start=0.1 → lr_end=1e-3`, exponential (Adam) | Coarse exploration early, fine refinement later |
| **Small mini-batch** | batch=64 | Stochastic gradient noise helps escape flat regions in the loss landscape |
| **Sparsity regularization** | `get_reg()` with L1 + row/col entropy on activation scales | Prunes weak edges, improves interpretability |
| **Wide architecture** | `width=[d_in, d_in, 1]` or wider | Increases probability of finding the right multiplicative structure |

Entropy regularization formula (averaged over all layers):
`H_ops = -1/(d_in * d_out) * Σ_{i,o,g} p_{i,o,g} * log(p_{i,o,g} + ε)`

---

## LOKAN API Reference

### Construction

```python
from lokan import LOKAN

model = LOKAN(
    width=[5, 5, 1],   # plain ints recommended (no explicit mult nodes)
    grid=5, k=3,
    noise_scale=0.3,
    base_fun='silu',   # 'silu' | 'identity' | 'zero'
    seed=42,
    device='cpu',
)
```

### Key methods

| Method | Description |
| --- | --- |
| `forward(x)` | Forward pass; caches activations when `save_act=True` |
| `update_grid(x)` | Refine B-spline grids to data quantiles |
| `get_reg(metric, lamb_l1, lamb_entropy)` | Differentiable sparsity regularization term |
| `attribute()` | Backward attribution scores for nodes/edges → `model.feature_score` |
| `plot(tau_decode=0.05, ...)` | Network diagram with operation-colored edges (blue=sum, others=mult groups) |
| `fix_symbolic(l, i, j, fun_name)` | Lock edge to a symbolic function from `SYMBOLIC_LIB` |
| `auto_symbolic()` | Automatically fit all active edges to best symbolic functions |
| `suggest_symbolic(l, i, j)` | Ranked candidates for one edge |
| `symbolic_formula(var=...)` | Extract closed-form sympy expression after symbolification |
| `set_mode(l, i, j, mode)` | Switch edge mode: `'n'` (spline), `'s'` (symbolic), `'sn'` (both) |
| `disable_symbolic_in_fit(lamb)` | Speed up training: temporarily skip symbolic branch |

`LOKAN` is also exported as `loKAN`.

### Visualization functions (`lokan/nexus_plot.py`)

| Function | Output |
| --- | --- |
| `plot_nexus_network(model, tau_decode=0.05)` | Full network with colored edges (dominant operation) and Σ/Π node annotations |
| `plot_ops_heatmap(model)` | Per-layer heatmap of P(sum) |
| `plot_activation_gallery(model)` | Grid of all B-spline curves |
| `plot_kan_fc_architecture(kan, fc)` | Unified KAN+FC flow diagram |

---

## Typical Training Loop

```python
import torch
import torch.optim as optim
from lokan import LOKAN
from nexuskan import NexusKANLayer  # or LOKANLayer from lokan

model = LOKAN(width=[d_in, d_in, 1], grid=5, k=3, seed=0, device=device)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

tau_start, tau_end = 2.0, 0.1
lr_start,  lr_end  = 0.1, 1e-3
lamb_start, lamb_end = 1e-2, 1.0
EPOCHS = 200

for epoch in range(EPOCHS):
    # update schedules
    t = epoch / max(EPOCHS - 1, 1)
    new_tau  = tau_start  * (tau_end  / tau_start)  ** t
    new_lr   = lr_start   * (lr_end   / lr_start)   ** t
    new_lamb = lamb_start * (lamb_end / lamb_start) ** t

    for pg in optimizer.param_groups:
        pg['lr'] = new_lr
    for m in model.modules():
        if hasattr(m, 'tau'):
            m.tau = new_tau

    for batch_x, batch_y in dataloader:
        out  = model(batch_x)
        loss = criterion(out, batch_y)

        # entropy regularization on operation logits
        ent_ops, n_ops = 0.0, 0
        for m in model.act_fun:
            probs = torch.softmax(m.logits / m.tau, dim=-1)
            ent_ops += (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean()
            n_ops += 1
        ent_avg = ent_ops / n_ops

        # sparsity regularization
        sparsity = model.get_reg('edge_forward_spline_n', lamb_l1=1.0, lamb_entropy=2.0)

        total = loss + new_lamb * ent_avg + 1e-3 * sparsity
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
```

---

## Key Notebooks

| Notebook | Purpose |
| --- | --- |
| `lokan_prod_group.ipynb` | **Main demo**: LOKAN training with full optimization pipeline, visualization, symbolic regression |
| `lokan_prod_group.pdf` | Rendered version of the above for reading without Jupyter |
| `nexuskan_prod_group.ipynb` | Analogous demo for NexusKAN (older) |
| `nexuskan_prod_group_mlp_viz.ipynb` | NexusKAN + MLP visualization comparison |
| `kan_multiplication.ipynb` | Experiments on multiplication learning |
| `kan_classification.ipynb` | Classification experiments |
| `kan_regularization.ipynb` | Regularization study |
| `kan_optimizers_fc_sigmoid.ipynb` | Optimizer experiments with FC+sigmoid hybrid |

---

## Modules to Ignore

- `experimental/` — early experimental code, not part of the active research
- `local_radon/` — unrelated module

---

## Dependencies

PyTorch, NumPy, Matplotlib, SymPy, scikit-learn, pandas, PyYAML, tqdm.
Optional: PySR (for benchmark comparison).
