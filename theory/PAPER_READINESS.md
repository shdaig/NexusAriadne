# LOKAN: Publication Readiness Gap Analysis

Comparison of the current NexusAriadne project state against the standard set by the original KAN paper (Liu et al., 2024).

---

## 1. KAN Paper Structure — What Makes It Publication-Quality

The KAN paper is organized around **three independent pillars**, each self-contained enough to justify publication on its own:

### Section structure
| Section | Content | Scientific contribution |
|---|---|---|
| **1 Introduction** | Motivation via Kolmogorov–Arnold theorem; KAN vs MLP design decision | Conceptual framing, identifies the research gap |
| **2 KAN: Kolmogorov–Arnold Networks** | Formal architecture definition; approximation theory (Theorem 2.1); B-spline parametrization; grid extension technique; simplified 2-layer motivating example | Mathematical foundation — makes the paper rigorous |
| **3.1 Accuracy** | Toy experiments, special functions, Feynman symbolic regression, PDE solving, continual learning without catastrophic forgetting | Empirical breadth — shows applicability beyond toy tasks |
| **3.2 Interpretability** | Supervised symbolic discovery (toy functions), unsupervised feature discovery (knot theory, Anderson localization) | Shows the "why KAN" advantage over black-box MLP |
| **4 Related Works** | Comparison to Kolmogorov–Arnold theorem work, B-splines in NNs, neuroevolution, symbolic regression, physics-informed NNs | Positions the work in literature |
| **5 Discussion** | Limitations (speed, hyperparameter sensitivity), failure cases, future directions | Scientific honesty |

### Key paper quality signals
1. **Formal theorem with proof** — Theorem 2.1 gives a convergence rate bound for KAN approximation. Without this, the paper is an engineering contribution, not a scientific one.
2. **Scaling laws** — Fig. 3 shows RMSE vs number of parameters on a log-log plot. This is the quantitative accuracy claim.
3. **Comparison table** — Table 1 (Feynman benchmark): KAN vs MLP, parameter counts, interpretability flag.
4. **Two orthogonal demonstrations** of interpretability: physics (knot invariants) and condensed matter (Anderson localization). These are domain expert collaborations, not synthetic demos.
5. **Ablation / failure case** — Section 5 honestly lists speed disadvantage and sensitivity to grid hyperparameters.
6. **Complete reproducibility** — all benchmark functions, seeds, training configs are specified.

---

## 2. LOKAN Project — Current Inventory

### Architecture (implemented, publication-ready)

| Component | File | Status |
|---|---|---|
| LOKANLayer: B-spline activations + logit routing | `lokan/LOKANLayer.py` | Complete |
| Soft product formula: `term = 1 − p + p·ỹ` | `lokan/LOKANLayer.py:208` | Complete |
| Empty-group bias fix: `w_g = 1 − ∏(1 − p_g)` | `lokan/LOKANLayer.py:217` | Complete |
| Temperature-softmax decode | `lokan/LOKANLayer.py:200` | Complete |
| Full LOKAN model (multi-layer, bias/scale nodes) | `lokan/LOKAN.py` | Complete |
| Grid extension / adaptive grid update | `lokan/LOKAN.py` + `spline.py` | Complete |
| Symbolic regression pipeline (fix/suggest/auto/formula) | `lokan/LOKAN.py` | Complete |
| Attribution (backward edge scores) | `lokan/LOKAN.py` | Complete |
| Pruning (node + edge + compound) | `lokan/LOKAN.py` | Complete |
| Sparsity regularization (`get_reg`) | `lokan/LOKAN.py` | Complete |
| Visualization: network diagram, heatmap, gallery | `lokan/nexus_plot.py` | Complete |

### Training infrastructure (implemented)

| Component | Status |
|---|---|
| Temperature annealing (τ: 2.0 → 0.1, exponential) | Implemented in demo notebook |
| Entropy regularization on ops (λ: 1e-2 → 1.0) | Implemented in demo notebook |
| LR schedule (0.1 → 1e-3, Adam) | Implemented in demo notebook |
| Sparsity regularization with `get_reg` | Implemented |

### Existing benchmarks

| File | Models compared | Status |
|---|---|---|
| `benchmarks_lite/sr_staged_regression.py` | **NexusKAN** vs MultKAN vs MLP vs PySR | Uses NexusKAN, **not LOKAN** |
| `benchmarks_lite/feynman_regression.py` | NexusKAN-based | Uses NexusKAN, **not LOKAN** |
| `benchmarks_lite/feature_ext_classification.py` | NexusKAN | Uses NexusKAN, **not LOKAN** |
| `lokan_prod_group.ipynb` | LOKAN on toy functions | Notebook demo, not a rigorous benchmark |

### Paper / writeup

| Item | Status |
|---|---|
| Russian-language paper draft (`paper.pdf`) | Exists, not peer-review ready |
| English-language paper | **Missing** |

---

## 3. Section-by-Section Gap Analysis

### Section 1: Introduction
**KAN paper**: Opens with the universal approximation theorem, then motivates KAN by asking "what if activations are on edges instead of nodes?" Closes with a summary of contributions.

**LOKAN status**:
- Motivation exists (CLAUDE.md, paper.pdf in Russian)
- **Gap**: No formal English introduction framing the *specific* research question: "KAN uses fixed summation; what if the aggregation operation is learned?" The transition from KAN to LOKAN must be stated as a scientific hypothesis, not an implementation choice.
- **Gap**: Need a clear contributions list (3–5 bullet points) stating exactly what LOKAN adds over KAN.

---

### Section 2: Architecture + Theory
**KAN paper**: Theorem 2.1 states the approximation rate. Section 2.4 proves the grid extension property. Section 2.5 shows a 2-layer example.

**LOKAN status**:
- Architecture is fully implemented and correct
- Empty-group bias fix is mathematically justified in CLAUDE.md
- **Gap (critical)**: No approximation theorem for LOKAN. Need at minimum one theoretical claim about LOKAN's representational capacity — e.g., that any function expressible as a KAN is also expressible as a LOKAN (LOKAN subsumes KAN when all logits route to the sum slot), and that LOKAN can additionally represent certain factored forms that KAN cannot represent without depth.
- **Gap**: No formal analysis of when temperature annealing converges to a discrete routing (i.e., connection to straight-through estimator / Gumbel-softmax literature).
- **Gap**: Need a motivating 2-layer worked example showing how LOKAN recovers `x0 * x1` vs `x0 + x1` from data.

---

### Section 3.1: Accuracy Benchmarks
**KAN paper**: Log-log scaling plots (RMSE vs #params) on toy functions; RMSE comparison table on 100 Feynman equations; PDE solving.

**LOKAN status**:
- **Gap (critical)**: All existing benchmark scripts (`sr_staged_regression.py`, `feynman_regression.py`) compare **NexusKAN** (the predecessor), not **LOKAN**. These must be rewritten/wrapped to use the `lokan` module.
- **Gap**: No scaling law plots for LOKAN. Need RMSE vs parameter count curves on at least 3–5 functions, compared to KAN (MultKAN) and MLP.
- **Gap**: No Feynman benchmark for LOKAN. The staged regression benchmark covers 4 hand-crafted expressions; need Feynman-100 or Feynman-20 with LOKAN.
- **Gap**: No PDE experiment. (Lower priority — the paper can omit this if the symbolic regression story is strong enough.)
- **Partial credit**: The staged regression benchmark concept (x0/x1, x0/x1+x2/x3, x0*x1+x2*x3, nested division) is well-designed and directly showcases LOKAN's advantage — this is the killer demo once it runs on LOKAN.

---

### Section 3.2: Interpretability
**KAN paper**: Two real-science applications from domain collaborators. This is the paper's strongest selling point.

**LOKAN status**:
- **Gap (critical)**: No real-science application. The notebook demonstrates recovery of synthetic functions, which is necessary but not sufficient.
- Minimum viable: Find one physics formula (e.g., from Feynman dataset: `F = q*E + q*v*B` or `E_kin = 0.5*m*v^2`) and show that LOKAN:
  1. Trains to high R²
  2. Prunes to the correct graph structure
  3. Symbolifies to the correct formula
  4. The operation routing (Σ vs Π nodes) matches the true mathematical structure
- **Partial credit**: `lokan_prod_group.ipynb` demonstrates recovery of simple product functions. Needs to be hardened into a reproducible benchmark with fixed seeds and tabulated results.

---

### Section 4: Related Works
**LOKAN status**:
- **Gap**: No written related works section. Need to situate LOKAN relative to:
  - KAN (Liu et al. 2024) — baseline and starting point
  - MultKAN / KAN 2.0 — direct predecessor (multiplicative nodes added as separate node type vs LOKAN's edge-level routing)
  - Gumbel-softmax / concrete distribution (Maddison et al., Jang et al.) — relevant to discrete operation learning
  - Neural Architecture Search / differentiable NAS — operation selection literature
  - Symbolic regression: PySR, EQL, DSR — competing approaches
  - Other learnable activation function work: Sinusoidal (SIREN), Swish, KAN variants (BKAN, WaveKAN, etc.)

---

### Section 5: Discussion / Limitations
**LOKAN status**:
- **Gap**: Need an honest limitations section covering:
  - Speed: LOKAN is slower than MLP (same as KAN)
  - Scaling: logit tensor size grows as `in_dim × out_dim × (in_dim//2 + 1)` — quadratic in `in_dim`
  - Hyperparameter sensitivity: τ schedule, λ schedule, number of mult groups G
  - Failure modes: when does operation routing fail to converge?

---

## 4. Priority Task List

### P0 — Blockers (paper cannot be submitted without these)

1. **Migrate all benchmarks from NexusKAN to LOKAN**
   - `sr_staged_regression.py`: replace `NexusKAN`/`NexusKANLayer` imports with `LOKAN`/`LOKANLayer`
   - `feynman_regression.py`: same migration
   - Verify results are at least as good as NexusKAN (they should be identical since the forward pass is the same)

2. **Write one end-to-end symbolic regression showcase**
   - Pick 2–3 Feynman formulas that mix multiplication and addition
   - Train LOKAN, show pruning, show symbolification, show that operation routing (logit argmax) matches the true formula structure
   - Report as a table: formula, LOKAN R², KAN R², MLP R², LOKAN symbolic output

3. **State the theoretical claim**
   - Write Proposition: "LOKAN with `G > 1` multiplicative groups can represent any function of the form `f = Σ_k ∏_i φ_{i,k}(x_i)` exactly with a 2-layer architecture."
   - This is almost certainly true given the architecture — needs a 1-page proof or proof sketch.

4. **Run the staged SR benchmark on LOKAN** and report results
   - The 4 stages directly test the architecture's raison d'être
   - Should show: LOKAN >> MLP on multiplicative stages, LOKAN ≥ KAN on additive stages

### P1 — High priority (needed for competitive submission)

5. **Scaling law plot**: RMSE vs number of parameters on `x^2`, `x*y`, `sin(x)*cos(y)`, and one Feynman formula. Log-log, comparing LOKAN vs KAN vs MLP.

6. **Ablation study**:
   - Temperature annealing: fixed τ=1.0 vs annealed τ
   - Entropy regularization: λ=0 vs λ schedule
   - Number of groups G: G=1 (degenerate, sum only) vs G=2 vs G=in_dim//2+1
   - This directly validates the design choices

7. **Empty-group bias fix ablation**: Compare old aggregation (`y = y_sum + Σ y_prod`) vs new (`y = y_sum + Σ w_g * y_prod`) — show that the fix removes the constant bias on functions without multiplicative structure.

8. **English paper draft**: At minimum an arXiv preprint structure (8–12 pages). The Russian `paper.pdf` can serve as the outline.

### P2 — Important for polish

9. **Continual learning experiment**: Train on function 1, then function 2. Show LOKAN retains grid extension benefit (same as KAN paper Section 3.1.5). This is a strong selling point that's cheap to implement.

10. **Visualization figures for the paper**:
    - Figure 1: Architecture diagram (LOKAN layer with routing arrows)
    - Figure 2: Comparison of KAN aggregation (fixed Σ) vs LOKAN (learned Σ/Π routing)
    - Figure 3: Operation routing convergence during training (entropy vs epoch)
    - Figure 4: Visualization of a trained LOKAN on a symbolic regression task (from `plot_nexus_network`)

11. **Related works review** (1–2 pages): Position LOKAN in the NAS/differentiable-programs literature, not just as a KAN variant.

12. **Real-science application**: Identify one physics/chemistry dataset where multiplicative structure is known. Show LOKAN discovers it automatically. (Feynman dataset is the natural choice.)

### P3 — Nice to have

13. **PDE experiment**: Show LOKAN on a simple PDE (e.g., Poisson equation). Demonstrates applicability beyond SR.

14. **Classification benchmark (feature extraction mode)**: Port `feature_ext_classification.py` to LOKAN.

15. **KAN + FC hybrid**: Document the `plot_kan_fc_architecture` use case for classification tasks.

---

## 5. Experiment Execution Checklist

```
[ ] 1. Port sr_staged_regression.py to LOKAN
        - Replace NexusKAN/NexusKANLayer with LOKAN/LOKANLayer
        - Run 3+ seeds, report mean±std R²
        - Expected: LOKAN matches or beats NexusKAN

[ ] 2. Port feynman_regression.py to LOKAN
        - Select 20+ Feynman formulas mixing + and *
        - Report R² and whether symbolic formula was recovered

[ ] 3. Symbolic showcase (end-to-end)
        - Functions: x0/x1, x0*x1+x2*x3, (x0/x1)*(x2+x3)
        - Train → prune → symbolify → check formula string matches
        - Visualize op routing with plot_nexus_network

[ ] 4. Scaling law measurement
        - Vary grid parameter (num) and/or width
        - Log-log RMSE vs #params on 3 functions
        - Compare LOKAN vs MultKAN vs MLP

[ ] 5. Ablation: τ schedule
        - Fixed τ=1.0 vs exponential τ_start=2→τ_end=0.1
        - Metric: R² on stages 3 and 4 of sr_staged_regression

[ ] 6. Ablation: λ schedule (entropy reg)
        - λ=0 vs λ_start=1e-2→λ_end=1.0
        - Same metric

[ ] 7. Ablation: bias fix
        - Old: y = y_sum + sum(y_prod)
        - New: y = y_sum + sum(w_g * y_prod)
        - Test on pure-additive function: bias should be ~K for old formula

[ ] 8. Convergence visualization
        - Track entropy of op logits per epoch
        - Show it decreasing (routing becoming discrete) with temperature annealing

[ ] 9. English paper draft
        - Sections: Introduction, Architecture+Theory, Experiments, Related Works, Discussion
        - Target: arXiv, then ICLR/NeurIPS workshop or main track
```

---

## 6. Key Mathematical Claims to Formalize

### Claim 1 — LOKAN subsumes KAN
**Statement**: A LOKAN layer with `G = 1` (single group = summation only, no multiplicative slots) reduces exactly to a KAN layer.

**Proof sketch**: When `G = 1`, `logits` has shape `(in_dim, out_dim, 1)`, `softmax` over a single slot = 1.0, so `sum_probs = 1` and `prod_probs` is empty. The output is `y = Σ_i ỹ_{i,o}` — standard KAN aggregation.

### Claim 2 — LOKAN represents factored sums
**Statement**: A 2-layer LOKAN can exactly represent any function of the form `f(x) = Σ_{k=1}^K ∏_{i ∈ S_k} φ_k(x_i)` where `S_k` are disjoint index sets and `φ_k` are univariate functions, provided `K ≤ G-1` (number of multiplicative slots in the hidden layer).

**Proof sketch**: In the second layer, route input `i` to group `k` with probability 1 (logit argmax). Each group computes `∏_{i ∈ S_k} ỹ_i`. The B-spline `ỹ_i` can approximate `φ_k(x_i)` to arbitrary precision by the standard B-spline approximation theorem. Sum over groups gives the factored sum.

### Claim 3 — Empty-group bias elimination
**Statement**: The original aggregation `z = y_sum + Σ_g ∏_i term_{i,g}` has an additive constant `K` when all routing probabilities are zero. The corrected aggregation `z = y_sum + Σ_g w_g ∏_i term_{i,g}` has zero contribution from empty groups.

**Proof**: When `p_{i,o,g} = 0` for all `i`, `term_{i,g} = 1`, so `∏_i term_{i,g} = 1`. With weight `w_g = 1 - ∏_i(1-0) = 0`, the contribution is `0 * 1 = 0`. QED.

---

## 7. Summary Assessment

| Dimension | Status |
|---|---|
| Core architecture implemented | ✅ Complete |
| Architecture mathematically justified | Partial — bias fix proven, approximation theorem missing |
| Benchmarks on LOKAN (not NexusKAN) | ❌ Missing — all scripts use NexusKAN |
| Symbolic regression showcase | Partial — notebook exists, not hardened |
| Scaling law plots | ❌ Missing |
| Ablation studies | ❌ Missing |
| Real-science application | ❌ Missing |
| English paper draft | ❌ Missing |
| Related works | ❌ Missing |
| Visualization figures (paper-quality) | Partial — tools exist, figures not generated |

**Bottom line**: The architecture and training infrastructure are complete and correct. The gap is entirely on the experimental validation and writing side. Approximately 4–6 weeks of focused work (2 weeks experiments + 2 weeks writing + 2 weeks revisions) to reach arXiv submission quality.

The single highest-leverage action is **migrating `sr_staged_regression.py` to LOKAN and running it** — this immediately produces the primary quantitative result that the entire paper is built around.
