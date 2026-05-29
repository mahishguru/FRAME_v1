# FRAME — Theoretical Foundations

This document details the scientific and theoretical foundations underpinning the FRAME benchmark: from the motivating problem in computational engineering, through the information-theoretic design of the benchmark schema, to the evaluation methodology.

---

## 1. Motivation: The FEA Interpretation Gap

### 1.1 What FEA Post-Processing Demands

Finite Element Analysis produces field solutions (stress, strain, temperature, displacement, velocity, etc.) discretized over complex geometries. Interpreting these results requires simultaneous mastery of:

- **Continuum mechanics** — reading stress tensor invariants (von Mises, principal stresses), understanding constitutive relations
- **Numerical methods** — recognizing mesh artifacts vs. physical phenomena (singularities at re-entrant corners, locking in incompressible media)
- **Domain physics** — identifying resonance, buckling modes, flow separation, thermal runaway, fatigue initiation
- **Design engineering** — translating observations into actionable modifications with quantitative parameters

No existing AI benchmark tests this composite skill. Standard VQA benchmarks (ChartQA, AI2D, ScienceQA) test recognition and arithmetic; they do not require *causal physical reasoning* about simulation outputs.

### 1.2 Why Standard Multimodal Benchmarks Fail

General benchmarks suffer from three limitations for this domain:

1. **Surface-level visual understanding** — Asking "what color is the maximum?" vs. "why does stress concentrate at this fillet and how should the geometry change?"
2. **No multi-figure reasoning** — FEA papers typically present multiple views (deformed/undeformed, contour/mesh, time series); understanding requires synthesis across figures
3. **No structured actionability** — Benchmark answers are text strings, not engineering specifications that could feed into a design loop

FRAME addresses all three by requiring multi-image input, structured multi-part output, and explicit physical reasoning.

---

## 2. Benchmark Design Theory

### 2.1 Information-Theoretic Schema

The FRAME benchmark decomposes the FEA interpretation task into an **input specification** (what defines the simulation) and an **output analysis** (what a senior engineer would conclude).

#### Input Schema (4 Keys)

The four input keys form a complete specification of a simulation problem following the structure of an engineering simulation setup document:

| Key | Content | Information-Theoretic Role |
|-----|---------|---------------------------|
| `key_1` | Identity, Material, Application, Geometry | **Domain embedding** — positions the problem in the space of engineering disciplines |
| `key_2` | Boundaries, Loads, Interactions, Simulation Type | **Constraint specification** — defines the mathematical boundary value problem |
| `key_3` | Physics Solved, Target Outputs, Optimization Goal | **Objective function** — what the simulation measures and why |
| `key_4` | Design Space, Performance Limits, Process Constraints | **Feasibility manifold** — bounds on acceptable solutions |

This decomposition is *informationally complete* in the sense that any well-posed FEA study can be fully characterized by these four categories. The schema follows the structure of the Partial Differential Equation (PDE) formulation:

$$
\mathcal{L}(u) = f \quad \text{in } \Omega \quad (\text{key\_3: physics})
$$
$$
\mathcal{B}(u) = g \quad \text{on } \partial\Omega \quad (\text{key\_2: boundaries})
$$
$$
\Omega \subset \mathbb{R}^d \text{ with material } M \quad (\text{key\_1: domain})
$$
$$
u \in \mathcal{U}_{admissible} \quad (\text{key\_4: constraints})
$$

#### Output Schema (2 Parts)

| Part | Content | Cognitive Skill Tested |
|------|---------|----------------------|
| `output_1` | Physical Behavior: dominant fields, critical phenomena, failure modes | **Diagnostic reasoning** — reading and interpreting contour plots |
| `output_2` | Optimization Strategies: specific modifications with location and rationale | **Prescriptive reasoning** — converting diagnosis to actionable design changes |

This mirrors the expert engineer's workflow: first *understand* what the simulation shows, then *prescribe* improvements.

### 2.2 Multi-Image Multimodal Formulation

Each benchmark instance provides:
- Two FEA figures (`fig_1`, `fig_2`) showing different aspects (e.g., stress contour + deformation, temperature field + cooling curve)
- The full 4-key textual context

The model must jointly reason over visual and textual modalities. This tests:
- **Cross-modal grounding** — linking "maximum stress at fillet" (text) to a red region in the contour plot (image)
- **Multi-view synthesis** — combining information from different plots (e.g., comparing stress before/after optimization)

### 2.3 Output Structured Generation

Unlike free-form QA, FRAME requires outputs in a specific structured format:

```
output_1:
  - Dominant Physical Fields (with extremes, locations, figure references)
  - Critical Phenomena & Patterns (with physical reasoning)
  - Failure Modes / Performance Limiting Factors

output_2:
  - Strategy Category (Geometric Change / Process Parameter / Material / Mesh Refinement)
  - Specific Modification (parameterized)
  - Location/Scope
  - Physical Rationale (the "why")
```

This structure enables both automated metric computation and downstream integration into design workflows.

---

## 3. Data Pipeline Theory

### 3.1 Three-Pass Visual Filtering

The filtering pipeline implements a precision-recall tradeoff optimized for the FEA domain:

**Pass 1 (Rule-based, high recall):** Structural check — papers must have both images and extractable text. This eliminates ~60% of irrelevant entries.

**Pass 2 (VLM classification, balanced):** Qwen2-VL-7B binary classification asking whether images contain "finite element contour plots showing a variable as color gradients over a part's surface." This targets the core visual signature of FEA results.

**Pass 3 (VLM classification, high precision):** Refined prompt distinguishing genuine FE plots from:
- CFD streamline visualizations without mesh-based coloring
- Experimental optical/thermal images
- Schematic diagrams with gradient fills
- Optimization convergence plots

The progressive refinement avoids false positives from colored scientific visualizations that superficially resemble FE contour plots.

### 3.2 LLM-Based Structuring

The data structuring pipeline uses Gemma-3-27B-IT in a **multi-turn conversation** paradigm:

1. First message provides full paper text + all figure-caption pairs + first prompt (key_1)
2. Subsequent messages build on conversation history, allowing the model to reference previously extracted information
3. Input keys are extracted first, then output keys in a separate conversation with different system prompt

This conversational approach ensures:
- **Coherence** — later keys can reference information established in earlier keys
- **Completeness** — the model has the full paper context available throughout
- **Consistency** — system prompts enforce the structured format

The system prompt engineering distinguishes between the *input extraction* role (factual, specification-focused) and the *output analysis* role (analytical, recommendation-focused).

---

## 4. Fine-Tuning Theory

### 4.1 LoRA for Domain Adaptation

Low-Rank Adaptation (LoRA) fine-tunes a small number of additional parameters:

$$
W' = W + \Delta W = W + BA
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

For FRAME, LoRA is applied because:
- The domain shift is *knowledge-based* (engineering terminology, structured output format) rather than *capability-based* (the base model already understands images)
- Full fine-tuning would risk catastrophic forgetting of general reasoning abilities
- The training set is small (~50-100 examples per set), making LoRA's regularization effect beneficial

### 4.2 Training Configuration Rationale

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 2×10⁻⁴ | Standard for LoRA; high enough for rapid adaptation without instability |
| Epochs | 3 | Prevents overfitting on small dataset while ensuring convergence |
| Batch size | 1 (effective 8 via accumulation) | Memory constraint with large multimodal models + multi-image inputs |
| DeepSpeed ZeRO | Stage 2 | Enables 12B model training across 4 GPUs with optimizer state sharding |

### 4.3 Two-Set Evaluation Design

The benchmark uses two independent test sets (`set1`, `set2`) with different train/test splits. This design:
- Tests **generalization** across different FEA domains (not just memorization of training papers)
- Enables **cross-validation** style analysis without data leakage
- Allows **ablation** by training on one set and testing on the other

---

## 5. Evaluation Theory

### 5.1 Metric Selection Rationale

No single metric captures all aspects of FRAME task quality. The six-metric suite tests complementary dimensions:

#### Surface-Level Overlap Metrics

**ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation):
$$
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}
$$

where $R_{lcs}$ and $P_{lcs}$ are recall and precision based on Longest Common Subsequence. Tests whether the model produces text that shares subsequences with the reference — a proxy for correct terminology and structural patterns.

**METEOR** (Metric for Evaluation of Translation with Explicit ORdering):
Extends exact matching with stemming, synonymy, and word order penalties. Important for FRAME because engineering text uses domain-specific terminology with many valid paraphrases (e.g., "stress concentration" ≈ "localized high stress" ≈ "stress riser").

#### Semantic Similarity Metrics

**Cosine Similarity** (embedding-based):
$$
\text{sim}(a, b) = \frac{\mathbf{e}_a \cdot \mathbf{e}_b}{||\mathbf{e}_a|| \cdot ||\mathbf{e}_b||}
$$

where $\mathbf{e}$ are sentence-transformer embeddings. Captures semantic equivalence even when surface forms differ substantially.

#### Factuality Metrics

**Chunk OT Coverage** (Optimal Transport):

Decomposes both reference and candidate into semantic chunks, then computes optimal transport between chunk embeddings:

$$
\text{Coverage} = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \max_{c \in \mathcal{C}} \text{sim}(r, c)
$$

This measures whether all factual claims in the reference are covered by some part of the generated text — critical for FRAME where missing a failure mode or optimization strategy is a serious omission.

**SCALE Coverage** (factuality via claim decomposition):

Decomposes the reference into atomic claims, then verifies each claim against the generated text. This is more granular than ROUGE — a single sentence might contain multiple facts, and SCALE scores each independently.

#### Holistic Quality

**LLM-as-Judge** (0–5 scale):

Uses a capable LLM (GPT-4 class) to holistically evaluate:
- Technical accuracy
- Completeness of physics analysis
- Quality of optimization recommendations
- Appropriate use of quantitative values

This captures quality dimensions that automated metrics cannot: whether the reasoning chain is physically plausible, whether recommendations are actionable, whether the analysis demonstrates genuine understanding.

### 5.2 Human Evaluation Design

The human evaluation follows established psychometric principles:

- **Likert scale (1–5)** — provides granularity without cognitive overload
- **Four criteria** — decomposes quality into independently assessable dimensions
- **Three experts** — enables inter-rater reliability calculation
- **Gwet's AC₂** — chosen over Cohen's κ because it is more robust when raters have high agreement (avoids the "kappa paradox" where high agreement with skewed marginals produces low κ)

$$
AC_2 = \frac{P_a - P_{e|\gamma}}{1 - P_{e|\gamma}}
$$

where $P_a$ is observed agreement and $P_{e|\gamma}$ is chance agreement under the ordinal weights model.

Mean AC₂ = 0.91 indicates "almost perfect" agreement, validating that the benchmark's ground truth is well-defined and reproducible across expert evaluators.

### 5.3 Temperature Ablation

Models are evaluated at multiple temperature settings (0.5, 0.6, 0.7, 0.8) to characterize the precision-creativity tradeoff:
- Lower temperatures favor factual accuracy but may produce generic or incomplete responses
- Higher temperatures increase diversity but risk hallucination in technical content

This ablation identifies the optimal operating point for engineering applications where factual accuracy is paramount.

---

## 6. Domain Coverage

The FRAME benchmark covers a broad spectrum of FEA applications, organized by physics type:

### Structural Mechanics
- Linear/nonlinear static analysis (stress, displacement)
- Topology optimization (SIMP, BESO)
- Fatigue and fracture mechanics (crack propagation, SIF)
- Buckling analysis

### Thermal Analysis
- Transient heat transfer (quenching, plasma spraying)
- Thermal-structural coupling (residual stress)
- Phase change and solidification

### Manufacturing Processes
- Metal forming (forging, extrusion, die compaction)
- Additive manufacturing (WAAM residual stress)
- Machining (chip formation, cutting forces)

### Multiphysics
- Electromagnetic-structural coupling (motor vibration)
- Fluid-structure interaction (wave-structure problems)
- Hygro-thermo-mechanical (composite degradation)

### Biomechanics
- Soft tissue mechanics (hyperelastic, visco-hyperelastic)
- Cervical spine analysis
- Fiber-reinforced biological composites

This breadth ensures the benchmark tests genuine physics understanding rather than pattern matching within a narrow domain.

---

## 7. Theoretical Contributions

### 7.1 Benchmark Design Principle

FRAME establishes that meaningful evaluation of AI in engineering requires:
1. **Structured output** — not free text, but decomposed into assessable components
2. **Multi-modal input** — visual (contour plots) + textual (problem specification)
3. **Physics-grounded reasoning** — every claim must include causal explanation
4. **Actionability** — outputs must be parameterized enough for a designer to implement

### 7.2 Domain Adaptation Efficiency

The fine-tuning results demonstrate that:
- Small models (7-12B) with domain-specific LoRA can approach or exceed much larger general-purpose models on domain tasks
- The improvement is most pronounced on *structured output quality* (LLM-as-Judge) rather than surface-level overlap (ROUGE)
- This suggests fine-tuning primarily teaches the *format and reasoning pattern* of expert analysis, rather than new factual knowledge

### 7.3 Metric Complementarity

The six-metric evaluation reveals that different metrics capture genuinely different quality dimensions:
- A model can score high on ROUGE but low on SCALE (correct terminology but wrong facts)
- A model can score high on Cosine Similarity but low on LLM-as-Judge (semantically similar but lacks depth)
- Only LLM-as-Judge correlates well with human expert ratings, suggesting it best captures engineering quality

---

## 8. Limitations and Future Work

### Current Limitations
- **Dataset size** — ~100 papers limits statistical power for rare engineering domains
- **English only** — excludes significant non-English FEA literature
- **Static images** — does not test interpretation of animated results or interactive 3D views
- **Single-step reasoning** — does not test iterative design optimization loops

### Future Directions
- **Agentic evaluation** — testing models in closed-loop design optimization
- **3D mesh understanding** — extending to volumetric field data
- **Uncertainty quantification** — requiring models to express confidence in their analysis
- **Multi-step workflows** — chaining problem formulation → simulation setup → result interpretation → redesign

---

## References

1. Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Gwet, K.L. "Handbook of Inter-Rater Reliability." 4th Edition, 2014.
3. Lin, C.Y. "ROUGE: A Package for Automatic Evaluation of Summaries." ACL Workshop, 2004.
4. Banerjee, S. & Lavie, A. "METEOR: An Automatic Metric for MT Evaluation." ACL Workshop, 2005.
5. Reimers, N. & Gurevych, I. "Sentence-BERT." EMNLP 2019.
6. Zheng, L. et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
7. Zienkiewicz, O.C. et al. "The Finite Element Method." 7th Edition, Elsevier, 2013.
