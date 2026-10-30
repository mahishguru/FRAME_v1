## Epistemic Uncertainty Quantification

## for Pre-trained VLMs via Riemannian Flow Matching

```
Li Ju^1 Mayank Nautiyal^1 Andreas Hellander^1 Ekta Vats^1 Prashant Singh^2
```
## Abstract

```
Vision-Language Models (VLMs) are typically
deterministic in nature and lack intrinsic mech-
anisms to quantify epistemic uncertainty, which
reflects the model’s lack of knowledge or igno-
rance of its own representations. We theoretically
motivate negative log-density of an embedding as
a proxy for the epistemic uncertainty, where low-
density regions signify model ignorance. The pro-
posed method REPVLM computes the probabil-
ity density on the hyperspherical manifold of the
VLM embeddings using Riemannian Flow Match-
ing. We empirically demonstrate that REPVLM
achieves near-perfect correlation between uncer-
tainty and prediction error, significantly outper-
forming existing baselines. Beyond classification,
we also demonstrate that the model also provides
a scalable metric for out-of-distribution detection
and automated data curation.
```
## 1. Introduction

```
Vision-Language Models (VLMs) like CLIP (Radford et al.,
2021), BLIP (Li et al., 2022), and SigLIP (Zhai et al.,
2023) have significantly advanced cross-modal represen-
tation learning by aligning visual and textual data through
large-scale contrastive pre-training. Their ability to map
diverse modalities into a shared embedding space has en-
abled powerful zero-shot generalization across a variety
of downstream tasks (Zhou et al., 2023; Wortsman et al.,
2022). However, a fundamental limitation of these deter-
ministic pre-trained models is that they map inputs to sin-
gle, fixed points in a high-dimensional embedding space.
This approach provides no intrinsic mechanism to represent
the model’s internal uncertainty of its own representations
(Chun et al., 2021; Chun, 2024; Chun et al., 2025).
While recent uncertainty quantification (UQ) frameworks
```
(^1) Department of Information Technology, Uppsala University,
Uppsala, Sweden^2 SciLifeLab, Uppsala University, Uppsala, Swe-
den. Correspondence to: Li Ju <li.ju@it.uu.se>.
Preprint. January 30, 2026.
for VLMs have emerged, they predominantly target aleatoric
uncertainty (data ambiguity) through probabilistic embed-
dings (Chun et al., 2025; Baumann et al., 2024; Ju et al.,
2025; Venkataramanan et al., 2025). Existing strategies for
epistemic uncertainty (model ignorance), such as deep en-
sembles (Lakshminarayanan et al., 2017) or Monte Carlo
dropout (Gal & Ghahramani, 2016), remain either compu-
tationally prohibitive for large-scale pre-trained models or
yield suboptimal, batch-dependent estimates. Thus, there is
a pressing need for a scalable, intrinsic metric that directly
quantifies a model’s confidence of its own representations.
We propose a principled method to quantify the epistemic
uncertainty of pre-trained VLMs by estimating the prob-
ability densityp(z)of the embeddings directly on their
native manifold. We posit that a low density indicates a re-
gion sparsely populated by the training data, signaling high
epistemic uncertainty. Thus the negative log-likelihood,
− logp(z)can serve as a robust and intrinsic uncertainty
score. Further, we propose REPVLM (illustrated in Fig-
ure 1), a unified Conditional Riemannian Flow Matching
(CRFM) framework that leverages the hyperspherical geom-
etrySd−^1 of VLM embedding spaces to learn a modality-
conditioned vector field, which can be used to compute the
log-density of the embeddings.
Our contributions are summarized as follows:

- Theoretical Grounding. We establish a theoretical
    foundation demonstrating that the negative log-density
    − logp(z)can serve as a principled proxy for model con-
    fidence based on training data coverage.
- Manifold-Native Modeling. We propose REPVLM, an
    epistemic UQ method that extends Riemannian Flow
    Matching to embedding spaces for scalable and intrin-
    sic probability density computation for both visual and
    textual modalities.
- Empirical Validation. We demonstrate that REPVLM
    significantly outperforms baselines in selective classifi-
    cation, achieving near-perfect correlation with model er-
    ror, and provides a reliable metric for out-of-distribution
    (OOD) detection and automated data curation.

```
Outline Section 2 reviews related work in UQ for VLMs
and Flow Matching (FM). Section 3 establishes the theoret-
```
# arXiv:2601.21662v1 [cs.LG] 29 Jan 2026


## CLIP

```
What a great
day!
```
## REPVLM

```
Twas brilg,
slithy toves.
```
```
Text
Encoder
```
```
Image
Encoder
```
```
Figure 1. Overview of REPVLM. The framework estimates the probability densityp(z)of pre-trained VLM embeddings on the
hypersphereSd−^1. A unified model learns a vector fieldvtthat transports a simple uniform base distributionP 0 = Unif(Sd−^1 )to the
empirical modality-specific distributionsP 1 (Image and Text). As illustrated, standard inputs map to high-density regions (yellow), while
ambiguous or out-of-distribution inputs such as the distorted image or nonsensical text reside in low-density regions (purple). The negative
log-likelihood− log p(z|c) thus serves as a principled proxy for epistemic uncertainty Uep(z), reflecting the model’s confidence.
```
```
ical link between latent density and epistemic uncertainty.
Section 4 introduces REPVLM, detailing the geometry-
aware problem formulation and training objective. Section
5 presents empirical results and Section 6 concludes with a
discussion of limitations.
```
## 2. Related Work

VLMs Modern VLMs learn a shared semantic embedding
space by aligning visual and textual representations. Foun-
dational models such as CLIP (Radford et al., 2021), BLIP
(Li et al., 2022), and SigLip (Zhai et al., 2023) achieve this
through large-scale contrastive pre-training, optimizing an
objective based on the similarity between paired image and
text embeddings. While this paradigm has demonstrated
powerful zero-shot generalization, a fundamental structural
limitation is that these models typically map inputs to single,
fixed points in the embedding space. This approach provides
no intrinsic mechanism to represent the model’s internal
confidence or uncertainty regarding its own representations.
Consequently, it is difficult to assess the reliability of VLM
outputs, particularly when the model is presented with am-
biguous or out-of-distribution samples that it may not truly
“understand” (Zhang et al., 2024).

```
UQ for VLMs Table 1 compares existing methods for UQ
in VLMs. Current methods predominantly address aleatoric
uncertainty (data ambiguity). This is typically achieved by
learning probabilistic embeddings, either by training models
from scratch (Chun et al., 2021; Chun, 2024; Chun et al.,
2025) or by adapting pre-trained ones (Upadhyay et al.,
2023; Baumann et al., 2024; Ju et al., 2025; Venkataramanan
et al., 2025). While effective for modeling data ambiguity,
```
```
these approaches lack a principled measure of epistemic
uncertainty for embeddings (model confidence).
Alternatively, Gomez (2025) quantifies uncertainty in cross-
modal retrieval by measuring the minimal perturbation re-
quired to alter the top-1 result. While providing a practical
lens for retrieval tasks, it introduces certain challenges for
general-purpose UQ. Specifically, because the estimate is
influenced by the particular batch of target embeddings used
as candidates, it is not an intrinsic property of the query
itself. Ideally, a measure of uncertainty should depend ex-
clusively on the input and the model’s state. Furthermore,
the theoretical mapping of this perturbation-based metric
to formal definitions of epistemic or aleatoric uncertainty
remains an area for further clarification. To the best of our
knowledge, ProbVLM (Upadhyay et al., 2023) is one of
few existing approaches for quantifying the epistemic un-
certainty of pre-trained VLMs. Alternatively, by enabling
Monte Carlo dropout of the pre-trained models during the in-
ference stage with multiple passes, the epistemic uncertainty
can also be estimated through the variance of the resulting
embeddings (Gal & Ghahramani, 2016).
```
```
Normalizing Flows Chen et al. (2018) offer exact density
estimation by transforming simple base distributions into
complex distributions through invertible mappings. How-
ever, the requirement for a tractable Jacobian determinant
imposes significant computational cost for training (Grath-
wohl et al., 2018). Flow Matching (Lipman et al.) has
emerged as a more efficient, simulation-free method for
training Continuous Normalizing Flows. Instead of learn-
ing an invertible map, FM learns a vector field that defines
a probability flow between distributions, improving train-
ing stability and efficiency. This framework retains the
```

Table 1. Comparison of methods for uncertainty quantification for
vision language models.

```
Method Post-hoc Ual(z) Uep(z) Based on
Full Dropout✓✗✓ MC Dropout
PFE✗✓✗ Prob. Embed.
PCME+✗✓✗ Prob. Embed.
ProLIP✗✓✗ Prob. Embed.
BayesVLM✓✓✗ Prob. Embed.
AsymVLM✓✓✗ Prob. Embed.
Adv. Pert✓?? Perturbation
GroVE✓✓✗ Prob. Embed.
```
```
ProbVLM✓✓✓
Prob. Embed.
+MC Dropout
Ours✓✗✓ Prob. Density
```
ability to compute exact log-likelihoods by integrating the
divergence of the learned vector field along an ordinary dif-
ferential equation trajectory. Crucially, its recent extension
Riemannian Flow Matching (Chen & Lipman, 2024) gener-
alizes this approach to non-Euclidean manifolds, allowing
for probability modeling directly on the hypersphereSd−^1
where most VLM embeddings reside.

## 3. Rethinking Uncertainty Quantification

While the distinction between aleatoric and epistemic uncer-
tainty is well-established (Hullermeier & Waegeman, 2021), ̈
existing VLM frameworks struggle to decouple them ef-
fectively. Current state-of-the-art methods predominantly
focus on aleatoric uncertainty by learning probabilistic em-
beddings, i.e., mapping inputs to distributions to capture
data ambiguity. However, these methods often fail to pro-
vide a calibrated measure of the model’s own “ignorance”
(Upadhyay et al., 2023; Baumann et al., 2024).

Standard epistemic estimators like Bayesian Neural Net-
works (Blundell et al., 2015) or deep ensembles (Lakshmi-
narayanan et al., 2017) remain computationally prohibitive
at modern VLM-scale. Meanwhile, common approxima-
tions such as Monte Carlo Dropout (Upadhyay et al., 2023;
Srivastava et al., 2014) or heuristic perturbations (Gomez,
2025) often yield suboptimal or batch-dependent estimates.

```
We propose an alternative hypothesis: Epistemic uncer-
tainty can be treated as an intrinsic property of the em-
bedding manifold’s geometry. Estimating the probability
density of the VLM’s embedding allows us to move away
from expensive parameter-space sampling, toward a direct
measure of confidence.
```
In this setting, an embedding located in a region sparsely
populated by training data (lowp(z)) is a direct indica-
tor of high epistemic uncertainty. Thus, the negative log-
likelihood,− logp(z), serves as a principled, scalable proxy
for the model’s epistemic uncertainty.

```
3.1. Theoretical Motivation
```
```
We now provide a theoretical motivation for using the nega-
tive log-density of an embedding,− logp(z), as a principled
proxy for epistemic uncertaintyUep(x). Our argument es-
tablishes an inverse relationship between the embedding
density and the models’ internal confidence.
Definition 3.1 (Epistemic Uncertainty). The epistemic un-
certainty of a Bayesian encoderf (x;θ)for a given inputx
is defined as the trace of the covariance matrix of its em-
beddingz = f (x;θ)over the posterior distribution of the
parameters p(θ|D):
```
```
Uep(x) = Tr(Covp(θ|D)(z))
= Tr
```
### 

```
Ep(θ|D)
```
### 

```
(f (x;θ)− ̄)(z f (x;θ)− ̄)z⊤
```
### 

### ,

```
where ̄ =z Ep(θ|D)[f (x;θ)]represents the mean embedding
over the parameter posterior p(θ|D).
Assumption 3.2 (Local Linearity of the Encoder). We then
assume that the encoder functionf (x;θ)is locally linear
with respect to its parametersθwithin the high-density re-
gion of the posterior surrounding the maximum a posteriori
estimateθ∗. This can be expressed via a first-order Taylor
expansion around θ∗:
```
```
f (x;θ)≈ f (x;θ∗) + Jθ∗(x)(θ− θ∗),
```
```
whereJθ∗(x) =∂f(∂θx;θ)
θ=θ∗is the Parameter-Jacobian of
the encoder at θ∗.
```
```
Covariance to Jacobian Norm Given that the expectation
of the linearized function isEp(θ|D)[f (x;θ)] ≈ f (x;θ∗),
under Assumption 3.2 the covariance of the embeddingzis:
```
```
Cov(z)≈ Ep(θ|D)
```
### 

```
(Jθ∗(x)(θ− θ∗))(Jθ∗(x)(θ− θ∗))⊤
```
### 

```
= Jθ∗(x)ΣθJθ∗(x)⊤,
```
```
whereΣθ= Cov(θ)is the parameter covariance. The epis-
temic uncertainty, given by the trace of this matrix, is thus
proportional to a quadratic form of the Parameter-Jacobian:
Uep(x)≈ Tr(Jθ∗(x)ΣθJθ∗(x)⊤). This establishes our first
link: high epistemic uncertainty corresponds to a large
Parameter-Jacobian norm, indicating that the model’s output
is sensitive to perturbations in its learned parameters.
```
```
Jacobian Norm to Data Density The second link in our
reasoning connects the magnitude of the Parameter-Jacobian
to the training data densityp(x). We consider the encoder
trained by minimizing a riskR = Ex∼Px[ℓ(f (x;θ),x)]. At
the optimal parametersθ∗, the first-order optimality condi-
tion at θ∗,∇θR(θ∗) = 0 must hold. This implies:
```
```
Ex∼Px[(∇zℓ)⊤Jθ(x)] =
```
### Z

```
(∇zℓ)⊤Jθ(x)p(x)dx = 0.
```

To satisfy this condition, the optimization must suppress
the term(∇zℓ)⊤Jθ(x)in regions where the data density
p(x)is high. For typical in-distribution data, this forces the
Parameter-Jacobian norm∥Jθ(x)∥Fto be small.

Conversely, for an out-of-distribution samplexoutwhere
p(xout) ≈ 0 , its contribution to the integral is negligible.
As a result, the Jacobian norm∥Jθ(xout)∥Fremains uncon-
strained and is typically much larger than for in-distribution
samples. This establishes that low data density leads to high
model sensitivity (large Parameter-Jacobian norm).

Data Density to Embedding Density The final step trans-
lates input densityp(x)to computable embedding den-
sityp(z). The mappingz = f (x;θ∗)induces the distri-
butionPzvia the change of variables formula: p(z) =
p(x)| det(Jx(x))|−^1 , whereJx(x)is the input-Jacobian of
the encoder. In contrastive learning, objectives emphasizing
alignment and uniformity encourage the encoder to act as
a local isometry, achieving a state of dynamical isometry
where the network preserves the local geometric volume of
the data manifold (Wang & Isola, 2020; Saxe et al., 2014).
Under this condition,Jx(x)is nearly orthogonal, imply-
ing| det(Jx(x))|≈ 1. Consequently, for well-regularized
VLMs, the latent density serves as a direct proxy for input
density: p(z)∝ p(x).

The Full Chain of Reasoning Combining the established
relationships, we arrive at a complete theoretical path that
connects the measurable embedding densityp(z)directly to
the formal epistemic uncertainty Uep(x):

1. Low Epistemic Uncertainty ⇐⇒ Small Parameter-
    Jacobian: Uep(x)∝∥Jθ(x)∥^2 F.
2. Small Parameter-Jacobian ⇐⇒ High Data Density:
    The training objective minimizes∥Jθ(x)∥Fwherep(x)
    is high.
3. High Data Density ⇐⇒ High Latent Density: For
    well-behaved mappings, p(x)∝ p(z).

The resulting relationship can thus be expressed as:

```
↑ p(z) ⇐⇒↑ p(x) ⇐⇒ ↓∥Jθ(x)∥F ⇐⇒↓ Uep(x).
```
This establishes that the negative log-density is a well-
founded proxy for epistemic uncertainty. It allows us to
quantify the model’s confidence without the intractable com-
putation of the parameter posterior for large-scale VLMs.

3.2. Why Flow Matching

To estimate the density over the VLM’s embedding space,
the chosen method must satisfy three critical criteria: scal-
ability to high dimensions, efficiency in evaluation, and
geometric compatibility with the manifold. Flow Matching

```
(Lipman et al.) and its Riemannian extension (RFM) (Chen
& Lipman, 2024) are uniquely suited for this task as:
```
- Unlike standard Normalizing Flows, FM avoids costly
    Jacobian determinant computations. It enables efficient
    training for high-dimensional VLM embeddings by learn-
    ing a vector field.
- RFM models densities directly on the hypersphereSd−^1.
    Utilizing geodesic interpolation rather than Euclidean
    paths allows the model to respect the intrinsic geome-
    try of the semantic space.
- RFM computes the exact log-density by integrating the
    divergence of the vector field. Leveraging Hutchinson’s
    trace estimator, the integration can be approximated by a
    few steps, ensuring minimal inference overhead.

## 4. Method

```
We propose REPVLM, a framework to quantify epistemic
uncertainty by estimating the probability density of pre-
trained VLM embeddings. Building upon CRFM, we learn
the embedding distributions of both modalities within a
unified neural network.
```
```
4.1. Problem Formulation and Geometry
```
```
LetZ ⊆ Sd−^1 denote thed-dimensional shared embedding
space of a pre-trained VLM. For inputx, the encoderfθ
produces a ℓ− 2 normalized embedding z = fθ(x).
Following the theoretical grounding in Section 3, we de-
fine the epistemic uncertaintyUep(x)as the negative log-
likelihood of the embedding conditioned on its modality:
Uep(x) =− logp(z | c),
wherec∈{ 0 , 1 }serves as a discrete conditioning variable
(e.g.,c = 0for images,c = 1for text). To maintain
geometric consistency, all updates must respect the tangent
spaceTzSd−^1 = {v ∈ Rd: ⟨v,z⟩ = 0}via the projection
operator ΠTz(v) = (I − zz⊤)v.
```
```
4.2. Conditional Riemannian Flow Matching
We define a conditional probability pathpt(z | c)fort ∈
[0, 1]that transforms a uniform base distribution into the
target empirical distribution for modality c.
```
```
4.2.1. UNIFIED CONDITIONAL RIEMANNIAN FLOW
```
```
We parameterize a single time-dependent vector field
vt(z,c;φ)using a neural network with the embeddingz,
timetand modality indicatorcas inputs. The flow is defined
by the conditional Ordinary Differential Equation (ODE):
dzt
dt
```
```
= vt(zt,c;φ), z 0 ∼ p 0.
```
```
The boundary conditions are defined as follows:
```

- Base Distributionp 0 : A modality-agnostic uniform dis-
    tribution over the hypersphere, p 0 = Unif(Sd−^1 ).
- Target Distributionp 1 = p(z | c): The empirical
    distribution of VLM embeddings for modalityc, esti-
    mated from a proxy dataset of image-caption pairs (i.e.,
    p(z | c = 0) = pimage(z)andp(z | c = 1) = ptext(z)).
    Crucially, this proxy dataset is distinct from downstream
    task, enabling task-agnostic uncertainty estimation.

```
4.2.2. GEODESIC CONDITIONAL PATH
```
Unlike Euclidean flow matching, we construct paths along
geodesics. Given a base samplez 0 ∼ p 0 and a target sample
z 1 ∼ p(z | c), the sample-conditioned path follows:

```
zt=
```
```
sin((1− t)θ)
sinθ
```
```
z 0 +
```
```
sin(tθ)
sinθ
```
```
z 1 ,
```
whereθ = arccos(⟨z 0 ,z 1 ⟩)is the geodesic distance be-
tween the two points. The target vector fieldut(zt| z 1 )is
the tangent to the geodesic at time t (in closed-form):

```
ut(zt| z 1 ) =
θ
sinθ
```
### 

```
cos(tθ)z 1 − cos((1− t)θ)z 0
```
### 

### .

### 4.2.3. TRAINING OBJECTIVE

The unified network is trained using the RFM objective,
regressing the vector field onto the target ut(zt| z 1 ,c):

L(φ) = Ep(c),p(t),p(z 1 ),p(zt)

### 

```
∥vt(z,c;φ)−ut(z | z 1 ,c)∥^2
```
### 

### ,

wherec∼ Bernoulli(0.5),t∼ Unif([0, 1]),z 1 | c∼ p(z |
c)andzt∼ pt(zt| c). By conditioning onc, the networkφ
learns to disentangle the text and image embedding geome-
tries while sharing parameters where appropriate, since both
modalities are designed to embed the same semantic space.
The algorithm is summarized as Algorithm 1 in Appendix.

4.3. Inference and Uncertainty Quantification

To quantify the epistemic uncertainty for an embeddingzof
modalityc, we estimate its exact log-density by integrating
the divergence of the learned conditional vector field.

```
4.3.1. LOG-LIKELIHOOD VIA CONTINUITY EQUATION
```
Using the instantaneous change of variables formula on the
manifold, the log-density at the target distribution is:

logp 1 (z | c) = logp 0 (ψ 0 (z))−

### Z 1

```
0
```
```
divSd− 1 (vt(zt,c;φ))dt,
```
whereψtdenotes the flow map induced by integrating
the ODE backward fromt = 1tot = 0. Since the
base distribution is uniform, its log-density is constant:
logp 0 (z) = − log Vol(Sd−^1 ), which simplifies the like-
lihood computation to integrating only the divergence term.

```
Manifold-Aware ODE Integration We solve the reverse
ODE using a Riemannian-adapted Euler’s method. At each
integration step, intermediate states are projected back onto
Sd−^1 via normalization, and velocity evaluations are pro-
jected onto the tangent spaceTzSd−^1. This ensures numeri-
cal integrity and prevents drifting away from the manifold
over many integration steps.
```
```
Riemannian Divergence Estimation Computing the ex-
act divergencedivSd− 1 (vt)is computationally expensive
for high-dimensional VLM embeddings as it requires
O(d)backward passes. Instead, we employ a manifold-
constrained Hutchinson trace estimator (Hutchinson, 1989):
```
```
divSd− 1 (vt)≈⟨ ̃ε,∇z(vt· ̃)ε⟩,
where ̃ = Πε Tz(ε), ε∼N (0,I).
```
```
ΠTz(v) = v−⟨v,z⟩zprojects the random probe vectorε
onto the tangent space atz. This allows for an unbiased
estimate of the divergence with significant reduction in com-
putational cost.
```
```
Epistemic Uncertainty Score The resulting log-density
yields the epistemic uncertainty score:
```
```
Uep(x) =− logp 1 (z | c).
```
```
High uncertainty (low density) identifies samples that reside
in sparse regions of the manifold, such as outliers, noisy
inputs, and nonsensical content.
A more comprehensive derivation including the construc-
tion of the probability path, the training objective, and the
calculation of the likelihood, is deferred to Appendix A.
```
## 5. Empirical results

```
We evaluate REPVLM’s ability to quantify epistemic uncer-
tainty through selective classification on benchmark datasets
first. Then, we provide an ablation study for REPVLM. Fur-
ther, we show two applications of our uncertainty estimates
including out-of-distribution detection and data curation.
```
```
5.1. Datasets, baselines, and metrics
```
```
We train REPVLM on three proxy datasets: Conceptual
Captions (Sharma et al., 2018), DataComp-1B (Gadre et al.,
2023), and LAION-2B (Schuhmann et al., 2022). To en-
sure computational efficiency, we randomly sample 1M
pairs from each dataset. For evaluation, we assess zero-
shot classification across six benchmarks: ImageNet-1K
(Deng et al., 2009), Food101 (Bossard et al., 2014), Ci-
far100 (Krizhevsky et al., 2009), ObjectNet (Barbu et al.,
2019), ImageNet-R (Kornblith et al., 2019), and ImageNet-
Sketch (Wang et al., 2019). All experiments reported in the
main manuscript use OpenCLIP with a ViT-B/32 backbone
```

```
66
```
```
68
```
```
70
```
```
72
```
```
74
```
```
Accuracy (%)
Conceptual Captions
```
```
ImageNet-1K
```
```
80
```
```
82
```
```
84
```
```
86
```
```
88
```
```
90
```
```
Food
```
```
75
```
```
80
```
```
85
```
```
90
```
```
CIFAR
```
```
50
```
```
60
```
```
70
```
```
ObjectNet
```
```
72.
```
```
75.
```
```
77.
```
```
80.
```
```
82.
```
```
85.
```
```
87.
```
```
ImageNet-R
```
```
50
```
```
55
```
```
60
```
```
65
```
```
ImageNet-Sketch
```
```
66
```
```
68
```
```
70
```
```
72
```
```
74
```
```
DataCompAccuracy (%)
80
```
```
82
```
```
84
```
```
86
```
```
88
```
```
90
```
```
75
```
```
80
```
```
85
```
```
90
```
```
50
```
```
60
```
```
70
```
```
72.
```
```
75.
```
```
77.
```
```
80.
```
```
82.
```
```
85.
```
```
87.
```
```
50
```
```
55
```
```
60
```
```
65
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
66
```
```
68
```
```
70
```
```
72
```
```
74
```
```
LAIONAccuracy (%)
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
80
```
```
82
```
```
84
```
```
86
```
```
88
```
```
90
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
75
```
```
80
```
```
85
```
```
90
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
50
```
```
60
```
```
70
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
72.
```
```
75.
```
```
77.
```
```
80.
```
```
82.
```
```
85.
```
```
87.
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
50
```
```
55
```
```
60
```
```
65
```
```
REPVLM ProbVLM MC-Dropout
```
Figure 2. Accuracy-Rejection Curves for Zero-Shot Classification. We evaluate the utility of epistemic uncertainty estimates for
selective classification across three proxy datasets (rows) and six downstream benchmarks (columns). Thex-axis represents the fraction
of samples rejected based on high uncertainty, and they-axis shows the Top-1 Accuracy of the remaining retained data. A sharp upward
slope indicates that the model identifies its own errors via the uncertainty measure.

(Cherti et al., 2023). Results using SigLIP ViT-B/16 (Zhai
et al., 2023) are provided in Appendix 10.

We compare REPVLM against two existing methods for
epistemic uncertainty quantification for frozen VLMs: Prob-
VLM (Upadhyay et al., 2023) and Monte Carlo Dropout
(MCDO). Other methods are excluded as they focus on
aleatoric uncertainty (data ambiguity) and lack a principled
measure for model confidence (see Table 1).

We utilize selective classification (Geifman & El-Yaniv,
2017) to measure the effectiveness of the uncertainty scores,
reporting the following metrics:

- Accuracy-Rejection Curves: These plot Top-1 accuracy
    as the most uncertain samples (0–90%) are progressively
    rejected; an upward slope indicates the model successfully
    identifies its own errors via uncertainty measures.
- Acc@90% Rejection: Top-1 accuracy on the 10% most
    confident predictions.
- Spearman’sS: The correlation between the rejection
    fraction and accuracy. AnS = 1. 0 indicates a perfect
    monotonic relationship where increasing the rejection
    threshold consistently improves accuracy

All experiments were repeated five times with different ran-
dom seeds, and we report the mean with standard deviations.

5.2. Main results

The numerical results of the epistemic uncertainty eval-
uation are reported in Table 2, with the corresponding
accuracy-rejection curves shown in Figure 2.

```
Reliability on Standard Benchmarks For the standard
benchmarks including ImageNet-1K, Food101, and Ci-
far100, REPVLM consistently outperforms existing base-
lines. As shown in Figure 2, the accuracy-rejection curves
for REPVLM exhibit a sharper upward slope compared to
ProbVLM and MCDO, indicating more effective identifi-
cation of error-prone samples. Quantitatively, our method
achieves a Spearman’s rank correlation of nearly 1. 0 across
almost all general tasks. This demonstrate a near-perfect
relationship between estimated uncertainty and prediction
error. We note that on Cifar100, MCDO achieves compa-
rable or slightly better performance. This is likely due to
the low-resolution nature of the images in Cifar100 dataset
( 32 × 32 ), which are underrepresented in the proxy datasets
and thus limit the performance of REPVLM.
```
```
Robustness Under Distribution Shift The advantages of
REPVLM are more pronounced on challenging distribution-
shift datasets including ObjectNet, ImageNet-R, and
ImageNet-Sketch, where baselines often fail.
```
- ObjectNet: On this dataset, both baselines struggle sig-
    nificantly. MCDO shows a weak correlation (S = 0. 316 )
    and ProbVLM exhibits inconsistent behavior, includ-
    ing negative Spearman values (e.g.,S = − 0. 799 on
    LAION). In contrast, REPVLM maintains perfect correla-
    tion (S = 1. 000 ) across all proxy datasets while achieving
    substantially higher Acc@90% Rejection.
- ImageNet-R and ImageNet-Sketch: While MCDO
    shows strong correlations (S = 1. 000 ) on these datasets,
    REPVLM consistently yields higher Top-1 accuracy at
    the 90% rejection threshold. ProbVLM exhibits high


Table 2. Evaluation of model performance and uncertainty on benchmarks across different training configurations (Conceptual Caption,
DataComp, and LAION). For each method, we report Accuracy at 90% Rejection↑and Spearman’s rank correlationS ↑between
uncertainty and prediction error. Bold font denotes the best results.

```
CONCEPTUAL CAPTION DATACOMP LAION
EVAL. ON METHOD Acc. @ 90% Rej.↑ S ↑ Acc. @ 90% Rej.↑ S ↑ Acc. @ 90% Rej.↑ S ↑
MCDO 0.699± 0.009 0.990± 0.021 0.699± 0.009 0.990± 0.021 0.699± 0.009 0.990± 0.
ImageNet-1K ProbVLM 0.664± 0.002 0.423± 0.330 0.670± 0.012 0.435± 0.559 0.667± 0.004 0.202± 0.
REPVLM 0.730± 0.002 0.993± 0.006 0.723± 0.005 1.000± 0.000 0.734± 0.005 0.998± 0.
MCDO 0.853± 0.011 0.913± 0.092 0.853± 0.011 0.913± 0.092 0.853± 0.011 0.913± 0.
Food101 ProbVLM 0.871± 0.015 0.942± 0.110 0.861± 0.034 0.370± 0.773 0.833± 0.028 0.418± 0.
REPVLM 0.904± 0.001 1.000± 0.000 0.895± 0.003 1.000± 0.000 0.881± 0.007 0.998± 0.
MCDO 0.888± 0.014 1.000± 0.000 0.888± 0.014 1.000± 0.000 0.888± 0.014 1.000± 0.
Cifar100 ProbVLM 0.783± 0.009 0.927± 0.051 0.825± 0.033 0.778± 0.274 0.760± 0.020 -0.147± 0.
REPVLM 0.874± 0.010 0.998± 0.005 0.872± 0.007 1.000± 0.000 0.878± 0.014 1.000± 0.
MCDO 0.488± 0.038 0.316± 0.618 0.488± 0.038 0.316± 0.618 0.488± 0.038 0.316± 0.
ObjectNet ProbVLM 0.571± 0.030 0.968± 0.036 0.522± 0.075 0.559± 0.365 0.462± 0.025 -0.799± 0.
REPVLM 0.755± 0.010 1.000± 0.000 0.693± 0.013 1.000± 0.000 0.735± 0.006 1.000± 0.
MCDO 0.845± 0.011 1.000± 0.000 0.845± 0.011 1.000± 0.000 0.845± 0.011 1.000± 0.
ImageNet-R ProbVLM 0.793± 0.010 0.978± 0.014 0.781± 0.032 0.656± 0.433 0.751± 0.016 0.106± 0.
REPVLM 0.885± 0.003 1.000± 0.000 0.872± 0.009 1.000± 0.000 0.877± 0.008 1.000± 0.
MCDO 0.590± 0.013 1.000± 0.000 0.590± 0.013 1.000± 0.000 0.590± 0.013 1.000± 0.
ImageNet-Sketch ProbVLM 0.607± 0.009 1.000± 0.000 0.585± 0.018 0.998± 0.005 0.501± 0.019 -0.750± 0.
REPVLM 0.659± 0.001 1.000± 0.000 0.641± 0.004 1.000± 0.000 0.614± 0.004 1.000± 0.
```
```
performance variance, showing a negative correlation
(S = − 0. 750 ) on ImageNet-Sketch when using the
LAION proxy dataset.
```
Overall, the results demonstrate that REPVLM provides a
superior epistemic uncertainty measure that is highly corre-
lated with actual model performance, making it a reliable
tool for high-stakes selective classification.

5.3. Ablation study

We conduct ablation studies on ImageNet-1K with proxy
dataset Conceptual Captions to isolate the impact of our
design choices on uncertainty estimation quality.

Impact of Riemannian Geometry The defining charac-
teristic of REPVLM is its adherence to the manifold ge-
ometry of the embedding space. We compare our Rieman-
nian approach against two Euclidean variants: one using a
uniform base distribution on the hypersphere and another
using a standard Gaussian base. As shown in Figure 3
(left), the Riemannian formulation consistently outperforms
both Euclidean variants across all rejection thresholds. This
performance gain stems from respecting the intrinsic man-
ifold structure of the embedding space. Since VLM em-
beddings areℓ 2 -normalized and reside on the hypersphere
Sd−^1 , geodesic interpolation provides a more natural and
accurate path than straight-line Euclidean paths that ‘cut
through’ the sphere’s interior.

Scaling with Proxy Data We analyze the sensitivity of
REPVLM to the volume of training data by varying the
proxy dataset size from 50K to 1M image-caption pairs. As

```
shown in Figure 3 (center), uncertainty quality improves as
the size of the proxy dataset increases, allowing REPVLM
to capture the empirical embedding distribution more accu-
rately. However, the gains in accuracy at the 90% rejection
threshold begin to plateau beyond 500K samples. This sug-
gests that moderate-sized proxy datasets are sufficient for
generating reliable epistemic uncertainty estimates, making
REPVLM practical even with limited resources.
```
```
ODE Integration Dynamics To evaluate the impact of
numerical integration on uncertainty quality, we varied the
number of ODE integration steps. As shown in Figure 3
(right), the results indicate that a single step is insufficient
and two steps yield unstable results with high variance.
Performance robustly converges at only five steps. This
rapid stabilization is enabled by our manifold-aware inte-
gration strategy, which uses tangent space projections and
renormalization to prevent numerical drift away fromSd−^1.
Further, because the pre-trained CLIP embeddings are well-
regularized within a highly structured joint semantic space,
the learned vector field is sufficiently smooth to allow for
rapid convergence. This ensures minimal additional over-
head, maintaining the inherent efficiency of REPVLM.
```
```
5.4. Computational Cost
Uncertainty estimation overhead is primarily dictated by the
backbone model. While MCDO is the most computationally
demanding approach (92.50 GFLOPs) due toN = 10full
forward passes, ProbVLM and REPVLM achieve significant
gains by decoupling estimation from the heavy backbone.
Although ProbVLM is efficient (10.35 GFLOPs), it yields
unreliable epistemic uncertainty estimates, often showing
```

```
0 20 40 60 80
Rejected Fraction (%)
```
```
66
```
```
68
```
```
70
```
```
72
```
```
74
```
```
I2T Accuracy (%)
```
```
Flow Matching Geometry
Riemannian
Euclidean (Unif.)
Euclidean (Guass.)
```
```
0 20 40 60 80
Rejected Fraction (%)
```
```
Proxy Dataset Size
50K
100K
200K
500K
1M
0 20 40 60 80
Rejected Fraction (%)
```
```
Number of Integration Steps
1
2
5
8
```
Figure 3. Ablation studies of REPVLM. Left: Comparison of
the proposed Riemannian formulation against Euclidean variants
using uniform and Gaussian base distributions. Center: Impact of
proxy training data size, varying from 50K to 1M image-caption
samples. Right: Convergence of accuracy-rejection performance
relative to the number of ODE integration steps.

poor or even negative correlation with model error. Leverag-
ing a network architecture designed to match the unit cost
of ProbVLM (GProbVLM= GREPVLM= 0. 11 ), REPVLM
achieves even greater efficiency at 9.80 GFLOPs, requiring
only 0.55 GFLOPs of additional overhead.

Table 3. Computational complexity comparison in GFLOPs.N
denotes the number of stochastic samples for MCDO and Prob-
VLM, while T denotes the ODE integration steps in REPVLM.

```
METHOD COMPONENT UNIT COST H. PARAM. GFLOPS
Backbone Gbkb 9.25 - 9.
MCDO NGbkb 9.25 N = 10 92.
ProbVLM Gbkb+ NGProbVLM 0.11 N = 10 10.
REPVLM Gbkb+ TGREPVLM 0.11 T = 5 9.
```
5.5. Applications

We now demonstrate the practical utility of REPVLM’s
uncertainty score beyond zero-shot classification. We show-
case its effectiveness in identifying OOD data and perform-
ing automated data curation.

```
1600140012001000
Uncertainty Score
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
Density
```
```
Distributions
```
```
0.00 0.25 0.50 0.75 1.
FPR
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
1.
```
```
TPR
```
```
Mean ROC
```
```
0.00 0.25 0.50 0.75 1.
Recall
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
1.
```
```
Precision
```
```
Mean PR
```
```
ID (ImageNet1K) Near-OOD (ImageNet-R) Far-OOD (EuroSAT)
```
Figure 4. Out-of-Distribution Detection. Left: Distributions of
uncertainty scoresUep(x) =− log p(z|c)for ImageNet-1K (ID),
ImageNet-R (Near-OOD), and EuroSAT (Far-OOD). Center &
Right: Mean Receiver Operating Characteristic (ROC) and Mean
Precision-Recall (PR) curves.

Out-of-Distribution Detection Our uncertainty estimates
provide a principled and natural mechanism for OOD de-
tection by identifying samples that reside in low-density re-
gions of the embedding manifold. We designate ImageNet-
1K as the in-distribution (ID) dataset, with ImageNet-R
serving as near-OOD and EuroSAT as far-OOD. Following

```
2000 1800 1600 1400 1200 1000 800
Uncertainty Score
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
Density
```
```
Distribution of Uncertainties
Image
Text
```
```
Uncertain Samples
99.1x57mm White RefL7992-
"Gold® 30-inch GlassCooktop with
Radiant Element"12""/9"" Dual
Estrategia deMarketing
Sulaiman Abdul AzizAziz Quotes By
Al Rajhi
```
```
Figure 5. Data Curation via Epistemic Uncertainty. Left: The
density distribution of uncertainty scores, where the tail corre-
sponds to high-uncertainty samples. Right: Qualitative examples
of samples with the highest epistemic uncertainty.
```
```
our theoretical framework, we use the conditional negative
log-likelihood of the visual modality as the scoring func-
tion. As illustrated in Figure 4, standard ID inputs map
to high-density regions, whereas OOD inputs consistently
reside in the low-density regions. The structural separation
between ID and OOD data results in robust detection per-
formance, quantitatively supported by high Mean ROC and
Mean Precision-Recall (PR) curves.
```
```
Data Curation The estimated epistemic uncertainty
serves as an effective metric for cleaning large-scale datasets
by identifying outliers and poor-quality samples. As illus-
trated in Figure 5, samples flagged with the highest uncer-
tainty scores exhibit clear qualitative issues, such as heavy
blurs or nonsensical visual and textual content. Filtering
out these high-uncertainty samples will allow us to auto-
matically prune noise from large-scale web datasets before
training. The curation process potentially improves the sta-
bility and performance of future models without the need of
expensive manual verification.
```
## 6. Conclusion and limitation

```
We introduced REPVLM, a principled framework for quanti-
fying epistemic uncertainty in pre-trained Vision-Language
Models using Conditional Riemannian Flow Matching.
REPVLM estimates the probability density directly on the
embedding manifoldSd−^1 , establishing a rigorous link be-
tween embedding density and model confidence. Extensive
experiments demonstrate that this density-based approach
provides a scalable and reliable metric for model confidence,
significantly outperforming existing baselines in selective
classification and out-of-distribution detection.
```
```
Limitations REPVLM’s efficacy is contingent on the
availability of a representative proxy dataset; a significant
domain gap or low-resolution inputs can degrade density
estimation quality. Furthermore, reliance on embedding
density poses potential fairness risks, as data from underrep-
resented demographics may be flagged as high-uncertainty
due to inherent data sparsity rather than model error.
```

## Impact Statement

This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.

## References

Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gut-
freund, D., Tenenbaum, J., and Katz, B. Objectnet: A
large-scale bias-controlled dataset for pushing the lim-
its of object recognition models. Advances in neural
information processing systems, 32, 2019.

Baumann, A., Li, R., Klasson, M., Mentu, S., Karthik,
S., Akata, Z., Solin, A., and Trapp, M. Post-hoc
probabilistic vision-language models. arXiv preprint
arXiv:2412.06014, 2024.

Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra,
D. Weight uncertainty in neural network. In International
conference on machine learning, pp. 1613–1622. PMLR,
2015.

Bossard, L., Guillaumin, M., and Van Gool, L. Food-101–
mining discriminative components with random forests.
In European conference on computer vision, pp. 446–461.
Springer, 2014.

Chen, R. T. and Lipman, Y. Flow matching on general
geometries. In The Twelfth International Conference on
Learning Representations, 2024.

Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud,
D. K. Neural ordinary differential equations. Advances
in neural information processing systems, 31, 2018.

Cherti, M., Beaumont, R., Wightman, R., Wortsman,
M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt,
L., and Jitsev, J. Reproducible scaling laws for con-
trastive language-image learning. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 2818–2829, 2023.

Chun, S. Improved probabilistic image-text representations.
In The Twelfth International Conference on Learning
Representations, 2024.

Chun, S., Oh, S. J., De Rezende, R. S., Kalantidis, Y., and
Larlus, D. Probabilistic embeddings for cross-modal
retrieval. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 8415–
8424, 2021.

Chun, S., Kim, W., Park, S., and Yun, S. Probabilistic
language-image pre-training. In The Thirteenth Interna-
tional Conference on Learning Representations, 2025.

```
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,
L. Imagenet: A large-scale hierarchical image database.
In 2009 IEEE conference on computer vision and pattern
recognition, pp. 248–255. Ieee, 2009.
```
```
Gadre, S. Y., Ilharco, G., Fang, A., Hayase, J., Smyrnis, G.,
Nguyen, T., Marten, R., Wortsman, M., Ghosh, D., Zhang,
J., et al. Datacomp: In search of the next generation of
multimodal datasets. Advances in Neural Information
Processing Systems, 36:27092–27112, 2023.
```
```
Gal, Y. and Ghahramani, Z. Dropout as a bayesian approx-
imation: Representing model uncertainty in deep learn-
ing. In international conference on machine learning, pp.
1050–1059. PMLR, 2016.
```
```
Geifman, Y. and El-Yaniv, R. Selective classification for
deep neural networks. Advances in neural information
processing systems, 30, 2017.
```
```
Gomez, L. Over the top-1: Uncertainty-aware cross-modal
retrieval with clip. In The 41st Conference on Uncertainty
in Artificial Intelligence, 2025.
```
```
Grathwohl, W., Chen, R. T., Bettencourt, J., Sutskever, I.,
and Duvenaud, D. Ffjord: Free-form continuous dy-
namics for scalable reversible generative models. arXiv
preprint arXiv:1810.01367, 2018.
```
```
Hullermeier, E. and Waegeman, W. Aleatoric and epistemic ̈
uncertainty in machine learning: An introduction to con-
cepts and methods. Machine learning, 110(3):457–506,
2021.
```
```
Hutchinson, M. F. A stochastic estimator of the trace of the
influence matrix for laplacian smoothing splines. Com-
munications in Statistics-Simulation and Computation,
18(3):1059–1076, 1989.
```
```
Ju, L., Andersson, M., Fredriksson, S., Glockner, E., Hellan- ̈
der, A., Vats, E., and Singh, P. Exploiting the asymmetric
uncertainty structure of pre-trained vlms on the unit hy-
persphere. Advances in neural information processing
systems, 39, 2025.
```
```
Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet
models transfer better? In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pp. 2661–2671, 2019.
```
```
Krizhevsky, A., Hinton, G., et al. Learning multiple layers
of features from tiny images.(2009), 2009.
```
```
Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple
and scalable predictive uncertainty estimation using deep
ensembles. Advances in neural information processing
systems, 30, 2017.
```

Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping
language-image pre-training for unified vision-language
understanding and generation. In International confer-
ence on machine learning, pp. 12888–12900. PMLR,
2022.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and
Le, M. Flow matching for generative modeling. In The
Eleventh International Conference on Learning Repre-
sentations.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J.,
et al. Learning transferable visual models from natural
language supervision. In International conference on
machine learning, pp. 8748–8763. PMLR, 2021.

Saxe, A., McClelland, J., and Ganguli, S. Exact solutions to
the nonlinear dynamics of learning in deep linear neural
networks. In Proceedings of the International Confer-
ence on Learning Represenatations 2014. International
Conference on Learning Represenatations 2014, 2014.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C.,
Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis,
C., Wortsman, M., et al. Laion-5b: An open large-scale
dataset for training next generation image-text models.
Advances in neural information processing systems, 35:
25278–25294, 2022.

Sharma, P., Ding, N., Goodman, S., and Soricut, R. Con-
ceptual captions: A cleaned, hypernymed, image alt-text
dataset for automatic image captioning. In Proceedings
of the 56th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pp. 2556–
2565, 2018.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I.,
and Salakhutdinov, R. Dropout: a simple way to prevent
neural networks from overfitting. The journal of machine
learning research, 15(1):1929–1958, 2014.

Upadhyay, U., Karthik, S., Mancini, M., and Akata, Z. Prob-
vlm: Probabilistic adapter for frozen vision-language
models. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp. 1899–1910, 2023.

Venkataramanan, A., Bodesheim, P., and Denzler, J. Prob-
abilistic embeddings for frozen vision-language models:
Uncertainty quantification with gaussian process latent
variable models. arXiv preprint arXiv:2505.05163, 2025.

Wang, H., Ge, S., Lipton, Z., and Xing, E. P. Learning
robust global representations by penalizing local predic-
tive power. Advances in neural information processing
systems, 32, 2019.

```
Wang, T. and Isola, P. Understanding contrastive represen-
tation learning through alignment and uniformity on the
hypersphere. In International conference on machine
learning, pp. 9929–9939. PMLR, 2020.
```
```
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C.,
Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M.,
et al. Transformers: State-of-the-art natural language
processing. In Proceedings of the 2020 conference on em-
pirical methods in natural language processing: system
demonstrations, pp. 38–45, 2020.
```
```
Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith,
S., Roelofs, R., Lopes, R. G., Hajishirzi, H., Farhadi, A.,
Namkoong, H., et al. Robust fine-tuning of zero-shot
models. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 7959–7971,
2022.
```
```
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sig-
moid loss for language image pre-training. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pp. 11975–11986, 2023.
```
```
Zhang, J., Huang, J., Jin, S., and Lu, S. Vision-language
models for vision tasks: A survey. IEEE transactions on
pattern analysis and machine intelligence, 46(8):5625–
5644, 2024.
```
```
Zhou, Z., Lei, Y., Zhang, B., Liu, L., and Liu, Y. Zegclip: To-
wards adapting clip for zero-shot semantic segmentation.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 11175–11185, 2023.
```

## A. Derivations for Riemannian Flow Matching

In this section, we provide the complete mathematical derivations for our Conditional Riemannian Flow Matching framework
on the hypersphere Sd−^1.

A.1. Preliminaries: Geometry of the Hypersphere

The (d− 1)-dimensional unit hypersphere embedded in Rdis defined as:

```
Sd−^1 ={z ∈ Rd:∥z∥ 2 = 1}. (1)
```
Tangent Space. The tangent space at a point z ∈ Sd−^1 is the set of all vectors orthogonal to z:

```
TzSd−^1 ={v ∈ Rd:⟨v,z⟩ = 0}. (2)
```
This is a (d− 1)-dimensional linear subspace of Rd.

Riemannian Metric. The hypersphere inherits the standard Euclidean inner product as its Riemannian metric. For tangent
vectors u,v ∈ TzSd−^1 :
⟨u,v⟩z= u⊤v. (3)

Tangent Space Projection. Given an arbitrary vector v ∈ Rd, its projection onto the tangent space TzSd−^1 is:

```
ΠTz(v) = v−⟨v,z⟩z = (I − zz⊤)v. (4)
```
This projection removes the component of v normal to the sphere at z.

Exponential Map. The exponential mapexpz: TzSd−^1 → Sd−^1 maps a tangent vector to a point on the sphere by
following the geodesic:

```
expz(v) = cos(∥v∥)z + sin(∥v∥)
```
```
v
∥v∥
```
### . (5)

For v = 0, we have expz(0) = z.

Geodesic Distance. The geodesic (great-circle) distance between two points z 0 ,z 1 ∈ Sd−^1 is:

```
dS(z 0 ,z 1 ) = arccos(⟨z 0 ,z 1 ⟩). (6)
```
A.2. Construction of Probability Paths

Flow matching constructs a time-dependent probability densitypt(z)that interpolates between a simple base distributionp 0
and the target data distribution p 1.

Base Distribution. We use the uniform distribution on the hypersphere as our base:

```
p 0 (z) = Unif(Sd−^1 ) =
```
### 1

```
Vol(Sd−^1 )
```
### , (7)

where the volume of the (d− 1)-sphere is:

```
Vol(Sd−^1 ) =
2 πd/^2
Γ(d/2)
```
### . (8)

Sampling from this distribution can be achieved by drawing ξ ∼N (0,Id) and normalizing: z 0 = ξ/∥ξ∥.

Target Distribution. The target distribution p 1 (z|c) is the empirical distribution of VLM embeddings for modality c:

```
p 1 (z|c) =
```
### 1

```
Nc
```
```
XNc
```
```
i=
```
```
δ(z− zi(c)), (9)
```
where{z(ic)}Ni=1c are the embeddings from the proxy dataset for modality c.


Conditional Probability Path. Following the Riemannian Flow Matching framework (Chen & Lipman, 2024), we
construct a conditional probability pathpt(z|z 1 )that concentrates mass along the geodesic from a random base pointz 0 to a
fixed target point z 1. Marginalizing over the base distribution yields:

```
pt(z) =
```
### Z

```
Sd−^1
```
```
pt(z|z 1 )p 1 (z 1 )dμ(z 1 ), (10)
```
where dμ is the uniform measure on Sd−^1.

Spherical Linear Interpolation (Slerp). Given two pointsz 0 ,z 1 ∈ Sd−^1 with geodesic distanceθ = arccos(⟨z 0 ,z 1 ⟩),
the geodesic path connecting them is parameterized as:

```
zt= slerp(z 0 ,z 1 ;t) =
```
```
sin((1− t)θ)
sinθ
```
```
z 0 +
```
```
sin(tθ)
sinθ
```
```
z 1 , (11)
```
for t∈ [0, 1]. This satisfies the boundary conditions z 0 = slerp(z 0 ,z 1 ; 0) and z 1 = slerp(z 0 ,z 1 ; 1).

A.3. Target Vector Field

The target (conditional) vector field ut(zt|z 1 ) is defined as the velocity of the geodesic path at time t:

```
ut(zt|z 1 ) =
```
```
dzt
dt
```
### . (12)

Derivation. Taking the derivative of the slerp formula:

```
dzt
dt
```
### =

```
d
dt
```
### 

```
sin((1− t)θ)
sinθ
```
```
z 0 +
```
```
sin(tθ)
sinθ
```
```
z 1
```
### 

### (13)

### =

```
−θ cos((1− t)θ)
sinθ
```
```
z 0 +
```
```
θ cos(tθ)
sinθ
```
```
z 1. (14)
```
Simplifying:

```
ut(zt|z 1 ) =
```
```
θ
sinθ
```
```
[cos(tθ)z 1 − cos((1− t)θ)z 0 ]. (15)
```
A.4. Loss Function

Conditional Flow Matching Objective. The flow matching objective trains a neural network vt(z,c;φ) to approximate
the marginal vector field by regressing onto the conditional target vector field:

```
LCFM(φ) = Et,z 0 ,z 1 ,c
```
### 

```
∥vt(zt,c;φ)− ut(zt|z 1 )∥^2
```
### 

### , (16)

where:

- t∼ Unif([0, 1]),
- c∼ Bernoulli(0.5) (modality indicator),
- z 1 ∼ p 1 (z|c) (target sample from proxy dataset),
- z 0 ∼ p 0 = Unif(Sd−^1 ) (base sample),
- zt= slerp(z 0 ,z 1 ;t).

Tangent Space Projection. To ensure the learned vector field respects the manifold constraint, we apply tangent space
projection to the network output:

```
vt(z,c;φ) = ΠTz( ̃vt(z,c;φ)) = ̃tv −⟨ ̃vt,z⟩z, (17)
```
where ̃vtis the raw network output.


Algorithm 1 Training Conditional Riemannian Flow Matching

Require: Proxy datasetD ={(zimgi ,ztxti)}Ni=1, network vφ
1: while not converged do
2: Sample batch of pairs (zimg,ztxt)∼D{Draw from proxy modality pairs}
3: Sample c∼ Bernoulli(0.5) for each pair{Select conditioning modality}
4: Set z 1 = zimgif c = 0, else z 1 = ztxt{Define target sample}
5: Sample z 0 ∼ Unif(Sd−^1 ){Sample from simple base distribution}
6: Sample t∼ Unif([0, 1]){Draw random time step}
7: Compute θ = arccos(⟨z 0 ,z 1 ⟩){Calculate geodesic distance}
8: Compute zt=sin((1sin−θt)θ)z 0 +sin(sintθθ)z 1 {Spherical linear interpolation}
9: Compute ut=sinθθ[cos(tθ)z 1 − cos((1− t)θ)z 0 ]{Target geodesic velocity}
10: Compute ̃tv = vφ(zt,t,c){Evaluate raw network output}
11: Project: vt= ̃vt−⟨ ̃tv,zt⟩zt{Project to tangent space TztSd−^1 }
12: Compute loss: L =∥vt− ut∥^2 {Riemannian Flow Matching objective}
13: Update φ via gradient descent{Optimize vector field parameters}
14: end while

Training Algorithm. The complete training procedure is summarized in Algorithm 1.

A.5. Likelihood Computation

Change of Variables on Manifolds. For a flowψt: Sd−^1 → Sd−^1 generated by the vector fieldvt, the density evolves
according to the continuity equation:
∂pt
∂t
+ divSd− 1 (ptvt) = 0. (18)

This yields the instantaneous change of variables formula:

```
d logpt(zt)
dt
=− divSd− 1 (vt(zt)). (19)
```
Log-Likelihood Computation. Integrating from t = 0 to t = 1:

```
logp 1 (z 1 |c) = logp 0 (z 0 )−
```
### Z 1

```
0
```
```
divSd− 1 (vt(zt,c;φ))dt, (20)
```
where z 0 = ψ 0 (z 1 ) is obtained by integrating the ODE backward from z 1.

Since the base distribution is uniform:

```
logp 0 (z 0 ) =− log Vol(Sd−^1 ) =− log
```
```
2 πd/^2
Γ(d/2)
```
### . (21)

Riemannian Divergence. The divergence of a vector field on Sd−^1 can be computed as:

```
divSd− 1 (v) = divRd(v)− (d− 1)⟨v,z⟩. (22)
```
Since vt∈ TzSd−^1 implies⟨vt,z⟩ = 0, this simplifies to:

```
divSd− 1 (vt) = divRd(vt) = Tr
```
### 

```
∂vt
∂z
```
### 

### . (23)

Hutchinson Trace Estimator. Computing the exact trace requiresO(d)backward passes through the network. Instead,
we use the Hutchinson estimator:
Tr(A) = Eε∼p(ε)[ε⊤Aε], (24)


where p(ε) is any distribution with E[ε] = 0 and Cov(ε) = I.

For the manifold setting, we project the probe vector onto the tangent space:

```
divSd− 1 (vt)≈⟨ ̃ε,∇z(vt· ̃)ε⟩, (25)
```
where ̃ = Πε Tz(ε) = ε−⟨ε,z⟩z and ε∼N (0,Id).

Numerical Integration. We solve the reverse ODE using a Riemannian-adapted Euler’s method. At each step, we:

1. Evaluate the vector field at intermediate points.
2. Project velocity evaluations onto the tangent space.
3. Update the position and renormalize to the sphere.

The complete inference algorithm is given in Algorithm 2.

Algorithm 2 Uncertainty Inference via Likelihood Computation

Require: Query embedding z 1 ∈ Sd−^1 , modality c, trained network vφ, steps K
1: Initialize z ← z 1 , logp← 0 , ∆t← 1 /K
2: for k = K,K− 1 ,..., 1 do
3: t← k/K
4: Sample ε∼N (0,Id)
5: ̃←ε ε−⟨ε,z⟩z{Project probe to tangent space TzSd−^1 }
6: v ← vφ(z,t,c){Evaluate learned vector field}
7: div←⟨ ̃ε,∇z(v· ̃)ε⟩{Hutchinson trace estimator}
8: logp← logp + div·∆t{Integrate divergence term}
9: z ← z− v· ∆t{Reverse Euler step}
10: z ← z/∥z∥{Project back to sphere (renormalization)}
11: end for
12: logp 0 ←− log(2πd/^2 /Γ(d/2)){Uniform log-density on Sd−^1 }
13: return Uep=−(logp 0 − logp){Negative log-likelihood score}

## B. Experimental Details

B.1. Training Details

Proxy Datasets. We train our flow matching models on three large-scale image-caption datasets:

- Conceptual Captions (CC3M): 1M randomly sampled image-caption pairs from the 3.3M dataset (Sharma et al., 2018).
- DataComp-1B: 1M randomly sampled pairs from the curated DataComp-1B pool (Gadre et al., 2023).
- LAION-2B: 1M randomly sampled pairs from LAION-2B-en (Schuhmann et al., 2022).

Results reported in Table 2 and Figure 2 are based on models trained with 500K pairs. All images are preprocessed using
the standard preprocessing pipeline corresponding to the pre-trained VLM.

Hyperparameters. The complete hyperparameter configuration is provided in Table 4. The learning rate is chosen based
on a grid search over the set{ 1 × 10 −^6 , 5 × 10 −^6 , 1 × 10 −^5 , 5 × 10 −^5 , 1 × 10 −^4 }.

Computational Resources. All experiments were conducted on NVIDIA A100 GPUs (40GB).


```
Table 4. Training hyperparameters for the flow matching model.
```
```
Hyperparameter REPVLM ProbVLM
Optimizer AdamW AdamW
Learning rate 1 × 10 −^51 × 10 −^4
Weight decay 1 × 10 −^51 × 10 −^5
Batch size 2048 2048
Training steps 400,000 400,
Learning rate schedule Linear Warmup + Constant Cosine Annealing
Warmup steps 1,000 1,
```
VLM Backbones. We use pre-trained and frozen VLM encoders from the Huggingfacetransformerslibrary (Wolf
et al., 2020):

- CLIP ViT-B/32: 512-dimensional embeddings, trained on LAION-2B (Cherti et al., 2023).
- SigLIP ViT-B/16: 768-dimensional embeddings, trained with sigmoid loss (Zhai et al., 2023).

B.2. Model Architecture

The vector field networkvφ(z,t,c)is implemented as a deep residual network consisting of an input projection, a series of
residual blocks with adaptive normalization, and a final output projection.

Conditioning and Embedding. The network processes time t and modality c into a shared conditioning vector:

- Timet: Encoded via sinusoidal positional encodingγ(t)with 256 frequencies, followed by a 2-layer MLP to produce a
    dhidden-dimensional embedding.
- Modality c: Embedded via a learnable lookup table into a dhidden-dimensional vector.
- Combined Context: The conditioning vectorcis formed by the element-wise sum of the time and modality embeddings.

Residual Blocks. Each of the 8 residual blocks employs an Adaptive Layer Normalization (AdaLN) mechanism. Given the
input to the i-th block hi, the transformation is defined as:

```
hi+1= hi+ Linear (SiLU (Norm(hi)· (1 + scale(c)) + shift, (c))) (26)
```
wherescale(·)andshift(·)are linear projections of the conditioning vectorc. This allows the time and modality information
to dynamically modulate the hidden representations throughout the network depth.

```
Table 5. Vector field network architecture specifications.
```
```
Component Specification Dimension / Value
Input Projection Linear Layer d→ dhidden
Conditioning Dim (dhidden) Time + Modality dhidden
Residual Blocks Depth 6
Activation Non-linearity SiLU
Normalization Type AdaLN
Output Projection Linear (Zero-init) dhidden→ d
```
B.3. Full Ablation Study Results

We provide complete numerical results for all ablation studies conducted on ImageNet-1K with the Conceptual Captions
proxy dataset.


```
Table 6. Comparison of Riemannian and Euclidean flow matching variants.
```
```
Method Acc. @ 90% Rej. Spearman’s S
Euclidean + Gaussian base 0.697± 0.002 0.893± 0.
Euclidean + Uniform base 0.716± 0.003 0.973± 0.
Riemannian (Ours) 0.733± 0.004 0.993± 0.
```
### B.3.1. RIEMANNIAN VS. EUCLIDEAN FORMULATION

The Euclidean variants operate inRdand project points onto the sphere only during evaluation. The “Euclidean + Gaussian
base” usesN (0,I)as the base distribution, while “Euclidean + Uniform base” samples from the uniform distribution on
Sd−^1 but uses straight-line interpolation. The Riemannian formulation outperforms both variants by 3-5% in accuracy at
90% rejection, demonstrating the importance of respecting the manifold geometry.

```
B.3.2. PROXY DATASET SIZE
```
```
Table 7. Effect of proxy dataset size on uncertainty estimation quality.
```
```
Dataset Size Acc. @ 90% Rej. Spearman’s S
50K 0.733± 0.002 0.973± 0.
100K 0.733± 0.002 0.990± 0.
200K 0.735± 0.002 0.993± 0.
500K 0.734± 0.004 0.988± 0.
1M 0.738± 0.003 0.998± 0.
```
Performance improves significantly when the size of proxy dataset is small, after which gains plateau. This suggests that
moderate-sized proxy datasets are sufficient for reliable uncertainty estimation, making our method practical with limited
computational resources.

```
B.3.3. NUMBER OF ODE INTEGRATION STEPS
```
```
Table 8. Effect of ODE integration steps on uncertainty quality and inference time.
```
```
Steps Acc. @ 90% Rej. Spearman’s S GFLOPS
1 0.694± 0.009 0.990± 0.009 0.
2 0.729± 0.013 0.993± 0.006 0.
5 0.737± 0.002 1.000± 0.000 0.
8 0.735± 0.002 0.995± 0.006 0.
```
Performance stabilizes at 5 integration steps, with negligible improvement beyond this point. The rapid convergence
is facilitated by: (1) the smoothness of learned vector fields in the well-structured CLIP embedding space, and (2) our
manifold-aware integration scheme that prevents drift from the hypersphere.

B.4. Evaluation Datasets

B.5. Extended Results with SigLIP

To demonstrate the generalizability of our method across different VLM architectures, we evaluate on SigLIP ViT-B/16,
which uses a sigmoid-based contrastive loss instead of the softmax-based loss in CLIP.


```
Table 9. Summary of evaluation datasets.
```
```
Dataset Classes Test Size Description
ImageNet-1K 1000 50,000 Standard object classification
Food101 101 25,250 Fine-grained food recognition
CIFAR-100 100 10,000 Low-resolution ( 32 × 32 ) images
ObjectNet 313 50,000 Objects in unusual poses/contexts
ImageNet-R 200 30,000 Artistic renditions of ImageNet classes
ImageNet-Sketch 1000 50,889 Sketch drawings of ImageNet classes
EuroSAT 10 5,400 Satellite imagery (far-OOD)
```
Table 10. Evaluation results using SigLIP ViT-B/16 backbone. We report Accuracy at 90% Rejection and Spearman’s rank correlationS
between uncertainty and prediction error. The best results are highlighted in bold.

```
CONCEPTUAL CAPTION DATACOMP LAION
EVAL. ON METHOD Acc. @ 90% Rej.↑ S ↑ Acc. @ 90% Rej.↑ S ↑ Acc. @ 90% Rej.↑ S ↑
MCDO 0.759± 0.005 -0.362± 0.234 0.759± 0.005 -0.362± 0.234 0.759± 0.005 -0.362± 0.
ImageNet-1K ProbVLM 0.712± 0.005 -1.000± 0.000 0.779± 0.010 0.932± 0.101 0.737± 0.006 -0.947± 0.
REPVLM 0.824± 0.004 0.910± 0.096 0.847± 0.002 0.905± 0.037 0.820± 0.005 0.985± 0.
MCDO 0.919± 0.007 0.648± 0.318 0.919± 0.007 0.648± 0.318 0.919± 0.007 0.648± 0.
Food101 ProbVLM 0.894± 0.010 -0.959± 0.059 0.899± 0.016 -0.376± 0.778 0.888± 0.010 -0.985± 0.
REPVLM 0.930± 0.002 0.755± 0.178 0.928± 0.002 0.907± 0.101 0.917± 0.003 0.855± 0.
MCDO 0.739± 0.023 0.722± 0.164 0.739± 0.023 0.722± 0.164 0.739± 0.023 0.722± 0.
Cifar100 ProbVLM 0.778± 0.014 0.995± 0.006 0.798± 0.011 0.069± 0.606 0.707± 0.035 -0.137± 0.
REPVLM 0.816± 0.006 0.896± 0.245 0.823± 0.010 0.802± 0.003 0.809± 0.018 0.721± 0.
MCDO 0.605± 0.039 0.738± 0.463 0.605± 0.039 0.738± 0.463 0.605± 0.039 0.738± 0.
ObjectNet ProbVLM 0.470± 0.025 -0.990± 0.014 0.501± 0.034 -0.792± 0.269 0.461± 0.027 -0.998± 0.
REPVLM 0.644± 0.012 0.976± 0.017 0.687± 0.019 0.998± 0.005 0.687± 0.017 0.947± 0.
MCDO 0.929± 0.004 0.995± 0.006 0.929± 0.004 0.995± 0.006 0.929± 0.004 0.995± 0.
ImageNet-R ProbVLM 0.903± 0.005 0.903± 0.120 0.919± 0.009 0.995± 0.006 0.901± 0.006 0.821± 0.
REPVLM 0.942± 0.003 1.000± 0.000 0.944± 0.003 1.000± 0.000 0.939± 0.006 1.000± 0.
MCDO 0.716± 0.005 1.000± 0.000 0.716± 0.005 1.000± 0.000 0.716± 0.005 1.000± 0.
ImageNet-Sketch ProbVLM 0.682± 0.007 0.864± 0.107 0.672± 0.024 0.185± 0.892 0.654± 0.012 -0.413± 0.
REPVLM 0.776± 0.003 0.933± 0.065 0.777± 0.007 0.948± 0.190 0.749± 0.004 0.909± 0.
```

