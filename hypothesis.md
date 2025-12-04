## TODO Ideas

- **Gaussian quantile normalization**: Map rewards to Gaussian via quantile transform (matches ALL moments, not just mean/std)
- **Teacher forcing + RL**: L = L_SL + λ·L_RL, inject ground truth to fix cold start (see notes.md for derivation)
- **Qwen 0.5B experiment**: Test if softmax helps small models learn reasoning (currently stuck at 0%)
- **MATH level 4-5**: Competition math (AMC/AIME) experiments
- **Temperature sweep**: τ = 0.25, 0.5, 0.75, 1.0 with Gaussian quantile

---

Here's a tight summary of what you wrote.

---

### Goal

* Policy (\pi_\theta(y|x)) should maximize expected verifiable reward:
  [
  J(\theta) = \mathbb{E}*{y\sim \pi*\theta}[R(y)]
  ]

---

### Standard Policy Gradient vs GRPO

1. **REINFORCE gradient**
   [
   \nabla_\theta J(\theta)
   = \mathbb{E}*{\pi*\theta}[R(y),\nabla_\theta \log \pi_\theta(y|x)]
   ]
   Monte Carlo with baseline (b) gives an unbiased estimator:
   [
   \hat g = \frac{1}{N}\sum_i (R_i - b),\nabla_\theta \log \pi_\theta(y_i|x)
   ]

2. **GRPO / batch whitening**
   Use batch mean and std as baseline + scale:
   [
   A_i = \frac{R_i - \mu_R}{\sigma_R}
   ]
   [
   \hat g_{\text{GRPO}} = \frac{1}{N}\sum_i A_i ,\nabla_\theta \log \pi_\theta(y_i|x)
   ]
   This preserves the expected gradient (up to scale) but assumes a roughly Gaussian reward shape and treats deviations symmetrically around the mean.

---

### Your Proposal: Softmax Reweighting

* Define reward weights:
  [
  w_i = \frac{\exp(R_i/\tau)}{\sum_j \exp(R_j/\tau)}
  ]
* Use a weighted log-likelihood loss:
  [
  \mathcal{L}(\theta) = -\sum_i w_i \log \pi_\theta(y_i|x)
  ]
  [
  \nabla_\theta \mathcal{L}
  = -\sum_i w_i ,\nabla_\theta \log \pi_\theta(y_i|x)
  ]

This is **not** the unbiased gradient of (\mathbb{E}[R]). It is minimizing:

* Cross-entropy (H(w,\pi_\theta))
* Equivalently, (\mathrm{KL}(w ,|, \pi_\theta))

i.e. projecting the policy onto the **reward-weighted target distribution** (w).

---

### Control-as-Inference / AWR View

* In max-entropy RL, the optimal policy has form:
  [
  \pi^*(y|x) \propto \pi_0(y|x),\exp(R(y)/\tau)
  ]
* Your (w_i \propto \exp(R_i/\tau)) is an empirical approximation of this optimal distribution.
* Minimizing (\mathrm{KL}(w|\pi_\theta)) is exactly **Advantage-Weighted Regression (AWR)**: fit (\pi_\theta) to the exponentiated-reward distribution.

So your method is:
“Approximate the optimal Boltzmann policy and then do MLE onto it.”

---

### Sparse Correct Solutions / Prior Shape

* For math/code, correct solutions are **exponentially rare**.
* GRPO’s whitened advantages impose a **Gaussian-like** symmetric prior around the mean.
* Softmax over rewards imposes an **exponential, peaked** prior that matches the idea “only a tiny fraction of samples are truly good.”

Softmax weights are also the solution of:
[
\max_w H(w) ;\text{s.t.}; \mathbb{E}_w[R]=\bar R,\ \sum w_i=1
]
→ maximum-entropy distribution under reward constraint.

---

### Information Geometry

* Policies (\pi_\theta) live on a statistical manifold with Fisher metric.
* Natural gradient is steepest ascent in KL geometry.
* Your update (\arg\min_\theta \mathrm{KL}(w|\pi_\theta)) is an **m-projection** onto the model manifold: choose the policy closest (in KL) to the softmax-reward target.

---

### Comparison Table (Core Points)

* **REINFORCE / GRPO**:

  * Optimizes (\mathbb{E}_\pi[R]).
  * Linear weights in (R) (after baseline/whitening).
  * Unbiased but higher variance.
  * Implicitly Gaussian-ish, symmetric around mean.

* **Softmax reweighting (AWR-style)**:

  * Optimizes ((1/\tau)\log \mathbb{E}_\pi[e^{R/\tau}]) / (\mathrm{KL}(w|\pi)).
  * Weights = Softmax(R/τ) ∈ [0,1].
  * Biased for expected reward, much lower variance.
  * Exponentially peaked prior that matches sparse-solution regimes.

---

### Temperature Regimes

[
w_i = \text{Softmax}(R_i / \tau)
]

* (\tau \to 0): almost one-hot on best sample (best-of-N / rejection sampling).
* (\tau \approx 1): Boltzmann selection over batch.
* (\tau \to \infty): uniform over batch (just supervised learning on all rollouts).

So τ controls how aggressively you focus on the (exponentially small) high-reward region.

---

## Research Methodology

### Principle: Metrics Before Experiments

**Never blindly try an idea.** Before running any experiment:
1. Identify what metric would prove/disprove the hypothesis
2. Collect that metric from baseline first
3. Only then test the modification
4. Compare the exact metric that motivated the change

### Principle: Simplest Case First

Start with the smallest possible test that isolates the core question:
- Smallest model that shows the effect
- Shortest training that shows divergence
- Single variable changes only

Each "improvement" adds complexity → reduces probability of success.
Only add complexity when simple version is validated.

### Principle: One Ablation at a Time

If testing softmax vs linear:
- Same random seed
- Same hyperparameters  
- Same everything except the ONE thing being tested

Never stack changes. If A+B works, you don't know if it was A, B, or interaction.

---

## Metrics That Would Validate/Invalidate This Hypothesis

### Core Prediction
**If softmax helps:** On sparse-reward tasks, softmax should:
1. Put more weight on correct samples (higher max weight)
2. Have lower effective sample size (more peaked)
3. Learn faster in early training (when correct solutions are rare)

**If softmax hurts:** 
1. Gradient magnitude too small (weights sum to 1, not N)
2. Ignores "almost correct" samples that linear would use
3. Collapses to best-of-N (τ too low → no gradient diversity)

### Specific Metrics to Log

**Per training step:**
- Raw rewards: [R_1, R_2, ..., R_N] for each group
- Linear weights: (R - μ) / σ  
- Softmax weights: exp(R/τ) / Σexp(R/τ)
- Weight statistics: max, entropy, effective_n = 1/Σw²
- Gradient norm (to detect magnitude issues)
- Loss value

**Per eval:**
- Success rate (answer_reward)
- Format compliance rate

### Decision Criteria

**Softmax is better if:**
- Same or higher success rate with same training steps
- Faster initial learning (success rate at step 10, 20, 30)
- More stable gradients (lower variance in grad_norm)

**Softmax is worse if:**
- Lower success rate
- Gradient collapse (grad_norm → 0)
- Weight collapse (max_weight → 1.0, effective_n → 1)

**Inconclusive if:**
- Same performance → maybe task isn't sparse enough
- High variance between runs → need more seeds

---

## Experiment Plan (In Order)

### Phase 1: Baseline Characterization
Run GRPO baseline, log:
- What does reward distribution look like per group?
- How many samples per group are "correct"?
- What are typical linear advantage values?

This tells us if the task is actually sparse.

### Phase 2: Weight Distribution Analysis (No Training)
Before training, just compute:
- Linear weights for sample batches
- Softmax weights for sample batches
- Compare: which concentrates more on correct samples?

This validates the hypothesis offline before any GPU time.

### Phase 3: Training Comparison
If Phase 2 shows softmax concentrates better:
- Run baseline (linear) for N steps
- Run softmax (τ=1.0) for N steps  
- Compare success rate curves

### Phase 4: Temperature Sweep (Only if Phase 3 positive)
If softmax helps, find optimal τ:
- τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- Plot success_rate vs τ

---

## Potential Failure Modes & Diagnostics

| Failure Mode | Symptom | Diagnostic |
|-------------|---------|------------|
| Gradient too small | grad_norm ≈ 0 | Log grad_norm, compare to baseline |
| Weight collapse | Only 1 sample gets weight | Log max_weight, effective_n |
| τ wrong scale | Uniform or one-hot weights | Log weight entropy |
| Task not sparse | All samples similar reward | Log reward variance per group |
| Numerical instability | NaN/Inf in loss | Log softmax denominator |
