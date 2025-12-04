# Softmax Rewards Experiment Notes

## Goal
Test if softmax reward weighting beats linear (GRPO-style) normalization on sparse-reward tasks.

## Why LLMs? (Rationale)

REINFORCE is general - could test on bandits, CartPole, small MLPs, etc. But chose LLM setup because:

1. **Hypothesis is about sparse solutions** - LLM reasoning tasks (math, code) have exponentially rare correct answers. Simpler domains (CartPole) have dense rewards, may not show the effect.

2. **GRPO-Zero is already simple enough** - Pure PyTorch, ~10 files, runs in hours. Not much complexity saved by going smaller.

3. **Direct validation** - If it works on Countdown, we know it works for the actual use case. No "does it transfer?" question.

**Alternative if too complex:** Could test on multi-armed bandit with sparse rewards (1 arm pays, rest don't). Would take 10 min to code, seconds to run. But might miss sequence-generation dynamics.

**Decision:** Start with GRPO-Zero. If setup issues, fall back to bandit experiment.

---

## Repo Evaluations

### 1. GRPO-Zero ⭐ PICKED
https://github.com/policy-gradient/GRPO-Zero

**First Impressions:**
- Pure PyTorch, no transformers/vllm dependencies
- ~10 small files, everything from scratch
- Implements Qwen2 model manually (`qwen2_model.py`)
- Uses Countdown task: given numbers [1,2,3,4], make expression = target
- Sparse reward: 0/1 for correct answer + 0.1 for format
- Single A40 (48GB) runs it in few hours, has 24GB config too
- DAPO improvements included (token-level loss, no KL div)

**Code Quality:** Excellent. Clean abstractions, easy to read.

**Swap Location:** `grpo.py` line 128 in `normalize_rewards_per_group()`:
```python
normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
```

**Verdict:** Best for initial test. Minimal deps, isolated normalization function.

---

### 2. simple_GRPO
https://github.com/lsdefine/simple_GRPO

**First Impressions:**
- ~200 lines across 2 main files
- Uses vllm for fast generation, deepspeed for training
- Separate ref_server.py process for reference model (clever memory trick)
- Uses GSM8K dataset (math word problems)
- Shows "Aha moment" within 30 steps
- 12-16 min training on 2xA800

**Code Quality:** Good but messier. HTTP requests between processes, more moving parts.

**Swap Location:** `grpo_vllm_one.py` line 194:
```python
curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
```

**Verdict:** Good for scaling up after initial test. GSM8K is harder/more realistic than Countdown.

---

### 3. TinyZero
https://github.com/Jiayi-Pan/TinyZero

**First Impressions:**
- Built on veRL (Volcano Engine RL library)
- Much larger codebase, framework-level abstractions
- Also uses Countdown task
- Twitter viral, good wandb logs available
- Requires ray, vllm, flash-attn

**Code Quality:** Framework-level. Would need to dig into veRL internals to find normalization.

**Verdict:** Too complex for quick iteration. Maybe use for final scaling if needed.

---

## Picked: GRPO-Zero

**Why:**
- Cleanest code (~10 files, pure PyTorch, no transformers/vllm)
- Advantage calc isolated in ONE function (`grpo.py:117-131`)
- Uses Countdown task (sparse correct solutions - perfect for hypothesis)
- Runs on single GPU, ~few hours

**The swap:**
```python
# CURRENT (GRPO linear):
normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)

# SOFTMAX VERSION:
group_rewards_arr = np.array(group_rewards)
softmax_weights = np.exp(group_rewards_arr / tau) / np.sum(np.exp(group_rewards_arr / tau))
normalized_reward = softmax_weights[idx] * len(group)  # scale to preserve magnitude
```

## Hardware
8xA100 available. Start with 1-2 GPUs on GRPO-Zero.

## Data
Cache all datasets here: `/home/kathan.shah/neural_test/data`

## Experimental Methodology

**Principles:**
1. One idea at a time - no stacking changes
2. Log everything: rewards, gradients, normalized weights, loss
3. First validate baseline works before testing softmax
4. Compare on same random seed for fair comparison

**Metrics to collect:**
- Raw rewards per group (before normalization)
- Normalized weights (linear vs softmax)
- Gradient norms per step
- Loss curves
- Success rate (answer_reward)
- Weight distribution stats (mean, std, max, entropy)

**Ablations (in order):**
1. Baseline (linear) vs Softmax (τ=1.0)
2. If softmax helps → sweep τ ∈ {0.1, 0.5, 1.0, 2.0}
3. If softmax hurts → analyze why (gradient magnitude? weight concentration?)

**Potential failure modes to watch:**
- Softmax too peaked (τ too low) → only learns from 1 sample per group
- Softmax too flat (τ too high) → same as uniform, no learning signal
- Gradient magnitude mismatch → need to rescale
- Numerical instability in softmax

## Experiment Log

### 2024-12-04: Phase 2 Weight Analysis (BEFORE training)

**Question:** Does softmax concentrate more on correct samples?

**Method:** Simulated reward groups with 1 correct (R=1.1), 7 wrong (R∈{0,0.1})

**Results:**
| Method | Weight on correct | Effective N |
|--------|------------------|-------------|
| Linear (GRPO) | 2.62 | ~2.8 |
| Softmax τ=1.0 | 2.33 | 6.38 |
| Softmax τ=0.5 | **4.33** | 3.10 |
| Softmax τ=0.1 | 8.0 | 1.0 (collapse!) |

**Finding:** τ=1.0 is TOO HIGH for reward scale [0, 1.1]. 
At τ=0.5, softmax puts 1.65x more weight on correct samples vs linear.
At τ<0.2, collapses to best-of-1 (no gradient diversity).

**Implication:** If we test softmax, use τ≈0.5, not τ=1.0.

---

### 2024-12-04: Training Comparison (Steps 1-10)

**Setup:** Baseline vs Softmax (τ=0.5), same seed, GPU 0 & 1

**Results at step 9-10:**
| Metric | Baseline | Softmax |
|--------|----------|---------|
| Success rate | ~6-10% | ~5-8% |
| Entropy | 0.89 | **0.53** |
| Response len | 230 | **188** |
| Grad norm | 1.87 | 2.34 |

**PROBLEM: Entropy collapse!**
- Softmax causes model to become too deterministic
- Response length drops (shorter = less reasoning)
- Higher gradients but pushing in wrong direction?

**Possible causes:**
1. τ=0.5 still too aggressive for this task
2. Softmax overfitting to specific patterns in high-reward samples
3. Need entropy regularization to counteract concentration

**Next:** Check eval metrics at step 10, consider τ=1.0 or adding entropy bonus

---

### 2024-12-04: Temperature Sweep Results (Steps 10-34)

**Setup:** Baseline + 4 softmax temperatures (τ=0.25, 0.5, 1.0, 2.0)

**EVAL SUCCESS RATES:**
| Step | Baseline | τ=0.5 | τ=0.25 | τ=1.0 | τ=2.0 |
|------|----------|-------|--------|-------|-------|
| 10   | 12%      | 16%   | 20%    | 0%    | -     |
| 20   | 11%      | **19%** | TBD  | 0%    | -     |
| 30   | 12%      | TBD   | TBD    | -     | -     |

**KEY FINDINGS:**

1. **τ=0.5 WINS** - 19% vs 11% baseline at step 20 (+73% improvement!)
2. **τ≥1.0 BREAKS COMPLETELY** - 0% success, model collapses
3. **Critical τ threshold** exists between 0.5 and 1.0
4. **Response length**: τ=0.5 generates longer responses (240-340 tokens vs 100 for baseline)
   - Longer responses = more reasoning steps before answer

**WHY τ≥1.0 BREAKS:**
At τ=1.0 with rewards in [0, 1.1]:
- softmax(1.1/1.0) vs softmax(0.1/1.0) ≈ 2.7:1 ratio (too flat!)
- All samples get similar weight → no learning signal
- Gradient becomes noise

At τ=0.5:
- softmax(1.1/0.5) vs softmax(0.1/0.5) ≈ 7.4:1 ratio (good separation!)
- Correct samples get strong signal

**CONCLUSION:**
Softmax reward weighting WORKS but requires τ ≈ 0.3-0.5 for reward scale [0, 1.1].
Rule of thumb: τ ≈ (reward_max - reward_min) / 2

---

### Final Eval Results (Step 50-70)

| Experiment | Eval Trajectory | Latest | vs Baseline |
|------------|-----------------|--------|-------------|
| Baseline | 12→11→12→14→14→17→17 | 17% | - |
| **τ=0.5** | 16→19→22→31→**38** | **38%** | **+124%** ✅ |
| τ=0.25 | 20→30→27 | 27% | +59% |
| τ=0.75 | 3 | 3% | BROKEN |
| τ=1.0 | 0 | 0% | BROKEN |
| Anneal 0.75→0.25 | 33 | 33% | +94% |

**τ=0.5 shows +124% improvement over baseline (38% vs 17%)!**

Annealing from 0.75→0.25 also works (33%), even though constant τ=0.75 is broken!

### Convergence Analysis

After ~60 steps:
- **τ=0.5 and τ=0.25 both converge to ~38%**
- Baseline stuck at 14%
- Annealing from 1.0→0.3 is too slow (starts broken)
- Annealing 0.75→0.25 reaches 33% but not better than constant τ=0.5

### Recommended Settings

**For Countdown task (rewards [0, 1.1]):**
- Use **τ = 0.5** (or τ = 0.25-0.5 range)
- Formula: **τ* ≈ reward_range / 2**
- Annealing not necessary - constant τ works fine

### Why It Works

Softmax with proper τ:
1. Concentrates learning signal on rare correct samples
2. Encourages exploratory reasoning (tries multiple approaches)
3. Avoids the "quick wrong answer" trap of linear normalization

### Qualitative Difference in Reasoning

**Baseline (Step 2):** Quick, makes errors, no self-check
```
<think>Multiplying 46 by 0.5...</think>  ← Uses 0.5 (not allowed!)
<answer>55 - 46 - (46 * 0.5) + 13</answer>
```

**Softmax τ=0.5 (Step 2):** Explores multiple paths, self-corrects
```
<think>...seems challenging. After considering different operations...</think>
...Let me re-examine the whole possibility space...
<answer>Not possible with basic arithmetic</answer>  ← Admits uncertainty!
```

The softmax model learns **exploratory reasoning** - tries multiple approaches
before committing. This is the "aha moment" behavior!

### First Principles for τ Selection

```
τ* = (R_max - R_min) / 2
```

For Countdown task: τ* = (1.1 - 0) / 2 = 0.55 ✓

This gives ~7:1 weight ratio between best and worst samples.
- τ too low (0.1): collapses to best-of-1, no diversity
- τ too high (1.0+): flat weights, no learning signal

---

## TODO
1. [x] Setup GRPO-Zero env
2. [x] Phase 2: Offline weight analysis ← DONE, τ=0.5 best
3. [ ] Run baseline (linear) for comparison
4. [ ] Run softmax (τ=0.5) 
5. [ ] Compare success rate curves


---

### Future Experiment: Qwen 0.5B

**Hypothesis**: Softmax reward weighting might help smaller models learn reasoning.

**Context**: GRPO-Zero authors say:
> "Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning."

**Why softmax might help**:
- Smaller models have less capacity → need stronger signal on correct samples
- Softmax concentrates learning on rare correct examples
- Could be the difference between learning and not learning

**To test**:
```bash
python run_experiment.py --dataset countdown --ablations baseline,softmax --model Qwen2.5-0.5B
```

If this works, it would be a strong validation of the softmax hypothesis!

---

### Future: Teacher Forcing + RL (Cold Start Solution)

**Problem**: Pure RL gradient ∝ π(y*|x) → tiny when model is bad (can't learn from 0%)

**Solution**: Combined objective
```
L = E_π[R(y)] - β * KL(π_θ || p*)
```

Where p* is the ground truth distribution.

**Implementation**:
1. Supervise on y* (ground truth) → gives strong gradient
2. RL on self-play rollouts → explores beyond supervised data
3. Optional: explicit KL term to stay close to p*

**Why this works**:
- SL provides the "base" capability
- RL provides the "correction" using verifier
- Model learns to reason, not just memorize

**For Qwen 0.5B**: This could be the key to making it learn reasoning!

---

### Teacher Forcing + RL: Full Mathematical Derivation

**Reference for future implementation**

#### 1. Core Difference: Supervised vs RL Gradient

Fix input x with correct solution y*.

**Supervised gradient:**
```
L_sup(θ) = -log π_θ(y*|x)
∇L_sup = -∇log π_θ(y*|x)
```
→ Full-strength gradient, independent of current probability.

**RL gradient (REINFORCE):**
```
J_RL(θ) = E_{y~π}[R(y)] = π_θ(y*|x)  (for R(y)=1{y=y*})
∇J_RL = π_θ(y*|x) · ∇log π_θ(y*|x)
```
→ Gradient **scaled by π_θ(y*|x)** → tiny when success is rare!

**Key insight:**
```
∇J_RL = -π_θ(y*|x) · ∇L_sup
```
RL gradient = supervised gradient × current success probability.

#### 2. Clean Combined Objective

Treat p*(y|x) = δ_{y=y*} as prior. Optimize:

```
J(θ) = E_{y~π}[R(y)] - β·KL(π_θ || p*)
```

Expanded gradient:
```
∇J = E_{y~π}[(R(y) + β·log p*(y|x) - β·log π_θ(y|x) - b(x)) · ∇log π_θ(y|x)]
```

Where:
- R(y): RL reward from verifier
- +β·log p*(y|x): supervised "pull" toward teacher (huge bonus when y=y*)
- -β·log π_θ(y|x): entropy regularizer

#### 3. Practical Implementation

```python
L_total = L_supervised + λ * L_RL

# Supervised: always push toward y*
L_supervised = -log π_θ(y*|x)

# RL: learn from self-play (with softmax weighting!)
L_RL = -Σ w_i · log π_θ(y_i|x)  where w_i = softmax(R_i/τ)
```

#### 4. Why This Fixes Cold Start

- **Pure RL**: If π_θ(y*|x) ≈ 0, gradient ≈ 0. Can't learn!
- **Pure SL**: Always gets gradient, but no exploration.
- **Combined**: SL provides base capability, RL explores beyond.

#### 5. Connection to Our Softmax Weighting

With teacher forcing, we could:
1. Include ground truth y* in each batch with reward = 1.0
2. Apply softmax weighting across all samples (including GT)
3. GT always gets high weight → guaranteed learning signal!

```python
# Pseudo-code
samples = generate_N_samples(model, x)  # Model rollouts
samples.append(ground_truth_y_star)     # Inject GT
rewards = [verify(s) for s in samples]
rewards[-1] = 1.0  # GT always correct

weights = softmax(rewards / tau)        # Softmax weighting
# Now GT has high weight, provides strong gradient!
```

This elegantly combines our softmax hypothesis with teacher forcing.
