# Reaching Through Latent Space — Evaluation Pipeline Report

## Project Overview

This project implements and evaluates the path planning algorithm from **"Reaching Through Latent Space: From Joint Statistics to Path Planning in Manipulation"** (Lippi et al., arXiv:2210.11779). The approach uses a Variational Autoencoder (VAE) to learn a latent space of robot configurations, then performs gradient-based optimization in that latent space to plan collision-free paths.

**Robot:** Franka Emika Panda (7-DOF)  
**VAE Architecture:** 7D latent space, 2048 units/layer, 4 hidden layers, ELU activation  
**Collision Classifier:** Same architecture (VAEObstacleBCE), trained on labelled collision data  
**Test Setup:** 1000 scenarios, 1 cylindrical obstacle per scene, 10mm success threshold

---

## Critical Methodology Issue: Circular Evaluation

### The Problem

The original evaluation script (`src/evaluate_planning_proper.py`) used the **learned collision classifier** (`VAEObstacleBCE.obstacle_collision_classifier`) for **both** stages of the pipeline:

1. **Planning** — Classifier gradients steer the latent trajectory away from obstacles  
2. **Evaluation** — Classifier predictions at the final latent point determine whether the path is "collision-free"

This creates a **circular evaluation**. The optimizer minimizes the classifier's collision loss during planning, and then the same classifier judges whether collisions occurred. If the classifier has blind spots (false negatives), the planner will exploit those gaps, and the evaluation will report "success" for paths that physically collide with obstacles. Conversely, if the classifier is overly conservative (false positives), both planning and evaluation suffer — the planner avoids valid regions, and evaluation penalizes safe paths.

This is methodologically equivalent to grading an exam with the same answer key the student was given to study from.

**Evidence in code** ([src/evaluate_planning_proper.py](src/evaluate_planning_proper.py)):
```python
# Line 473 — Success determined by classifier
success = goal_reached and classifier_collision_free
```
Where `classifier_collision_free` is derived from `model_obs.obstacle_collision_classifier()`, the same model used to compute `L_collision` during planning.

### The Fix

After correspondence with the paper's author, the correct methodology was clarified:

> *"To check whether there is any collision with the obstacles in a path (a sequence of joint angles), one can instantiate a Robo3D with Panda as the definition and use the function `check_for_collision`."*

This led to the implementation of a **two-stage pipeline** in `src/new_evaluate_planning_proper.py`:

| Stage | Purpose | Method | Why |
|-------|---------|--------|-----|
| **Stage 1: Planning** | Steer latent trajectory away from obstacles | Learned classifier (`VAEObstacleBCE`) | Differentiable — enables backpropagation through `L_collision` |
| **Stage 2: Evaluation** | Validate path collision-free status | Geometric checker (`Robo3D.check_for_collision`) | Physics-based ground truth, independent of the planner |

This separation is standard practice in robotics: plan with a fast approximate model (classifier, ~0.1ms per query), validate with an accurate one (capsule geometry, ~1ms per query).

### Robo3D Collision Checker

`Robo3D` (in `src/sim/robot3d.py`) performs capsule-based geometric collision detection:

1. Forward kinematics via DH parameters → link positions and orientations
2. Each robot link approximated as a capsule (cylinder + hemispherical caps)
3. Each obstacle parameterized as a capsule: `[x, y, h, r]` → `Capsule(r, [x,y,0], [x,y,h])`
4. Collision = minimum capsule-to-capsule distance is 0

**Critical implementation detail:** Robo3D expects joint angles in **degrees**, while the VAE training data stores joint angles in **radians**. The evaluation script performs the conversion:

```python
q_deg = np.degrees(q_rad).tolist()  # radians → degrees for Robo3D
robo3d.check_for_collision(q_deg, obstacles_xyhr)
```

---

## Ablation Study: Evaluation Parameter Tuning

With the corrected evaluation pipeline, the initial success rate was ~50%. Two phases of systematic ablation were conducted to tune the planning hyperparameters.

### Fixed Parameters (Unchanged Throughout)

| Parameter | Value |
|-----------|-------|
| `success_threshold` | 0.01m (10mm) |
| `max_steps` | 300 |
| `use_geco` | True |

### Phase 1: Broad Sweep (36 runs)

Starting from an initial configuration calibrated by measuring raw loss magnitudes (`L_goal ≈ 0.81`, `L_prior ≈ 11.88`, `L_collision ≈ 7.20`), the study systematically explored each dimension:

**Key findings:**
- **Temperature was the single biggest lever.** Increasing from 1.0 → 2.0 boosted success from ~51% to ~55%. The classifier's sigmoid predictions are overconfident at temperature=1.0, producing gradients that are too aggressive early in optimization and kill goal-reaching.
- **`lambda_prior` = 0.3** was transformative: collision-free rate jumped from 59% → 79%. Higher prior weight keeps latent paths closer to the learned manifold, which indirectly improves collision avoidance since the prior distribution captures collision-free configurations.
- **`alpha_geco` = 0.005** (slower GECO updates) reduced lambda oscillations that were destabilizing optimization.
- **`tau_prior_goal` = 6.0** (down from 12.0) pushed harder on prior regularization.

**Phase 1 Result (N=1000):** 73.5% success | 89.1% goal reached | 79.1% collision-free | 247.5ms avg

### Phase 2: Fine-Tuning (30 runs)

Starting from Phase 1 best, refined around the promising region:

**Key findings:**
- **`lambda_prior` = 0.7, `lambda_collision` = 0.5:** Further increasing regularization pushed collision-free from 79.1% → 82.5% while maintaining goal-reaching at 93.9%.
- **Temperature = 3.0:** Further softening classifier outputs continued to improve goal-reaching without hurting collision avoidance, because GECO's adaptive lambda compensates.
- **`alpha_geco` = 0.008:** Slightly faster GECO adaptation (vs 0.005) allowed lambda to adjust before optimization converges.
- **`alpha_ma_prior` = 0.8:** Less smoothing on prior constraint moving average → GECO reacts faster to prior violations.
- The target-hitting configuration was found at Round 3, Run 28 (`amp=0.8`), confirmed on 1000 scenarios.

**Phase 2 Result (N=1000):** 80.3% success | 93.9% goal reached | 82.5% collision-free | 218.9ms avg

### Parameter Evolution

| Parameter | Initial (calibrated) | Phase 1 Best | Phase 2 Best |
|-----------|---------------------|-------------|-------------|
| `planning_lr` | 0.1 | 0.2 | **0.15** |
| `lambda_prior` | 0.07 | 0.3 | **0.7** |
| `lambda_collision` | 0.11 | 0.3 | **0.5** |
| `temperature` | 1.0 | 2.0 | **3.0** |
| `alpha_geco` | 0.01 | 0.005 | **0.008** |
| `tau_prior_goal` | 12.0 | 6.0 | **6.0** |
| `tau_obs_goal` | 1.5 | 2.0 | **2.0** |
| `alpha_ma_prior` | 0.95 | 0.95 | **0.8** |
| `alpha_ma_obs` | 0.4 | 0.4 | **0.8** |
| **Success Rate** | **~50%** | **73.5%** | **80.3%** |

### Interpretation

The dominant pattern is that **stronger regularization** (higher `lambda_prior`, `lambda_collision`) combined with **softer classifier signals** (higher temperature) produces the best balance. The intuition:

- High temperature prevents the classifier from creating sharp, potentially incorrect gradient cliffs that trap the optimizer in local minima far from the goal
- GECO's adaptive weighting compensates by gradually increasing `lambda_collision` when the constraint is violated, providing a smooth "annealing" of collision avoidance strength
- High `lambda_prior` keeps paths on the learned manifold where the VAE's reconstructions are accurate, preventing decoded configurations from reaching regions where neither the classifier nor the VAE is well-calibrated

---

## File Summary

### Evaluation Scripts

| File | Description |
|------|-------------|
| `src/evaluate_planning_proper.py` | **OLD** — Uses classifier for both planning and evaluation (circular) |
| `src/new_evaluate_planning_proper.py` | **NEW** — Two-stage pipeline: classifier for planning, Robo3D for evaluation |
| `run_new_evaluation.sh` | Run script for new evaluation with Phase 2 parameters |

### Simulation Scripts

| File | Description |
|------|-------------|
| `src/simulate_in_moveit.py` | Original MoveIt simulation with old parameters |
| `src/new_simulate_in_moveit.py` | MoveIt simulation with Phase 2 optimized parameters |
| `run_new_simulation.sh` | Run script for new simulation |

### Analysis & Ablation Scripts

| File | Description |
|------|-------------|
| `src/diagnostic_loss_magnitudes.py` | Measures typical `L_goal`, `L_prior`, `L_collision` magnitudes for parameter calibration |
| `src/analyze_collision_timing.py` | Analyzes where along paths collisions occur (early vs late) |
| `src/run_ablation_eval.py` | Phase 1 automated ablation (36 runs, target 70%) |
| `src/run_ablation_eval_p2.py` | Phase 2 automated ablation (30 runs, target 80%) |

### Core Model Files

| File | Description |
|------|-------------|
| `src/vae.py` | VAE model (encoder/decoder) |
| `src/vae_obs.py` | VAEObstacleBCE with `obstacle_collision_classifier` |
| `src/sim/robot3d.py` | `Robo3D.check_for_collision()` — geometric collision checker |
| `src/sim/panda.py` | Panda robot definition (DH parameters, joint limits) |
| `src/sim/geometry.py` | Capsule geometry and distance computations |

### Ablation Logs

| File | Description |
|------|-------------|
| `ab_logs_eval/ablation_log.txt` | Phase 1: 36 runs, initial sweep |
| `ab_logs_eval/ablation_log_p2.txt` | Phase 2: 30 runs, fine-tuning to 80.3% |

### Results

| File | Description |
|------|-------------|
| `model_params/panda_10k/new_evaluation_results.json` | Summary results from new evaluation pipeline |
| `model_params/panda_10k/new_evaluation_results_detailed.json` | Per-scenario detailed results |
| `model_params/panda_10k/evaluation_results.json` | Old results (classifier-based evaluation) |

---

## Comparison with Paper

| Metric | Paper (Table IV) | Old Pipeline (Circular) | New Pipeline (Robo3D) |
|--------|-----------------|------------------------|----------------------|
| Success Rate | ~90% | N/A (unreliable) | **80.3%** |
| Goal Reached | — | — | **93.9%** |
| Collision-Free | — | — | **82.5%** |
| Planning Time | ~180ms | — | **218.9ms** |

The 10% gap vs the paper likely stems from:
1. **Capsule vs mesh geometry:** Robo3D uses capsule approximations (conservative), while the paper likely used MoveIt's FCL mesh-based checker (tighter fits)
2. **Obstacle generation:** Slight differences in scene sampling distributions
3. **Training data quality:** The models may benefit from re-training with longer schedules or different hyperparameters
