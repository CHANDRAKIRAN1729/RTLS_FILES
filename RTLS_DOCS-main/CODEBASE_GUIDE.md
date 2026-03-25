# Reaching Through Latent Space — Complete Codebase Guide

> **Paper:** "Reaching Through Latent Space: From Joint Statistics to Path Planning in Manipulation"  
> **arXiv:** [2210.11779](https://arxiv.org/abs/2210.11779)  
> **Robot:** Franka Emika Panda — 7-DOF manipulator

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [The Two Models](#3-the-two-models)
4. [Datasets — Exactly What Each Field Means](#4-datasets--exactly-what-each-field-means)
5. [Training Pipeline: Step-by-Step](#5-training-pipeline-step-by-step)
   - 5A. [Stage 1 — VAE Training](#5a-stage-1--vae-training)
   - 5B. [Stage 2 — Binary Collision Classifier Training](#5b-stage-2--binary-collision-classifier-training)
6. [GECO Algorithm — What It Does and Why](#6-geco-algorithm--what-it-does-and-why)
7. [Path Planning via Latent Space Optimization](#7-path-planning-via-latent-space-optimization)
8. [Geometric Collision Checker (Robo3D)](#8-geometric-collision-checker-robo3d)
9. [File-by-File Reference](#9-file-by-file-reference)
10. [Config Files](#10-config-files)
11. [Critical Implementation Details (Gotchas)](#11-critical-implementation-details-gotchas)
12. [How This Connects to CBF Planning](#12-how-this-connects-to-cbf-planning)

---

## 1. High-Level Overview

The core idea is:

1. **Train a VAE** on collision-free robot configurations so that its latent space encodes a "manifold of valid poses."
2. **Train a binary collision classifier** on top of the frozen VAE encoder, so given a latent code `z` and obstacle parameters `o`, it predicts whether that configuration collides.
3. **Plan paths** by doing gradient descent in the 7D latent space, optimizing a composite loss:

$$L_{\text{total}} = L_{\text{goal}} + \lambda_p \cdot L_{\text{prior}} + \lambda_c \cdot L_{\text{collision}}$$

This approach avoids explicit inverse kinematics, sampling-based planning, or high-dimensional configuration-space search. The latent space is smooth, low-dimensional (7D), and encodes only physically meaningful robot configurations.

---

## 2. Architecture Diagram

```
                        TRAINING PHASE
                        ==============

  ┌──────────────────── Stage 1: VAE Training ────────────────────┐
  │                                                                │
  │  free_space_100k_train.dat                                     │
  │  ┌──────────────────────────┐                                  │
  │  │ Each row: 13 floats      │                                  │
  │  │  [j1..j7, ex,ey,ez,      │     ┌──────────┐                │
  │  │   gx,gy,gz]              │     │  Dataset  │                │
  │  └──────────────────────────┘     │ returns:  │                │
  │                                   │ x = first │                │
  │                                   │    10 dims │                │
  │       Input x (10-dim)           │ g = last  │                │
  │  ┌────────────────────────┐      │    3 dims  │                │
  │  │j1 j2 j3 j4 j5 j6 j7   │      └──────────┘                │
  │  │         ex ey ez        │                                   │
  │  └────────────┬───────────┘                                    │
  │               │ normalize: (x - μ) / σ                         │
  │               ▼                                                │
  │  ┌────────────────────────┐                                    │
  │  │       ENCODER           │                                   │
  │  │  10 → 2048 → 2048 →    │                                   │
  │  │  2048 → 2048            │                                   │
  │  │       ↓         ↓       │                                   │
  │  │     μ (7)    logVar(7)  │                                   │
  │  └───────┬─────────┬──────┘                                    │
  │          │         │                                           │
  │          ▼         ▼                                           │
  │   z = μ + ε·√(softplus(h) + 1e-5)    ← Reparameterization     │
  │          │                                                     │
  │          ▼                                                     │
  │  ┌────────────────────────┐                                    │
  │  │       DECODER           │                                   │
  │  │  7 → 2048 → 2048 →     │                                   │
  │  │  2048 → 2048 → 10      │                                   │
  │  └────────────┬───────────┘                                    │
  │               │ denormalize: x' = x_norm·σ + μ                 │
  │               ▼                                                │
  │  ┌────────────────────────┐                                    │
  │  │  Reconstructed x' (10) │                                    │
  │  │j1'j2'j3'j4'j5'j6'j7'  │                                   │
  │  │         ex'ey'ez'       │                                   │
  │  └────────────────────────┘                                    │
  │                                                                │
  │  Loss = GECO(MSE(x, x'), KL(q(z|x) || p(z)))                 │
  │       = KL + λ_geco · MSE                                     │
  │  λ_geco auto-adapts so MSE → goal (0.0008)                    │
  │                                                                │
  │  16,000 epochs, batch_size=4096, lr=0.0001                    │
  └────────────────────────────────────────────────────────────────┘


  ┌──────────── Stage 2: Collision Classifier Training ───────────┐
  │                                                                │
  │  collision_100k_train.dat                                      │
  │  ┌──────────────────────────────────┐                          │
  │  │ Each row: 15 floats              │                          │
  │  │  [j1..j7, ex,ey,ez,             │                          │
  │  │   ox,oy,oh,or,                  │                          │
  │  │   collision_label]              │                          │
  │  └──────────────────────────────────┘                          │
  │                                                                │
  │  Dataset returns 3 tensors:                                    │
  │    (x[0:10], obs[10:14], label[14])                           │
  │                                                                │
  │       x (10-dim)                obs (4-dim)                    │
  │  ┌──────────────┐          ┌───────────────┐                   │
  │  │j1..j7 ex,ey,ez│         │ ox, oy, oh, or│                  │
  │  └──────┬───────┘          └───────┬───────┘                   │
  │         │ normalize with           │ normalize with            │
  │         │ FREE-SPACE μ,σ           │ collision dataset μ,σ     │
  │         ▼                          │                           │
  │  ┌──────────────┐                  │                           │
  │  │ FROZEN ENCODER│ ← weights       │                           │
  │  │  (from Stage 1)│  loaded from   │                           │
  │  │               │  VAE checkpoint │                           │
  │  └──────┬───────┘                  │                           │
  │         │                          │                           │
  │       z (7-dim)                    │                           │
  │         │                          │                           │
  │         ▼                          ▼                           │
  │  ┌──────────────────────────────────┐                          │
  │  │     CLASSIFIER HEAD              │                          │
  │  │  concat(z, obs) = 11-dim        │                          │
  │  │  11 → 2048 → ELU → 2048 → ELU  │                          │
  │  │  → 2048 → ELU → 2048 → ELU → 1 │                          │
  │  └──────────────┬──────────────────┘                          │
  │                 │                                              │
  │                 ▼                                              │
  │          logit (1 scalar)                                      │
  │          sigmoid(logit) → P(collision)                         │
  │                                                                │
  │  Loss = BCE(logit, label)                                      │
  │  Only classifier head weights are trained.                     │
  │  Encoder + Decoder weights are frozen (loaded, not optimized). │
  │                                                                │
  │  16,000 epochs, batch_size=4096, lr=0.0001                    │
  │  (But only ~230 epochs found in saved checkpoint)              │
  └────────────────────────────────────────────────────────────────┘


                        INFERENCE PHASE
                        ===============

  ┌────────── Path Planning via Latent Optimization ──────────────┐
  │                                                                │
  │  Given: q_start (7 joints), e_target (3D EE goal),            │
  │         obstacles [ox, oy, oh, or]                            │
  │                                                                │
  │  1. Encode q_start → z_init via frozen encoder                │
  │  2. Gradient descent on z:                                     │
  │                                                                │
  │     For each step:                                             │
  │       x' = Decoder(z)              → get decoded config       │
  │       e' = x'[7:10]               → predicted EE position    │
  │       L_goal = ||e' - e_target||   → go to goal               │
  │       L_prior = 0.5·||z||²        → stay in distribution     │
  │       L_collision = Σ -log(1-σ(classifier(z, obs)))           │
  │                                    → avoid obstacles          │
  │       L_total = L_goal + λ_p·L_prior + λ_c·L_collision       │
  │       z ← z - lr·∇L_total                                    │
  │                                                                │
  │  3. Stop when ||e' - e_target|| < threshold (10mm)            │
  │  4. Decode final z → joint angles for execution               │
  └────────────────────────────────────────────────────────────────┘
```

---

## 3. The Two Models

### 3A. VAE (Variational Autoencoder) — `src/vae.py`

**Purpose:** Learn a 7D latent space that represents the manifold of valid (collision-free) Panda arm configurations.

| Component | Architecture |
|-----------|-------------|
| **Encoder** | `Linear(10→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→2048) → ELU` then splits into `μ = Linear(2048→7)` and `h = Linear(2048→7)` |
| **Variance** | `logVar = softplus(h) + 1e-5` (NOT raw log-variance, it's softplus-transformed) |
| **Reparameterization** | `z = μ + ε * √(logVar)`, where `ε ~ N(0, I)` |
| **Decoder** | `Linear(7→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→10)` |

**Key detail:** 4 hidden layers with 2048 units each, ELU activation (not ReLU). The bottleneck is 10D → 7D → 10D.

`forward(x)` returns `(x_reconstructed, μ, logVar)`.

### 3B. VAEObstacleBCE (VAE + Binary Collision Classifier) — `src/vae_obs.py`

**Purpose:** Predict whether a robot configuration (encoded as `z`) is in collision with a given cylindrical obstacle.

This model **inherits the exact same encoder/decoder as the VAE**, plus adds a classifier head:

| Component | Architecture |
|-----------|-------------|
| **Classifier Head** | `concat(z[7], obs_normalized[4]) = 11-dim input → Linear(11→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→2048) → ELU → Linear(2048→1)` |

The classifier takes:
- `z` (7D) — the latent code from the frozen encoder  
- `obs` (4D) — normalized obstacle parameters `[ox, oy, oh, or]`

And outputs a single scalar logit. Apply `sigmoid(logit)` to get `P(collision)`.

`forward(x, obs)` returns `(x_reconstructed, μ, logVar, obs_logit)`.

**Critical:** During classifier training, only the classifier head weights are optimized. The encoder and decoder weights are loaded from the pre-trained VAE and **frozen** (the optimizer only includes classifier parameters).

---

## 4. Datasets — Exactly What Each Field Means

### 4A. Free-Space Dataset (`free_space_100k_train.dat`)

Used for **VAE training** (Stage 1).

Each sample is a `.dat` file with rows of 13 floating-point values:

```
Index:   0    1    2    3    4    5    6    7    8    9    10   11   12
Field:  j1   j2   j3   j4   j5   j6   j7   ex   ey   ez   gx   gy   gz
        ╰────────── joint angles ──────────╯ ╰── EE pos ──╯ ╰── goal ──╯
        (7 values, radians)                  (3 values, m)   (3 values, m)
```

| Field | Index | Description | Units | Used By |
|-------|-------|-------------|-------|---------|
| `j1`–`j7` | 0–6 | Joint angles of all 7 Panda joints | **radians** | VAE input (encoded) |
| `ex`, `ey`, `ez` | 7–9 | End-effector (EE) XYZ position | **meters** | VAE input (encoded) |
| `gx`, `gy`, `gz` | 10–12 | Goal EE XYZ position | **meters** | Returned by dataset but **not used** by VAE loss |

**What the dataset class returns (`RobotStateDataset.__getitem__`):**
```python
return (jpos_ee_xyz[:10], goal_xyz[-3:])
#       ╰── j1..j7,ex,ey,ez ──╯  ╰── gx,gy,gz ──╯
```

**Normalization:** Z-score normalization using training set statistics:
- `mean_train` = mean of all 13 columns over training set → shape `(1, 13)`
- `std_train` = std of all 13 columns over training set → shape `(1, 13)`
- Applied to first 10 dims: `x_normalized = (x - mean[:10]) / std[:10]`

**Split:** 80% training / 20% validation (from the 100k training file). Plus a separate 10k test file.

**Important:** All configurations in this dataset are **collision-free** (free space). There are no obstacles or collision labels.

### 4B. Collision Dataset (`collision_100k_train.dat`)

Used for **collision classifier training** (Stage 2).

Each sample is a `.dat` file with rows of 15 floating-point values:

```
Index:   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
Field:  j1   j2   j3   j4   j5   j6   j7   ex   ey   ez   ox   oy   oh   or   c
        ╰────────── joint angles ──────────╯ ╰── EE pos ──╯ ╰── obstacle ──╯ ╰─╯
        (7 values, radians)                  (3 values, m)   (4 values)       label
```

| Field | Index | Description | Units | Used By |
|-------|-------|-------------|-------|---------|
| `j1`–`j7` | 0–6 | Joint angles of all 7 Panda joints | **radians** | Passed through frozen encoder to get z |
| `ex`, `ey`, `ez` | 7–9 | End-effector XYZ position | **meters** | Part of encoder input |
| `ox`, `oy` | 10–11 | Obstacle center XY position on ground plane | **meters** | Classifier input (obstacle param) |
| `oh` | 12 | Obstacle height (cylinder extends from z=0 to z=oh) | **meters** | Classifier input (obstacle param) |
| `or` | 13 | Obstacle radius (cylinder radius) | **meters** | Classifier input (obstacle param) |
| `c` | 14 | Collision label: `0` = free, `1` = collision | binary | BCE loss target |

**Obstacle geometry:** Each obstacle is a vertical cylinder sitting on the ground, defined by center (ox, oy), height oh, radius or.

**What the dataset class returns (`RobotObstacleDataset.__getitem__`):**
```python
return (jpos_ee_xyz[:10], obs_xyhr[10:14], obs_label[14])
#       ╰── j1..j7,ex,ey,ez ──╯  ╰── ox,oy,oh,or ──╯   ╰── 0 or 1 ──╯
```

**Normalization — this is critical:**
- The first 10 dimensions (joint angles + EE) are normalized using the **free-space dataset's** mean/std (NOT the collision dataset's own statistics). This ensures the encoder sees the same distribution it was trained on.
- The obstacle dimensions (indices 10–13) are normalized using the **collision dataset's** own mean/std.
- The collision label (index 14) is **NOT normalized** (its mean is forced to 0, std to 1 in the normalization arrays).

```python
# How normalization works inside RobotObstacleDataset:
mean_train[:10] = free_space_mean[:10]   # from free-space dataset
std_train[:10]  = free_space_std[:10]    # from free-space dataset
mean_train[10:14] = collision_mean[10:14]  # obstacle stats from collision data
std_train[10:14]  = collision_std[10:14]   # obstacle stats from collision data
mean_train[14] = 0.0   # label NOT normalized
std_train[14]  = 1.0   # label NOT normalized
```

### 4C. How Data Was Generated

The data files are pre-generated `.dat` binary files containing float32 arrays. They were likely created by:
1. Sampling random joint configurations within Panda's joint limits
2. Computing forward kinematics to get EE positions
3. For collision data: placing random cylindrical obstacles and using Robo3D geometric checking to label each (configuration, obstacle) pair as collision/free

The dataset sizes:
- **Training:** 100,000 samples each (`free_space_100k_train.dat`, `collision_100k_train.dat`)
- **Testing:** 10,000 samples each (`free_space_10k_test.dat`, `collision_10k_test.dat`)

---

## 5. Training Pipeline: Step-by-Step

Training happens in two sequential stages. Stage 2 depends on Stage 1's output.

The shell script `train_models.sh` automates both stages.

### 5A. Stage 1 — VAE Training

**Script:** `src/train_vae.py`  
**Config:** `config/vae_config/panda_10k.yaml`  
**Command:** `python train_vae.py --c ../config/vae_config/panda_10k.yaml`

#### Hyperparameters (from YAML config):
| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_dim` | 10 | 7 joint angles + 3 EE coordinates |
| `latent_dim` | 7 | Dimensionality of latent space z |
| `units_per_layer` | 2048 | Width of each hidden layer |
| `num_hidden_layers` | 4 | Number of hidden layers in encoder/decoder |
| `lr_vae` | 0.0001 | Adam learning rate |
| `batch_size` | 4096 | Mini-batch size |
| `epochs_vae` | **16,000** | Total training epochs |
| `g_goal` | 0.0008 | GECO target reconstruction error (MSE) |
| `g_lr` | 0.005 | GECO step size |
| `g_alpha` | 0.95 | GECO exponential moving average factor |
| `g_init` | 1 | GECO initial lambda |
| `save_every` | 50 | Save checkpoint every N epochs |
| `num_best_ckpt` | 1 | Keep only the best checkpoint (by AM AUC) |

#### What Happens Each Epoch:

```
For each batch in train_loader:
    1. x, goal = batch             # x is (batch, 10), goal is (batch, 3)
    2. x_normalized = normalize(x)
    3. x_recon, μ, logVar = model(x_normalized)
    4. MSE = mean_squared_error(x_recon, x_normalized)
    5. KL  = -0.5 * sum(1 + log(logVar) - μ² - logVar)  [per sample, then mean]
    6. loss = geco.loss(MSE, KL)    # = KL + λ_geco * MSE
    7. loss.backward()
    8. optimizer.step()
```

**The GECO algorithm** (see Section 6) automatically adjusts `λ_geco` so that the MSE reconstruction error converges toward the target `g_goal = 0.0008`. This replaces the typical β-VAE fixed weighting.

#### Evaluation Metrics (computed periodically):

1. **Sample Consistency (Posterior):** Encode x → decode → get EE' position via FK, compare with EE position from decoded joints. This checks if decoded (joints, EE) are self-consistent. Small = good.

2. **Sample Consistency (Prior):** Sample z ~ N(0,I) → decode → same FK consistency check. Tests if *random* latent codes decode to valid configurations.

3. **Activation Maximization (AM) Distance:** For random target EE positions, do gradient descent in latent space to reach the target (just L_goal, no obstacles). Measures how close the optimized configuration's EE gets to the target. The **AM AUC** metric summarizes this across thresholds — higher = better at reaching goals.

4. **AM AUC** is the metric used to select the best checkpoint.

#### Output Files:
- `model_params/panda_10k/snapshots/model.ckpt-015350.pt` — best checkpoint (selected by AM AUC, at epoch 15350)
- `model_params/panda_10k/model.ckpt-016000.pt` — final checkpoint (epoch 16000)
- `model_params/panda_10k/runs/` — TensorBoard logs
- `model_params/panda_10k/YYYYMMDD_HHMMSS-runcmd.json` — saved args for reproducibility

### 5B. Stage 2 — Binary Collision Classifier Training

**Script:** `src/train_vae_obs.py`  
**Config:** `config/vae_obs_config/panda_10k.yaml`  
**Command:** `python train_vae_obs.py --c ../config/vae_obs_config/panda_10k.yaml`

#### Hyperparameters (from YAML config):
| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr_obs` | 0.0001 | Adam learning rate (for classifier head only) |
| `batch_size` | 4096 | Mini-batch size |
| `epochs_obs` | **16,000** | Total training epochs |
| `save_every` | 2 | Save checkpoint every 2 epochs |
| `num_best_ckpt` | 1 | Keep best checkpoint (by val collision prediction rate) |
| `pretrained_checkpoint_path` | `snapshots/model.ckpt-015350.pt` | Path to pre-trained VAE |
| `vae_run_cmd_path` | `*-runcmd.json` | Path to VAE training args (to read architecture params) |

#### How It Loads the Pre-trained VAE:

```python
# 1. Create full VAEObstacleBCE model (has encoder + decoder + classifier)
model = VAEObstacleBCE(input_dim=10, latent_dim=7, units_per_layer=2048, num_hidden_layers=4)

# 2. Load pre-trained VAE weights
checkpoint_vae = torch.load(pretrained_checkpoint_path)
pretrained_dict = checkpoint_vae['model_state_dict']

# 3. Filter: only keep keys that exist in VAEObstacleBCE model
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 4. Overwrite model weights with pre-trained VAE weights
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 5. Create optimizer with ONLY classifier head parameters
obs_params = list(model.fc32.parameters()) + list(model.fc42.parameters())
for fc in model.fc_obs:
    obs_params += list(fc.parameters())
optimiser_obs = optim.Adam(obs_params, lr=0.0001)
```

**Key point:** `fc32` and `fc42` are the first two layers of the classifier head. `fc_obs` is the `ModuleList` of remaining classifier layers. The encoder (`fc1`, `fc21`, `fc22`) and decoder (`fc3`, `fc_dec`, `fc4`) weights are loaded from the VAE but **never updated** because they're not in the optimizer.

#### What Happens Each Epoch:

```
For each batch in train_loader:
    1. x, obs, label = batch       # x is (batch,10), obs is (batch,4), label is (batch,1)
    2. All are already normalized by the dataset class
    3. x_recon, μ, logVar, obs_logit = model(x, obs)
    4. loss = BCE_with_logits(obs_logit, label)    # Binary cross-entropy
    5. loss.backward()              # Gradients only flow into classifier params
    6. optimiser_obs.step()
```

**Note:** Even though the full model `forward()` runs the encoder and decoder, only the classifier loss is computed and only classifier parameters receive gradient updates.

#### Evaluation Metric:

**Collision Prediction Rate:** For each batch, count how many samples the classifier labels correctly (using 0.5 threshold on sigmoid(logit)), divide by total. This is essentially classification accuracy.

#### Output Files:
- `model_params/panda_10k/snapshots_obs/model.ckpt-015350-000230.pt` — best classifier checkpoint (VAE epoch 15350, classifier epoch 230)
- `model_params/panda_10k/model.ckpt-015350-016000.pt` — final combined checkpoint
- `model_params/panda_10k/checkpoint_obs` — text file pointing to latest checkpoint path
- `model_params/panda_10k/runs_obs/` — TensorBoard logs

**Checkpoint naming convention:** `model.ckpt-{vae_epoch}-{classifier_epoch}.pt`

The checkpoint contains:
```python
{
    'epoch_vae': 15350,           # Which VAE checkpoint was used
    'epoch_obs': 230,             # How many classifier epochs
    'model_state_dict': {...},    # Full model weights (encoder+decoder+classifier)
    'optimiser_obs_state_dict': {...}
}
```

---

## 6. GECO Algorithm — What It Does and Why

**File:** `src/geco.py`  
**Paper:** "Taming VAEs" (Rezende & Viola, 2018)

### The Problem
In a standard VAE, the loss is `L = MSE + β·KL`. Choosing β is hard:
- Too high β → blurry reconstructions (underfitting)
- Too low β → poor latent structure (KL collapses)

### The Solution: GECO (Generalized Constraint Optimization)
Instead of manually tuning β, GECO turns it into a constrained optimization:
- **Constraint:** MSE ≤ goal (e.g., 0.0008)
- **Objective:** minimize KL
- **Adaptive λ:** automatically increases/decreases to enforce the constraint

### Algorithm (from `geco.py`):

```python
def loss(self, err, kld):
    # err = MSE, kld = KL divergence
    
    constraint = err - self.goal    # How far MSE is from target
    
    # Exponential moving average of constraint
    self.cma = α * self.cma + (1-α) * constraint
    
    # Adaptive lambda update
    factor = exp(step_size * self.cma)
    self.geco_lambda = clamp(factor * self.geco_lambda, min, max)
    
    # Final loss
    loss = kld + self.geco_lambda * err
    return loss
```

**Intuition:**
- If MSE > goal: `constraint > 0` → `cma > 0` → `factor > 1` → λ increases → more pressure on reconstruction
- If MSE < goal: `constraint < 0` → `cma < 0` → `factor < 1` → λ decreases → more freedom for KL
- The `cma` (cumulative moving average) smooths out noise

**Parameters for this codebase:**
- `goal = 0.0008` — target MSE
- `step_size = 0.005` — how aggressively λ adapts
- `alpha = 0.95` — EMA smoothing factor
- `lambda_init = 1` — starting λ
- `lambda_min = 1e-5`, `lambda_max = 100000` — clamp range

**GECO is also used during planning** (in `evaluate_planning_proper.py`) to adaptively balance the three planning losses, though with different parameters.

---

## 7. Path Planning via Latent Space Optimization

**Files:** `src/evaluate_planning_proper.py`, `src/new_evaluate_planning_proper.py`

### Algorithm (Paper Algorithm 2):

```
Input: q_start (7 joint angles), e_target (3D goal), obstacles [(ox,oy,oh,or), ...]
Output: q_final (7 joint angles reaching the goal collision-free)

1. Encode start: z = Encoder(normalize(q_start, e_start))  [take μ only]
2. For step = 1 to max_steps:
   a. Decode: x' = Decoder(z), denormalize to get q', e'
   b. L_goal = ||e' - e_target||₂
   c. L_prior = 0.5·||z||₂²
   d. For each obstacle obs_i:
        logit_i = Classifier(z, normalize(obs_i))
        p_i = sigmoid(logit_i / temperature)
        L_collision += -log(1 - p_i)
   e. L_total = L_goal + λ_p·L_prior + λ_c·L_collision
   f. z ← z - lr·∇_z(L_total)     [Adam optimizer]
   g. If ||e' - e_target|| < threshold: DONE
3. Decode final z → q_final joint angles
```

### What Each Loss Does:

| Loss | Formula | Purpose |
|------|---------|---------|
| `L_goal` | $\|\|e' - e_{target}\|\|_2$ | Drive the end-effector toward the goal position |
| `L_prior` | $\frac{1}{2}\|\|z\|\|^2_2$ | Keep z near the origin of the latent space (where decoded configs are meaningful) |
| `L_collision` | $-\log(1 - \sigma(\text{logit}/T))$ | Push z away from regions where the classifier predicts collision |

**Why L_prior matters:** Without it, z can drift far from N(0,I) where the decoder was trained. Out-of-distribution z values decode to nonsensical joint configurations.

**Why L_collision matters:** Without it, the path goes straight through obstacles. The classifier gradient tells the optimizer which direction in latent space moves away from collision.

### Obstacle Normalization During Planning:

This is a common source of bugs. During planning, raw obstacle parameters `[ox, oy, oh, or]` must be normalized using the **same statistics** used during classifier training:

```python
obs_normalized = (obs_raw - mean_obs) / std_obs
# where mean_obs, std_obs come from collision_100k_train.dat, indices 10:14
```

---

## 8. Geometric Collision Checker (Robo3D)

**Files:** `src/sim/robot3d.py`, `src/sim/panda.py`, `src/sim/geometry.py`, `src/sim/object3d.py`, `src/sim/transform_matrix.py`

This is a **ground-truth geometric collision checker** — no learning, purely analytical.

### How It Works:

1. **Robot Model** (`Panda`): The 7-DOF Panda arm is defined using Modified DH parameters. Each joint link is approximated by one or more **capsules** (cylinder + hemisphere caps).

2. **Forward Kinematics**: Given 7 joint angles (in **degrees**), compute the 3D pose of every link using recursive transformation matrices.

3. **Obstacle Model**: Each obstacle is a vertical cylinder from `(ox, oy, 0)` to `(ox, oy, oh)` with radius `or`, represented as a Capsule.

4. **Collision Check**: Compute the minimum distance between every robot capsule and every obstacle capsule using segment-to-segment distance (`dist3d_segment_to_segment`). If any distance is 0, there's a collision.

### Key API:

```python
from sim.panda import Panda
from sim.robot3d import Robo3D

robot_def = Panda()                     # Define the robot
robo = Robo3D(robot_def)                # Create geometric model
is_colliding = robo.check_for_collision(
    jpos=[j1, j2, ..., j7],            # Joint angles in DEGREES
    obstacles_xyhr=[[ox, oy, oh, or]]   # Obstacle parameters
)
```

### CRITICAL: Unit Conversion

| Context | Joint Angle Units |
|---------|-------------------|
| Dataset (training data) | **Radians** |
| VAE input/output | **Radians** (after denormalization) |
| `Robo3D.check_for_collision()` | **Degrees** |
| `Panda.FK()` default | **Degrees** (unless `rad=True`) |

When using Robo3D to validate planned paths, you **must** convert: `degrees = radians × 180 / π`

### Robot Capsule Approximation:

Each of the 7 links has 1–2 capsules. The radii range from 0.025m to 0.06m. This is a conservative approximation — slightly larger than the real arm to provide safety margin.

```
Link 0: 1 capsule (r=0.06, from base to first joint)
Link 1: 1 capsule (r=0.06)
Link 2: 1 capsule (r=0.06)
Link 3: 1 capsule (r=0.06)
Link 4: 2 capsules (r=0.06 and r=0.025)
Link 5: 1 capsule (r=0.05)
Link 6: 2 capsules (r=0.04 and r=0.03, including gripper)
Base:   1 capsule (r=0.06)
```

---

## 9. File-by-File Reference

### Core Model Files

| File | Purpose | Key Details |
|------|---------|-------------|
| `src/vae.py` | VAE architecture definition | Classes: `Encoder`, `Decoder` nested inside `VAE`. 10→7→10 with 4×2048 hidden layers, ELU, softplus variance. |
| `src/vae_obs.py` | VAE + collision classifier | Extends VAE with `obstacle_collision_classifier(z, obs)` method. Classifier: 11→2048→2048→2048→2048→1. |
| `src/geco.py` | GECO adaptive loss weighting | Balances reconstruction vs KL divergence during training. |

### Training Scripts

| File | Purpose | Key Details |
|------|---------|-------------|
| `src/train_vae.py` | Train the VAE (Stage 1) | GECO loss, AM evaluation, FK consistency check. **16,000 epochs.** |
| `src/train_vae_obs.py` | Train collision classifier (Stage 2) | Loads frozen VAE, trains classifier head with BCE loss. Optimizer only includes classifier params. |
| `src/training_utils.py` | Helper: save args, move optimizer to device | `save_args()` saves timestamped JSON of all hyperparams. |
| `src/yaml_loader.py` | YAML config parser | Merges YAML config with argparse defaults. |

### Dataset Files

| File | Purpose | Key Details |
|------|---------|-------------|
| `src/robot_state_dataset.py` | Dataset for VAE (free-space configs) | Returns `(x[10], goal[3])`. Z-score normalization. 80/20 split. |
| `src/robot_obs_dataset.py` | Dataset for classifier (collision-labeled) | Returns `(x[10], obs[4], label[1])`. Uses free-space stats for first 10 dims. Label NOT normalized. |

### Evaluation & Planning Scripts

| File | Purpose | Key Details |
|------|---------|-------------|
| `src/evaluate_planning_proper.py` | Evaluation using classifier for both planning and checking | Uses classifier-based collision checking (threshold 0.9). |
| `src/new_evaluate_planning_proper.py` | **Fixed** evaluation: classifier for planning, Robo3D for checking | Two-stage pipeline. Ground-truth geometric validation. |
| `src/simulate_in_moveit.py` | MoveIt simulation (original) | ROS/MoveIt integration for real robot validation. |
| `src/new_simulate_in_moveit.py` | MoveIt simulation with tuned params | Phase 2 ablation-optimized parameters. |

### Visualization

| File | Purpose | Key Details |
|------|---------|-------------|
| `src/visualize_latent_am.py` | Latent space PCA visualization | Shows AM trajectories with/without prior and obstacle losses. Demonstrates why each loss is needed. |

### Simulation/Geometry Module (`src/sim/`)

| File | Purpose | Key Details |
|------|---------|-------------|
| `src/sim/panda.py` | Panda robot DH definition + differentiable FK | 7 links, DH params from Franka docs. `FK()` for batch forward kinematics (torch, differentiable). |
| `src/sim/robot3d.py` | Geometric robot model + collision checking | `Robo3D.check_for_collision(jpos, obstacles)` — ground truth. Joint angles in **degrees**. |
| `src/sim/geometry.py` | Capsule-capsule distance computation | `dist3d_capsule_to_capsule()` using segment-to-segment algorithm. |
| `src/sim/object3d.py` | 3D object with parent-child transforms | Recursive `transform_point()` through kinematic chain. |
| `src/sim/transform_matrix.py` | 4×4 homogeneous transform matrices | `z_rotation_matrix_tensor_batch()` for batch FK. |

### Shell Scripts

| File | Purpose |
|------|---------|
| `train_models.sh` | Run both Stage 1 + Stage 2 training sequentially |
| `run_proper_evaluation.sh` | Run classifier-based evaluation |
| `run_simulation_optimized.sh` | Run MoveIt simulation |
| `run_selected_scenes.sh` | Run evaluation on selected test scenes |
| `run_ablation_v5.sh` | Hyperparameter ablation study |
| `run_latent_visualization.sh` | Generate latent space visualizations |
| `run_new_simulation.sh` | Run new MoveIt simulation with Phase 2 params |

### Config Files

| File | Purpose |
|------|---------|
| `config/vae_config/panda_10k.yaml` | VAE training hyperparameters |
| `config/vae_obs_config/panda_10k.yaml` | Classifier training hyperparameters + pre-trained VAE path |

---

## 10. Config Files

### VAE Config (`config/vae_config/panda_10k.yaml`)

```yaml
# Model architecture
input_dim         : 10        # 7 joints + 3 EE
latent_dim        : 7         # Latent space dimensionality
units_per_layer   : 2048      # Hidden layer width
num_hidden_layers : 4         # Number of hidden layers
lr_vae            : 0.0001    # Adam learning rate

# Training
batch_size   : 4096           # Large batch for stable GECO
epochs_vae   : 16000          # Total training epochs
seed         : 1

# GECO parameters
g_goal  : 0.0008   # Target reconstruction MSE
g_lr    : 0.005    # GECO step size
g_alpha : 0.95     # GECO EMA factor
g_init  : 1        # Initial lambda
g_min   : 0.00001  # Lambda lower bound
g_max   : 100000   # Lambda upper bound

# Activation maximization evaluation
am_lr        : 0.03    # AM gradient descent learning rate
am_steps     : 300     # AM optimization steps
am_samples   : 100     # Number of AM targets to evaluate
am_auc_parts : 100     # AUC resolution
am_auc_max   : 0.1     # Maximum distance for AUC (10cm)

# Data
path_to_dataset : '../data'
train_data_name : 'free_space_100k_train.dat'
test_data_name  : 'free_space_10k_test.dat'

# Checkpointing
save_every     : 50    # Save every 50 epochs
num_best_ckpt  : 1     # Keep 1 best checkpoint
```

### Classifier Config (`config/vae_obs_config/panda_10k.yaml`)

```yaml
# Classifier training
lr_obs     : 0.0001   # Adam learning rate (classifier head only)
batch_size : 4096
epochs_obs : 16000     # Total classifier training epochs
seed       : 1

# Data
path_to_dataset            : '../data'
train_data_name            : 'collision_100k_train.dat'
test_data_name             : 'collision_10k_test.dat'

# Pre-trained VAE (loaded and frozen)
vae_run_cmd_path           : '../model_params/panda_10k/YYYYMMDD-runcmd.json'
pretrained_checkpoint_path : '../model_params/panda_10k/snapshots/model.ckpt-015350.pt'

# Checkpointing
save_every     : 2     # Save every 2 epochs (converges faster than VAE)
num_best_ckpt  : 1     # Keep 1 best checkpoint
```

---

## 11. Critical Implementation Details (Gotchas)

### 1. Normalization Consistency
The collision classifier's encoder input MUST be normalized with free-space statistics, not collision dataset statistics. The `RobotObstacleDataset` handles this automatically by loading the free-space dataset's mean/std.

### 2. Radians vs Degrees
- **Training data:** radians
- **VAE input/output:** radians (after denormalization)
- **Robo3D collision checker:** degrees
- **Panda.FK():** degrees by default, radians if `rad=True`
- Always convert when bridging between VAE output and Robo3D.

### 3. Softplus Variance (NOT Log-Variance)
Despite the variable name `logVar`, the VAE uses `softplus(h) + 1e-5` — this is NOT `log(variance)`. The KL divergence formula in `train_vae.py` uses `torch.log(logvar)` which means it's computing `log(softplus(h) + 1e-5)`. This is different from the standard β-VAE formula.

### 4. Classifier Inputs Are Normalized
During planning, obstacle parameters must be normalized before passing to the classifier. Raw `[ox, oy, oh, or]` values from scenario generation must be transformed:
```python
obs_normalized = (obs_raw - mean_obs[10:14]) / std_obs[10:14]
```

### 5. Frozen Encoder During Classifier Training
The optimizer in `train_vae_obs.py` only includes `fc32`, `fc42`, and `fc_obs` parameters. The encoder parameters (`fc1`, `fc21`, `fc22`) and decoder parameters (`fc3`, `fc_dec`, `fc4`) are present in the model but never updated.

### 6. Collision Label Not Normalized
In `RobotObstacleDataset`, the collision label (index 14) has its mean set to 0 and std set to 1 in the normalization arrays. This means the label passes through unchanged — it's always exactly 0.0 or 1.0.

### 7. Forward Kinematics in Training
The FK used during VAE evaluation (`Panda.FK()`) is differentiable (torch-based) and used to verify self-consistency of decoded configurations. It is NOT used in the training loss — only for evaluation metrics.

### 8. Actual Trained Epochs
- VAE: Trained for 16,000 epochs, best checkpoint at epoch 15,350 (selected by AM AUC)
- Classifier: Config says 16,000 but saved checkpoint shows epoch 230 (`model.ckpt-015350-000230.pt`), suggesting either early stopping or interrupted training. The classifier converges much faster than the VAE.

---

## 12. How This Connects to CBF Planning

If you're implementing Control Barrier Functions (CBF) on top of this codebase, here's what matters:

### What You Get From This Codebase:

1. **A learned latent space** where gradient descent produces valid robot configurations
2. **A differentiable collision predictor** `σ(Classifier(z, obs))` that gives you P(collision) for any latent code z and obstacle params
3. **A differentiable decoder** `Decoder(z) → (q, e)` that maps latent codes to joint configurations
4. **Differentiable FK** `Panda.FK(q)` that maps joint angles to EE positions

### How CBF Could Use These:

A CBF $h(x) \geq 0$ defines a safe set. In this latent-space context:

- **State:** $z \in \mathbb{R}^7$ (latent code)
- **Safety function:** $h(z, \text{obs}) = 1 - \sigma(\text{Classifier}(z, \text{obs}))$ — positive when collision-free
- **Dynamics:** $\dot{z} = u$ (gradient descent step in latent space)
- **CBF constraint:** $\dot{h}(z) + \alpha \cdot h(z) \geq 0$ ensures the system stays in the safe set

The classifier gradient $\nabla_z \text{Classifier}(z, \text{obs})$ gives you the direction in latent space that moves toward/away from collision. This is already implicitly used in the current `L_collision` term, but a CBF formulation would give formal safety guarantees.

### Key Data Flow for CBF:

```
z (7D latent) 
  → Decoder → q (7 joints, radians) + e (3D EE position)
  → Classifier(z, obs_normalized) → logit → sigmoid → P(collision)
  → h(z) = 1 - P(collision)    ← This is your safety barrier function
  → ∇_z h(z) via autograd      ← This is your CBF gradient
```

### What You'd Need to Add:
- CBF-QP solver that constrains latent-space velocity $\dot{z}$ to satisfy $\dot{h} + \alpha h \geq 0$
- Integration with a nominal controller (the current gradient descent could serve as nominal)
- Possibly a dynamics model in latent space (currently assumes quasi-static, instantaneous z updates)

---

## Summary of Training Time & Compute

| Stage | Epochs | Batch Size | Samples | Batches/Epoch | Parameters Trained |
|-------|--------|------------|---------|---------------|-------------------|
| VAE | 16,000 | 4,096 | 80,000 (80% of 100k) | ~20 | All encoder + decoder (~34M params) |
| Classifier | 230* | 4,096 | 80,000 (80% of 100k) | ~20 | Only classifier head (~17M params) |

*Actual saved checkpoint epoch; config allows up to 16,000.

Total model parameter estimate (rough):
- Encoder: 10×2048 + 3×2048² + 2×2048×7 ≈ 12.6M
- Decoder: 7×2048 + 3×2048² + 2048×10 ≈ 12.6M
- Classifier: 11×2048 + 3×2048² + 2048×1 ≈ 12.6M
- **Total: ~37.8M parameters**
