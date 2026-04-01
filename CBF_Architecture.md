# CBF Architecture - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Phase 1: VAE Training Architecture](#phase-1-vae-training-architecture)
4. [Phase 2: CBF Training Architecture](#phase-2-cbf-training-architecture)
5. [Phase 3: CBF-based Planning Architecture](#phase-3-cbf-based-planning-architecture)
6. [End-to-End Data Flow](#end-to-end-data-flow)
7. [Implementation Algorithm](#implementation-algorithm)
8. [Configuration Parameters](#configuration-parameters)
9. [Key Classes and Functions Reference](#key-classes-and-functions-reference)

---

## Overview

This project implements a **Control Barrier Function (CBF)** integrated with a **Variational Autoencoder (VAE)** for safe robot motion planning. The CBF provides formal safety guarantees by learning a barrier function B(x) that distinguishes safe states (B >= 0) from unsafe/collision states (B < 0).

### Key Innovations

| Approach | Method | Output | Safety Guarantee |
|----------|--------|--------|------------------|
| Traditional | Binary Classifier | P(collision) in [0, 1] | Probabilistic |
| **CBF (This Project)** | Barrier Function | B(x) in R | Formal (dB/dt >= -alpha*B) |

### Core Components

1. **VAE**: Learns a compressed latent representation of valid robot configurations
2. **CBF Head**: Predicts barrier function values given latent state and obstacle parameters
3. **Safe Planner**: Uses CBF gradient-based projection for collision-free path planning

---

## Project Structure

```
/home/rck/RTLS/
├── 3.CBFS/                      # Basic CBF tutorial (2D systems)
├── 4.RTLS/                      # Original RTLS without CBF (binary classifier)
└── 5.RTLS_CBFS/                 # Main project: RTLS with CBF integration
    ├── config/
    │   ├── vae_config/          # VAE training configuration
    │   │   └── panda_10k.yaml
    │   ├── vae_obs_config/      # Obstacle classifier configuration (legacy)
    │   │   └── panda_10k.yaml
    │   └── vae_cbf_config/      # CBF training configuration
    │       └── panda_10k.yaml
    ├── data/                    # Training/test datasets
    │   ├── free_space_100k_train.dat
    │   ├── free_space_100k_test.dat
    │   ├── collision_100k_train.dat
    │   └── collision_100k_test.dat
    ├── model_params/            # Saved model checkpoints
    ├── src/                     # Source code
    │   ├── sim/                 # Robot simulation modules
    │   │   ├── panda.py         # Panda robot kinematics
    │   │   ├── robot3d.py       # 3D collision checking
    │   │   └── geometry.py      # Geometric primitives
    │   ├── vae.py               # Base VAE model
    │   ├── vae_obs.py           # VAE with binary classifier (legacy)
    │   ├── vae_cbf.py           # VAE with CBF head
    │   ├── train_vae.py         # VAE training script
    │   ├── train_vae_obs.py     # Obstacle classifier training (legacy)
    │   ├── train_vae_cbf.py     # CBF training script
    │   ├── robot_state_dataset.py    # Free-space dataset loader
    │   ├── robot_cbf_dataset.py      # CBF dataset loader
    │   ├── evaluate_planning.py      # Original planning evaluation
    │   └── evaluate_planning_cbf.py  # CBF planning evaluation
    ├── train_models.sh          # Train VAE + obstacle classifier
    ├── train_cbf.sh             # Train CBF model
    ├── run_evaluation_cbf.sh    # Evaluate CBF planning
    └── CBF.md                   # Detailed documentation
```

---

## Phase 1: VAE Training Architecture

### Purpose

Learn a compact latent representation of valid robot configurations that captures the robot's kinematic structure while enabling smooth interpolation in latent space.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VAE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Robot State x = [j1, j2, ..., j7, ee_x, ee_y, ee_z]                 │
│         Dimension: 10 (7 joint angles + 3D end-effector position)           │
│                                     │                                        │
│                                     ▼                                        │
│         ┌───────────────────────────────────────────────────────┐           │
│         │                     ENCODER                            │           │
│         ├───────────────────────────────────────────────────────┤           │
│         │  fc1:    Linear(10 → 2048) + ELU activation           │           │
│         │  fc2[0]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc2[1]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc2[2]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc2[3]: Linear(2048 → 2048) + ELU                    │           │
│         │                       │                                │           │
│         │              ┌────────┴────────┐                       │           │
│         │              ▼                 ▼                       │           │
│         │     fc21: Linear(2048→7)  fc22: Linear(2048→7)        │           │
│         │           │                    │                       │           │
│         │           ▼                    ▼                       │           │
│         │         μ (mean)         log(σ²) (log variance)        │           │
│         └───────────────────────────────────────────────────────┘           │
│                                     │                                        │
│                                     ▼                                        │
│         ┌───────────────────────────────────────────────────────┐           │
│         │            REPARAMETERIZATION TRICK                    │           │
│         │                                                        │           │
│         │         z = μ + ε * exp(0.5 * log(σ²))                │           │
│         │         where ε ~ N(0, I)                              │           │
│         │         Latent dimension: 7                            │           │
│         └───────────────────────────────────────────────────────┘           │
│                                     │                                        │
│                                     ▼                                        │
│         ┌───────────────────────────────────────────────────────┐           │
│         │                     DECODER                            │           │
│         ├───────────────────────────────────────────────────────┤           │
│         │  fc3:    Linear(7 → 2048) + ELU activation            │           │
│         │  fc4[0]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc4[1]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc4[2]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc4[3]: Linear(2048 → 2048) + ELU                    │           │
│         │  fc41:   Linear(2048 → 10)                            │           │
│         └───────────────────────────────────────────────────────┘           │
│                                     │                                        │
│                                     ▼                                        │
│  OUTPUT: Reconstructed State x' = [j1', j2', ..., j7', ee_x', ee_y', ee_z'] │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VAE TRAINING DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DATA LOADING                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  File: free_space_100k_train.dat                                │     │
│     │  Format: [j1, j2, ..., j7, ee_x, ee_y, ee_z, goal_x, y, z]     │     │
│     │  Robot states NOT in collision with any obstacles               │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  2. PREPROCESSING (RobotStateDataset)                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  a. Extract: jpos_ee_xyz = data[:, 0:10]   # State              │     │
│     │  b. Extract: goal_xyz = data[:, 10:13]     # Goal position      │     │
│     │  c. Compute: mean, std over training set                        │     │
│     │  d. Normalize: x_norm = (x - mean) / std                        │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  3. FORWARD PASS                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Input:  x_norm (batch_size x 10)                               │     │
│     │  Encode: z, μ, logVar = encoder(x_norm)                         │     │
│     │  Decode: x_recon = decoder(z)                                   │     │
│     │  Output: x_recon (batch_size x 10)                              │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  4. LOSS COMPUTATION (GECO)                                                 │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Reconstruction Loss:                                           │     │
│     │    L_recon = MSE(x_recon, x_norm)                               │     │
│     │                                                                  │     │
│     │  KL Divergence:                                                 │     │
│     │    L_KL = -0.5 * sum(1 + logVar - μ² - exp(logVar))            │     │
│     │                                                                  │     │
│     │  GECO Balancing:                                                │     │
│     │    - Adaptively adjusts weight between L_recon and L_KL        │     │
│     │    - Ensures reconstruction quality meets threshold (g_goal)    │     │
│     │    - λ_GECO updated via: λ = λ * exp(g_lr * (L_recon - g_goal))│     │
│     │                                                                  │     │
│     │  Total Loss:                                                    │     │
│     │    L_VAE = λ_GECO * L_recon + L_KL                              │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  5. BACKPROPAGATION & UPDATE                                                │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  optimizer.zero_grad()                                          │     │
│     │  L_VAE.backward()                                               │     │
│     │  optimizer.step()                                               │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Input/Output Specification

| Component | Input | Output | Dimensions |
|-----------|-------|--------|------------|
| **Dataset** | free_space_100k_train.dat | (jpos_ee_xyz, goal_xyz) | (N x 10, N x 3) |
| **Encoder** | Normalized robot state | (z, μ, logVar) | (B x 7, B x 7, B x 7) |
| **Decoder** | Latent code z | Reconstructed state | B x 10 |
| **Loss** | (x, x_recon, μ, logVar) | Scalar loss | 1 |

### Key Files

- **Model Definition**: `src/vae.py` (lines 8-79)
- **Training Script**: `src/train_vae.py`
- **Dataset Loader**: `src/robot_state_dataset.py`
- **Configuration**: `config/vae_config/panda_10k.yaml`

---

## Phase 2: CBF Training Architecture

### Purpose

Learn a Control Barrier Function B(x) that:
- Returns B(x) >= 0 for safe states (no collision)
- Returns B(x) < 0 for unsafe states (collision)
- Satisfies the CBF constraint: dB/dt >= -alpha * B for safety guarantees

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VAECBF ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT 1: Robot State x = [j1, ..., j7, ee_x, ee_y, ee_z]  (dim: 10)        │
│  INPUT 2: Obstacle params obs = [x, y, h, r]               (dim: 4)         │
│                                                                              │
│           Robot State                        Obstacle Params                 │
│                │                                    │                        │
│                ▼                                    │                        │
│  ┌─────────────────────────────────┐               │                        │
│  │         ENCODER (Frozen)        │               │                        │
│  │  (Same architecture as VAE)     │               │                        │
│  │                                 │               │                        │
│  │  fc1 → fc2[0-3] → fc21/fc22     │               │                        │
│  │        ↓                        │               │                        │
│  │  z = reparameterize(μ, logVar)  │               │                        │
│  └─────────────────────────────────┘               │                        │
│                │                                    │                        │
│                │ z (dim: 7)                        │ obs (dim: 4)           │
│                │                                    │                        │
│                └────────────┬───────────────────────┘                        │
│                             │                                                │
│                             ▼                                                │
│           ┌─────────────────────────────────────────────────┐               │
│           │              CONCATENATION                       │               │
│           │         [z, obs] = (dim: 7 + 4 = 11)            │               │
│           └─────────────────────────────────────────────────┘               │
│                             │                                                │
│                             ▼                                                │
│           ┌─────────────────────────────────────────────────┐               │
│           │                 CBF HEAD                         │               │
│           ├─────────────────────────────────────────────────┤               │
│           │  fc_cbf1:    Linear(11 → 2048) + ELU            │               │
│           │  fc_cbf[0]:  Linear(2048 → 2048) + ELU          │               │
│           │  fc_cbf[1]:  Linear(2048 → 2048) + ELU          │               │
│           │  fc_cbf[2]:  Linear(2048 → 2048) + ELU          │               │
│           │  fc_cbf[3]:  Linear(2048 → 2048) + ELU          │               │
│           │  fc_cbf_out: Linear(2048 → 1)                   │               │
│           └─────────────────────────────────────────────────┘               │
│                             │                                                │
│                             ▼                                                │
│           ┌─────────────────────────────────────────────────┐               │
│           │              BARRIER VALUE                       │               │
│           │         B(x, obs) ∈ ℝ                           │               │
│           │                                                  │               │
│           │    Interpretation:                               │               │
│           │      B >= 0  →  SAFE (no collision)             │               │
│           │      B < 0   →  UNSAFE (collision)              │               │
│           └─────────────────────────────────────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CBF Loss Function

The CBF training uses a three-term loss function derived from CBF theory:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CBF LOSS FUNCTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L_CBF = L_safe + L_unsafe + λ_cbf * L_constraint                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TERM 1: Safe State Loss (Equation 6 from CBF theory)               │    │
│  │                                                                      │    │
│  │    L_safe = (1/N_safe) * Σ max(0, -B(x))                            │    │
│  │                          x∈safe                                      │    │
│  │                                                                      │    │
│  │    Purpose: Push B(x) >= 0 for safe states                          │    │
│  │    ReLU(-B): Penalizes when B is negative for safe states           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TERM 2: Unsafe State Loss (Equation 7 from CBF theory)             │    │
│  │                                                                      │    │
│  │    L_unsafe = (1/N_unsafe) * Σ max(0, B(x))                         │    │
│  │                              x∈unsafe                                │    │
│  │                                                                      │    │
│  │    Purpose: Push B(x) < 0 for unsafe/collision states               │    │
│  │    ReLU(B): Penalizes when B is positive for unsafe states          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TERM 3: CBF Constraint Loss (Equation 8 from CBF theory)           │    │
│  │                                                                      │    │
│  │    L_constraint = (1/N) * Σ max(0, -(dB/dt + α * B))                │    │
│  │                                                                      │    │
│  │    where: dB/dt ≈ (B(x_{t+1}) - B(x_t)) / Δt                        │    │
│  │                                                                      │    │
│  │    Purpose: Enforce CBF Nagumo condition                            │    │
│  │    Ensures system can always maintain safety                         │    │
│  │                                                                      │    │
│  │    Note: x_{t+1} is generated synthetically via small perturbation  │    │
│  │          x_{t+1} = x_t + ε, where ε ~ N(0, perturbation_scale)      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CBF TRAINING DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DATA LOADING                                                            │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  COLLISION DATA: collision_100k_train.dat                      │      │
│     │  Format: [j1-j7, ee_xyz, obs_x, obs_y, obs_h, obs_r, label=1] │      │
│     │  Robot states IN collision with obstacle                       │      │
│     │                                                                 │      │
│     │  FREE-SPACE DATA: free_space_100k_train.dat                   │      │
│     │  (Augmented with random obstacle params, label=0)              │      │
│     │  Robot states NOT in collision                                 │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                     │                                        │
│                                     ▼                                        │
│  2. DATASET PREPARATION (RobotCBFDataset / RobotCBFDatasetSimple)           │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  Option A - Simple (single states):                            │      │
│     │    return (jpos_ee_xyz, obs_xyhr, label)                       │      │
│     │    Shapes: (10,), (4,), (1,)                                   │      │
│     │                                                                 │      │
│     │  Option B - Trajectory pairs (for CBF constraint):             │      │
│     │    x_current → x_next (synthetic perturbation)                 │      │
│     │    return (jpos, jpos_next, obs, label, label_next, dt)        │      │
│     │    Used for computing dB/dt                                    │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                     │                                        │
│                                     ▼                                        │
│  3. FORWARD PASS                                                            │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │                                                                 │      │
│     │  a. Encode robot state:                                        │      │
│     │     z, μ, logVar = encoder(x_norm)                             │      │
│     │                                                                 │      │
│     │  b. Compute barrier value:                                     │      │
│     │     z_obs = concat(z, obs)        # [B x 11]                   │      │
│     │     B = barrier_function(z, obs)  # [B x 1]                    │      │
│     │                                                                 │      │
│     │  c. (Optional) VAE reconstruction:                             │      │
│     │     x_recon = decoder(z)          # [B x 10]                   │      │
│     │                                                                 │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                     │                                        │
│                                     ▼                                        │
│  4. LOSS COMPUTATION                                                        │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │                                                                 │      │
│     │  a. Separate safe and unsafe samples:                          │      │
│     │     B_safe = B[label == 0]                                     │      │
│     │     B_unsafe = B[label == 1]                                   │      │
│     │                                                                 │      │
│     │  b. Compute CBF losses:                                        │      │
│     │     L_safe = mean(ReLU(-B_safe))                               │      │
│     │     L_unsafe = mean(ReLU(B_unsafe))                            │      │
│     │                                                                 │      │
│     │  c. (If using trajectory data) CBF constraint loss:            │      │
│     │     B_next = barrier_function(z_next, obs)                     │      │
│     │     dB_dt = (B_next - B_current) / dt                          │      │
│     │     L_constraint = mean(ReLU(-(dB_dt + alpha * B_current)))    │      │
│     │                                                                 │      │
│     │  d. VAE loss (if training jointly):                            │      │
│     │     L_VAE = GECO_loss(MSE, KL)                                 │      │
│     │                                                                 │      │
│     │  e. Total loss:                                                │      │
│     │     L = λ_vae * L_VAE + λ_cbf_total * (L_safe + L_unsafe       │      │
│     │                                       + λ_cbf * L_constraint)   │      │
│     │                                                                 │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                     │                                        │
│                                     ▼                                        │
│  5. BACKPROPAGATION & UPDATE                                                │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  optimizer.zero_grad()                                         │      │
│     │  L.backward()                                                  │      │
│     │  optimizer.step()                                              │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Input/Output Specification

| Component | Input | Output | Dimensions |
|-----------|-------|--------|------------|
| **CBF Dataset Simple** | .dat files | (state, obs, label) | (B x 10, B x 4, B x 1) |
| **CBF Dataset Trajectory** | .dat files | (state, state_next, obs, label, label_next, dt) | Complex |
| **Encoder** | Normalized state | (z, μ, logVar) | (B x 7, B x 7, B x 7) |
| **Barrier Function** | (z, obs) | B(x) | B x 1 |
| **CBF Loss** | (B, labels) | Scalar loss | 1 |

### Key Files

- **Model Definition**: `src/vae_cbf.py` (lines 21-185)
- **Training Script**: `src/train_vae_cbf.py`
- **Dataset Loader**: `src/robot_cbf_dataset.py`
- **Configuration**: `config/vae_cbf_config/panda_10k.yaml`

---

## Phase 3: CBF-based Planning Architecture

### Purpose

Use the learned CBF for safe motion planning by projecting nominal goal-seeking trajectories onto the safe manifold defined by B(x) >= 0.

### Planning Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CBF-BASED PLANNING ALGORITHM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALGORITHM: plan_with_cbf(q_start, e_target, obstacles)                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  INITIALIZATION                                                      │    │
│  │                                                                      │    │
│  │  1. Normalize input:                                                │    │
│  │       x_start = [q_start, e_start]                                  │    │
│  │       x_norm = (x_start - mean) / std                               │    │
│  │                                                                      │    │
│  │  2. Encode to latent space:                                         │    │
│  │       z = encoder(x_norm)[0]  # Take mean (no sampling)             │    │
│  │                                                                      │    │
│  │  3. Initialize path:                                                │    │
│  │       path = [(q_start, e_start)]                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MAIN PLANNING LOOP (for step = 1 to max_steps)                     │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  STEP 1: GOAL-SEEKING (Gradient Descent)                      │  │    │
│  │  │                                                                │  │    │
│  │  │    a. Decode current latent:                                  │  │    │
│  │  │         x_decoded = decoder(z)                                 │  │    │
│  │  │         e_decoded = x_decoded[7:10]  # End-effector position  │  │    │
│  │  │                                                                │  │    │
│  │  │    b. Compute goal loss:                                      │  │    │
│  │  │         L_goal = ||e_decoded - e_target||_2                   │  │    │
│  │  │                                                                │  │    │
│  │  │    c. (Optional) Prior regularization:                        │  │    │
│  │  │         L_prior = 0.5 * ||z||_2^2                             │  │    │
│  │  │                                                                │  │    │
│  │  │    d. Total loss and gradient:                                │  │    │
│  │  │         L = L_goal + λ_prior * L_prior                        │  │    │
│  │  │         L.backward()                                          │  │    │
│  │  │                                                                │  │    │
│  │  │    e. Compute nominal next state:                             │  │    │
│  │  │         z_nom = z - lr * ∇_z L                                │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                             │                                        │    │
│  │                             ▼                                        │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  STEP 2: CBF SAFE PROJECTION                                  │  │    │
│  │  │                                                                │  │    │
│  │  │    z_safe = latent_safe_update(z_nom, z_current, obstacles)   │  │    │
│  │  │                                                                │  │    │
│  │  │    (See detailed algorithm below)                             │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                             │                                        │    │
│  │                             ▼                                        │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  STEP 3: UPDATE AND RECORD                                    │  │    │
│  │  │                                                                │  │    │
│  │  │    a. Update latent state:                                    │  │    │
│  │  │         z = z_safe                                            │  │    │
│  │  │                                                                │  │    │
│  │  │    b. Decode to joint angles:                                 │  │    │
│  │  │         x_decoded = decoder(z)                                │  │    │
│  │  │         q = denormalize(x_decoded[0:7])                       │  │    │
│  │  │         e = denormalize(x_decoded[7:10])                      │  │    │
│  │  │                                                                │  │    │
│  │  │    c. Record waypoint:                                        │  │    │
│  │  │         path.append((q, e))                                   │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                             │                                        │    │
│  │                             ▼                                        │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  STEP 4: TERMINATION CHECK                                    │  │    │
│  │  │                                                                │  │    │
│  │  │    dist_to_goal = ||e - e_target||                            │  │    │
│  │  │                                                                │  │    │
│  │  │    if dist_to_goal < threshold:                               │  │    │
│  │  │        break  # Goal reached                                  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  OUTPUT: path = [(q_0, e_0), (q_1, e_1), ..., (q_n, e_n)]          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Latent Safe Update Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LATENT SAFE UPDATE ALGORITHM                            │
│                  (CBF Gradient-Based Projection)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FUNCTION: latent_safe_update(z_nom, z_current, model_cbf, obs, α, dt)      │
│                                                                              │
│  PURPOSE: Project nominal state z_nom onto safe manifold where B >= 0       │
│                                                                              │
│  FORMULA (from CBF theory):                                                 │
│                                                                              │
│       z_safe = z_nom + λ * ∇B(z_nom)                                        │
│                                                                              │
│       where: λ = [(1 - α*dt) * B(z_current) - B(z_nom)] / ||∇B||²          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ALGORITHM STEPS:                                                   │    │
│  │                                                                      │    │
│  │  1. Get current barrier value (no gradient):                        │    │
│  │       with torch.no_grad():                                         │    │
│  │           B_current = barrier_function(z_current, obs)              │    │
│  │                                                                      │    │
│  │  2. Compute barrier and gradient at nominal state:                  │    │
│  │       z_nom.requires_grad_(True)                                    │    │
│  │       B_nom = barrier_function(z_nom, obs)                          │    │
│  │       grad_B = autograd.grad(B_nom.sum(), z_nom)[0]                 │    │
│  │                                                                      │    │
│  │  3. Compute target barrier value:                                   │    │
│  │       B_target = (1 - α * dt) * B_current                           │    │
│  │                                                                      │    │
│  │       Note: This ensures B decreases at most rate α                 │    │
│  │             Derived from CBF condition: dB/dt ≥ -α * B              │    │
│  │                                                                      │    │
│  │  4. Compute correction step size:                                   │    │
│  │       grad_norm_sq = ||∇B||² + ε   (ε=1e-8 for numerical stability) │    │
│  │       λ = (B_target - B_nom) / grad_norm_sq                         │    │
│  │                                                                      │    │
│  │  5. Apply correction (only if needed):                              │    │
│  │       if λ > 0:                                                     │    │
│  │           # Constraint violated, project toward safety              │    │
│  │           z_safe = z_nom + λ * ∇B                                   │    │
│  │       else:                                                         │    │
│  │           # Already safe, keep nominal                              │    │
│  │           z_safe = z_nom                                            │    │
│  │                                                                      │    │
│  │  6. Return: (z_safe, λ, B_nom)                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  MULTI-OBSTACLE VERSION:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  For each obstacle in obstacles:                                    │    │
│  │      z_safe = latent_safe_update(z_safe, z_current, obs_i)          │    │
│  │                                                                      │    │
│  │  Or: Use most constraining obstacle (lowest B value)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TWO-STAGE PLANNING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 1: CBF-BASED PLANNING (Learned Model)                        │    │
│  │                                                                      │    │
│  │    Input:  (q_start, e_start, e_target, obstacles)                  │    │
│  │                                                                      │    │
│  │    Process:                                                         │    │
│  │      1. Gradient descent in latent space toward goal                │    │
│  │      2. CBF projection for each step to maintain safety             │    │
│  │      3. Decode latent trajectory to joint space                     │    │
│  │                                                                      │    │
│  │    Output: path = [(q_0, e_0), ..., (q_n, e_n)]                     │    │
│  │                                                                      │    │
│  │    Note: Uses LEARNED CBF, may have approximation errors            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 2: GEOMETRIC VALIDATION (Ground Truth)                       │    │
│  │                                                                      │    │
│  │    Input:  path from Stage 1, obstacles                             │    │
│  │                                                                      │    │
│  │    Process:                                                         │    │
│  │      For each waypoint (q, e) in path:                              │    │
│  │        1. Forward kinematics: link_positions = FK(q)                │    │
│  │        2. Collision check: capsule vs cylinder intersection         │    │
│  │        3. Record collision status                                   │    │
│  │                                                                      │    │
│  │    Output:                                                          │    │
│  │      - is_collision_free: bool                                      │    │
│  │      - num_collisions: int                                          │    │
│  │      - collision_ratio: float                                       │    │
│  │                                                                      │    │
│  │    Note: Uses EXACT GEOMETRY, ground truth for evaluation           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FINAL RESULT                                                        │    │
│  │                                                                      │    │
│  │    success = goal_reached AND collision_free                        │    │
│  │                                                                      │    │
│  │    Metrics:                                                         │    │
│  │      - Success rate: % of scenarios with success                    │    │
│  │      - Goal reach rate: % reaching goal (ignoring collisions)       │    │
│  │      - Collision-free rate: % with no collisions                    │    │
│  │      - Path length: Number of waypoints                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Input/Output Specification

| Component | Input | Output |
|-----------|-------|--------|
| **plan_with_cbf()** | (q_start, e_start, e_target, obstacles) | path: List[(q, e)] |
| **latent_safe_update()** | (z_nom, z_current, model_cbf, obs) | (z_safe, lambda, B) |
| **validate_path()** | (path, obstacles) | (is_collision_free, num_collisions) |
| **ObstacleScenarioGenerator** | (num_scenarios, config) | List[Scenario] |

### Key Files

- **Planning Script**: `src/evaluate_planning_cbf.py`
- **Collision Checker**: `src/sim/robot3d.py`
- **Robot Kinematics**: `src/sim/panda.py`
- **Run Script**: `run_evaluation_cbf.sh`

---

## End-to-End Data Flow

### Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      END-TO-END SYSTEM DATA FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ═══════════════════                                │
│                           ║  TRAINING PHASE ║                                │
│                           ═══════════════════                                │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                      RAW DATA FILES                                 │     │
│  │                                                                     │     │
│  │  free_space_100k_train.dat    collision_100k_train.dat             │     │
│  │  ┌─────────────────────┐      ┌─────────────────────────┐          │     │
│  │  │ j1-j7 (7 joints)    │      │ j1-j7 (7 joints)        │          │     │
│  │  │ ee_x, ee_y, ee_z    │      │ ee_x, ee_y, ee_z        │          │     │
│  │  │ goal_x, goal_y, z   │      │ obs_x, obs_y, obs_h, r  │          │     │
│  │  │                     │      │ collision_label (0/1)   │          │     │
│  │  └─────────────────────┘      └─────────────────────────┘          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                    │                                │                        │
│                    ▼                                ▼                        │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐     │
│  │     PHASE 1: VAE TRAINING   │    │     PHASE 2: CBF TRAINING       │     │
│  │                             │    │                                  │     │
│  │  train_vae.py               │    │  train_vae_cbf.py                │     │
│  │                             │    │                                  │     │
│  │  Input: free_space data     │    │  Input: collision + free data   │     │
│  │  Output: VAE checkpoint     │───▶│  Input: Pretrained VAE          │     │
│  │                             │    │  Output: VAECBF checkpoint       │     │
│  │  Loss: GECO(MSE, KL)        │    │  Loss: L_safe + L_unsafe + L_cbf│     │
│  └─────────────────────────────┘    └─────────────────────────────────┘     │
│                                                      │                       │
│                                                      ▼                       │
│                           ═══════════════════════════════                    │
│                           ║   INFERENCE/PLANNING PHASE  ║                    │
│                           ═══════════════════════════════                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     SCENARIO GENERATION                             │     │
│  │                                                                     │     │
│  │  ObstacleScenarioGenerator:                                        │     │
│  │    - Random start configuration q_start                            │     │
│  │    - Random goal position e_target                                 │     │
│  │    - Random obstacles (x, y, h, r)                                 │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        CBF PLANNING                                 │     │
│  │                                                                     │     │
│  │  ┌──────────────────────────────────────────────────────────────┐  │     │
│  │  │  FOR EACH PLANNING STEP:                                     │  │     │
│  │  │                                                               │  │     │
│  │  │  1. ENCODE: x → z                                            │  │     │
│  │  │       z = encoder(normalize(x))                              │  │     │
│  │  │                                                               │  │     │
│  │  │  2. GOAL GRADIENT: z_nom = z - lr * ∇L_goal                  │  │     │
│  │  │       L_goal = ||decoder(z)[7:10] - e_target||               │  │     │
│  │  │                                                               │  │     │
│  │  │  3. CBF PROJECTION: z_safe = project(z_nom)                  │  │     │
│  │  │       If B(z_nom) < threshold:                               │  │     │
│  │  │         z_safe = z_nom + λ * ∇B(z_nom)                       │  │     │
│  │  │                                                               │  │     │
│  │  │  4. DECODE: z_safe → (q, e)                                  │  │     │
│  │  │       x = denormalize(decoder(z_safe))                       │  │     │
│  │  │       q = x[0:7], e = x[7:10]                                │  │     │
│  │  └──────────────────────────────────────────────────────────────┘  │     │
│  │                                                                     │     │
│  │  Output: path = [(q_0, e_0), (q_1, e_1), ..., (q_n, e_n)]         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    GEOMETRIC VALIDATION                             │     │
│  │                                                                     │     │
│  │  For each waypoint in path:                                        │     │
│  │    1. Forward Kinematics: q → link_positions                       │     │
│  │    2. Build robot capsules from link positions                     │     │
│  │    3. Build obstacle cylinders from obs params                     │     │
│  │    4. Check capsule-cylinder intersections                         │     │
│  │    5. Record: collision or no collision                            │     │
│  │                                                                     │     │
│  │  Output: (is_collision_free, collision_count)                      │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                      FINAL METRICS                                  │     │
│  │                                                                     │     │
│  │  Success Rate = (goal_reached AND collision_free) / total_scenarios│     │
│  │  Goal Reach Rate = goal_reached / total_scenarios                  │     │
│  │  Collision-Free Rate = collision_free / total_scenarios            │     │
│  │  Average Path Length = mean(len(path))                             │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Transformations Summary

| Stage | Input Format | Transformation | Output Format |
|-------|--------------|----------------|---------------|
| Raw Data | .dat file (13 cols) | Load + split | numpy array |
| Dataset | numpy array | Normalize, batch | PyTorch tensor |
| Encoder | Tensor [B x 10] | FC layers + reparam | z [B x 7], μ [B x 7], σ [B x 7] |
| Decoder | z [B x 7] | FC layers | x' [B x 10] |
| CBF Head | [z, obs] [B x 11] | FC layers | B [B x 1] |
| Planning | z [7] | Iterative update | path [(10,), ...] |
| Validation | path + obstacles | FK + collision | (bool, int) |

---

## Implementation Algorithm

### Complete Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   IMPLEMENTATION ALGORITHM - ALL PHASES                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 1: VAE PRE-TRAINING                                            ║  │
│  ║  Command: ./train_models.sh (Step 1)                                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│    Algorithm:                                                               │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  1. Load free_space_100k_train.dat                                  │  │
│    │  2. Compute normalization statistics (mean, std)                    │  │
│    │  3. Initialize VAE model with random weights                        │  │
│    │  4. FOR epoch = 1 to 16000:                                         │  │
│    │       FOR each batch in dataloader:                                 │  │
│    │         a. x_norm = normalize(batch)                                │  │
│    │         b. z, μ, logVar = encoder(x_norm)                           │  │
│    │         c. x_recon = decoder(z)                                     │  │
│    │         d. L_recon = MSE(x_recon, x_norm)                           │  │
│    │         e. L_KL = KL_divergence(μ, logVar)                          │  │
│    │         f. L = GECO_balance(L_recon, L_KL)                          │  │
│    │         g. optimizer.step(L)                                        │  │
│    │       END FOR                                                       │  │
│    │       IF epoch % 1000 == 0: save_checkpoint()                       │  │
│    │     END FOR                                                         │  │
│    │  5. Save final model: model.ckpt-16000.pt                           │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│    Outputs:                                                                 │
│      - Trained encoder: maps robot states to 7D latent space               │
│      - Trained decoder: reconstructs robot states from latent codes        │
│      - Normalization parameters: mean, std for input preprocessing         │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 2: CBF TRAINING                                                ║  │
│  ║  Command: ./train_cbf.sh                                              ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│    Algorithm:                                                               │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  1. Load pretrained VAE checkpoint                                  │  │
│    │  2. Load collision_100k_train.dat + free_space_100k_train.dat       │  │
│    │  3. Initialize CBF head with random weights                         │  │
│    │  4. FOR epoch = 1 to 16000:                                         │  │
│    │       FOR each batch in dataloader:                                 │  │
│    │         a. Separate safe (label=0) and unsafe (label=1) samples     │  │
│    │         b. z = encoder(x_norm)                                      │  │
│    │         c. B = barrier_function(z, obs)                             │  │
│    │                                                                     │  │
│    │         # Safe state loss                                          │  │
│    │         d. B_safe = B[label == 0]                                   │  │
│    │         e. L_safe = mean(ReLU(-B_safe))                             │  │
│    │                                                                     │  │
│    │         # Unsafe state loss                                        │  │
│    │         f. B_unsafe = B[label == 1]                                 │  │
│    │         g. L_unsafe = mean(ReLU(B_unsafe))                          │  │
│    │                                                                     │  │
│    │         # CBF constraint loss (optional)                           │  │
│    │         h. Generate x_next via perturbation                        │  │
│    │         i. z_next = encoder(x_next)                                 │  │
│    │         j. B_next = barrier_function(z_next, obs)                   │  │
│    │         k. dB_dt = (B_next - B) / dt                                │  │
│    │         l. L_constraint = mean(ReLU(-(dB_dt + α*B)))                │  │
│    │                                                                     │  │
│    │         # Total loss                                               │  │
│    │         m. L_CBF = L_safe + L_unsafe + λ_cbf * L_constraint         │  │
│    │         n. L_VAE = GECO_loss(MSE, KL) # optional joint training     │  │
│    │         o. L = λ_vae * L_VAE + λ_cbf_total * L_CBF                  │  │
│    │         p. optimizer.step(L)                                        │  │
│    │       END FOR                                                       │  │
│    │       IF epoch % 1000 == 0: save_checkpoint(), eval_accuracy()      │  │
│    │     END FOR                                                         │  │
│    │  5. Save final model: vaecbf.ckpt-16000.pt                          │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│    Outputs:                                                                 │
│      - Trained CBF head: predicts B(x, obs) for any robot-obstacle pair    │
│      - Safety classification: B >= 0 means safe, B < 0 means collision     │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 3: CBF-BASED PLANNING                                          ║  │
│  ║  Command: ./run_evaluation_cbf.sh                                     ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│    Algorithm:                                                               │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  1. Load trained VAECBF model                                       │  │
│    │  2. Generate test scenarios: (q_start, e_target, obstacles)         │  │
│    │  3. FOR each scenario:                                              │  │
│    │       a. Initialize: z = encoder(q_start, e_start)                  │  │
│    │       b. path = [(q_start, e_start)]                                │  │
│    │                                                                     │  │
│    │       FOR step = 1 to max_steps:                                    │  │
│    │         # Goal-seeking                                              │  │
│    │         c. x_decoded = decoder(z)                                   │  │
│    │         d. L_goal = ||x_decoded[7:10] - e_target||                  │  │
│    │         e. z_nom = z - lr * ∇L_goal                                 │  │
│    │                                                                     │  │
│    │         # CBF safe projection                                      │  │
│    │         f. B_current = barrier_function(z, obs)                     │  │
│    │         g. B_nom, ∇B = barrier_with_grad(z_nom, obs)               │  │
│    │         h. B_target = (1 - α*dt) * B_current                        │  │
│    │         i. λ = (B_target - B_nom) / ||∇B||²                         │  │
│    │         j. IF λ > 0: z_safe = z_nom + λ * ∇B                        │  │
│    │            ELSE: z_safe = z_nom                                     │  │
│    │                                                                     │  │
│    │         # Update                                                   │  │
│    │         k. z = z_safe                                               │  │
│    │         l. (q, e) = decode_and_denormalize(z)                       │  │
│    │         m. path.append((q, e))                                      │  │
│    │                                                                     │  │
│    │         # Check termination                                        │  │
│    │         n. IF ||e - e_target|| < threshold: BREAK                   │  │
│    │       END FOR                                                       │  │
│    │                                                                     │  │
│    │       # Geometric validation                                       │  │
│    │       o. collision_free = validate_path(path, obstacles)            │  │
│    │       p. goal_reached = ||path[-1].e - e_target|| < threshold       │  │
│    │       q. success = collision_free AND goal_reached                  │  │
│    │     END FOR                                                         │  │
│    │                                                                     │  │
│    │  4. Compute metrics: success_rate, collision_rate, etc.             │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│    Outputs:                                                                 │
│      - Planned paths for each scenario                                     │
│      - Success/failure metrics                                             │
│      - Collision statistics                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Parameters

### VAE Configuration (`config/vae_config/panda_10k.yaml`)

```yaml
# ═══════════════════════════════════════════════════════════════════════════
#                           VAE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Model Architecture
input_dim         : 10      # 7 joint angles + 3D end-effector position
latent_dim        : 7       # Dimensionality of latent space
units_per_layer   : 2048    # Hidden layer width
num_hidden_layers : 4       # Number of hidden layers in encoder/decoder
activation        : elu     # Activation function

# Training Parameters
lr_vae            : 0.0001  # Learning rate
batch_size        : 4096    # Batch size
epochs_vae        : 16000   # Total training epochs
seed              : 1       # Random seed for reproducibility
log_interval      : 1000    # Checkpoint saving interval

# GECO Parameters (Constrained Optimization)
g_goal            : 0.0008  # Target reconstruction error
g_lr              : 0.005   # Learning rate for GECO multiplier
g_alpha           : 0.95    # Exponential moving average for constraint

# Evaluation Parameters
samples           : 10      # Number of evaluation samples
am_lr             : 0.03    # Amortized inference learning rate
am_steps          : 300     # Amortized inference steps

# Device
device            : cuda    # Training device (cuda/cpu)
```

### CBF Configuration (`config/vae_cbf_config/panda_10k.yaml`)

```yaml
# ═══════════════════════════════════════════════════════════════════════════
#                           CBF CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Model Architecture (same as VAE)
input_dim         : 10
latent_dim        : 7
units_per_layer   : 2048
num_hidden_layers : 4

# Training Parameters
lr                : 0.0001  # Learning rate
batch_size        : 4096    # Batch size
epochs            : 16000   # Total training epochs
seed              : 1

# CBF-Specific Parameters
alpha_cbf         : 1.0     # CBF decay rate (α in dB/dt ≥ -αB)
lambda_vae        : 1.0     # Weight for VAE reconstruction loss
lambda_cbf_total  : 1.0     # Weight for total CBF loss
lambda_cbf        : 0.1     # Weight for CBF constraint term

# Trajectory Data Parameters
use_trajectory_data: false  # Whether to use trajectory pairs for dB/dt
dt                : 0.1     # Time step for finite difference approximation
perturbation_scale: 0.01    # Scale for synthetic next-state generation

# GECO Parameters (for VAE component)
geco_goal         : 0.1     # Target constraint value
geco_lr           : 0.00001 # GECO multiplier learning rate
geco_speedup      : 10.0    # Speedup factor for GECO
geco_alpha        : 0.99    # EMA coefficient

# Device
device            : cuda
```

### Planning Configuration (in `evaluate_planning_cbf.py`)

```python
# Planning Parameters
planning_lr       = 0.03    # Learning rate for latent gradient descent
max_steps         = 300     # Maximum planning steps
goal_threshold    = 0.05    # Distance threshold for goal reached
lambda_prior      = 0.0001  # Weight for latent space regularization
alpha             = 1.0     # CBF decay rate during planning
dt                = 0.1     # Time step for CBF update rule
```

---

## Key Classes and Functions Reference

### VAE Component

| File | Class/Function | Lines | Description |
|------|----------------|-------|-------------|
| `vae.py` | `VAE` | 8-79 | Main VAE class with encoder/decoder |
| `vae.py` | `VAE.encoder()` | 42-53 | Encode x → (z, μ, logVar) |
| `vae.py` | `VAE.decoder()` | 55-63 | Decode z → x' |
| `vae.py` | `VAE.reparametrize()` | 38-40 | Reparameterization trick |
| `train_vae.py` | `loss_function()` | 108-122 | VAE loss with GECO |
| `train_vae.py` | `train()` | - | Training epoch |
| `robot_state_dataset.py` | `RobotStateDataset` | - | Dataset loader |
| `geco.py` | `GECO` | - | Adaptive loss balancing |

### CBF Component

| File | Class/Function | Lines | Description |
|------|----------------|-------|-------------|
| `vae_cbf.py` | `VAECBF` | 21-185 | VAE with CBF head |
| `vae_cbf.py` | `VAECBF.barrier_function()` | - | Compute B(z, obs) |
| `vae_cbf.py` | `VAECBF.get_barrier_with_grad()` | - | B(x) with gradient |
| `train_vae_cbf.py` | `loss_function_cbf_simple()` | 196-220 | L_safe + L_unsafe |
| `train_vae_cbf.py` | `loss_function_cbf_full()` | 222-263 | Full CBF loss |
| `train_vae_cbf.py` | `eval_cbf_accuracy()` | - | Classification accuracy |
| `robot_cbf_dataset.py` | `RobotCBFDataset` | 31-189 | Trajectory pairs dataset |
| `robot_cbf_dataset.py` | `RobotCBFDatasetSimple` | 192-299 | Single states dataset |

### Planning Component

| File | Class/Function | Lines | Description |
|------|----------------|-------|-------------|
| `evaluate_planning_cbf.py` | `latent_safe_update()` | 142-189 | Single-obstacle CBF projection |
| `evaluate_planning_cbf.py` | `latent_safe_update_multi_obstacle()` | - | Multi-obstacle version |
| `evaluate_planning_cbf.py` | `plan_with_cbf()` | 223-361 | Main planning function |
| `evaluate_planning_cbf.py` | `validate_path_with_geometric_checker()` | - | Ground truth validation |
| `evaluate_planning_cbf.py` | `ObstacleScenarioGenerator` | - | Test scenario generation |

### Robot Simulation Component

| File | Class/Function | Description |
|------|----------------|-------------|
| `sim/panda.py` | `Panda` | Panda robot with DH parameters |
| `sim/panda.py` | `Panda.FK()` | Forward kinematics |
| `sim/robot3d.py` | `Robo3D` | Robot collision model with capsules |
| `sim/robot3d.py` | `Robo3D.check_for_collision()` | Geometric collision check |
| `sim/robot3d.py` | `RoboCapsule` | Capsule collision primitive |
| `sim/geometry.py` | `dist3d_capsule_to_capsule()` | Capsule-capsule distance |

---

## Summary

This CBF architecture provides a principled approach to safe robot motion planning:

1. **VAE** learns a compact latent representation of valid robot configurations
2. **CBF Head** learns to predict safety (B >= 0) vs collision (B < 0) states
3. **Planning** uses gradient-based projection to maintain safety while reaching goals
4. **Validation** uses exact geometric collision checking for ground truth evaluation

The key advantage over binary classifiers is the **continuous barrier function** that provides:
- Formal safety guarantees through the CBF constraint dB/dt >= -α*B
- Smooth safety margins for gradient-based optimization
- Principled projection onto the safe manifold during planning

---

## Quick Start Commands

```bash
# Phase 1: Train VAE
cd /home/rck/RTLS/5.RTLS_CBFS
./train_models.sh  # Step 1 only

# Phase 2: Train CBF
./train_cbf.sh

# Phase 3: Evaluate CBF Planning
./run_evaluation_cbf.sh
```

---

*Documentation generated for RTLS-CBF project. For detailed implementation, refer to `CBF.md` and source files.*
