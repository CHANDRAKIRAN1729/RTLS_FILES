# Control Barrier Functions (CBF) Implementation

This document describes the implementation of Control Barrier Functions (CBFs) for safe robot motion planning, replacing the original binary collision classifier in the RTLS system.

---

## Table of Contents

1. [Overview](#overview)
2. [Background: Original Collision Classifier](#background-original-collision-classifier)
3. [Control Barrier Functions: Theory](#control-barrier-functions-theory)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Implementation Changes](#implementation-changes)
6. [File Descriptions](#file-descriptions)
7. [Training Pipeline](#training-pipeline)
8. [Planning with CBF](#planning-with-cbf)
9. [Usage Instructions](#usage-instructions)
10. [Comparison: Original vs CBF](#comparison-original-vs-cbf)

---

## Overview

The original RTLS (Reaching Through Latent Space) system uses a **binary collision classifier** to predict whether a robot configuration collides with obstacles. This classifier outputs a probability P(collision) ∈ [0, 1] and is trained using binary cross-entropy loss.

This implementation replaces the collision classifier with a **Control Barrier Function (CBF)**, which provides:

1. **Formal safety guarantees** through mathematical constraints
2. **Smooth safety margins** instead of binary classification
3. **Gradient-based safety projection** during planning

The CBF outputs a barrier value B(x) ∈ ℝ where:
- **B(x) ≥ 0**: Safe state (no collision)
- **B(x) < 0**: Unsafe state (collision)

---

## Background: Original Collision Classifier

### Architecture

The original system consists of:

```
Input: [j1, j2, ..., j7, ee_x, ee_y, ee_z]  (10 dimensions)
                    ↓
         ┌─────────────────────┐
         │    VAE Encoder      │
         │  10 → 2048 → ... → 7│
         └─────────────────────┘
                    ↓
              Latent z (dim 7)
                    ↓
         ┌─────────────────────┐
         │ Collision Classifier│
         │ [z, obs] → sigmoid  │
         │     → P(collision)  │
         └─────────────────────┘
```

### Loss Function

The original classifier uses **Binary Cross-Entropy (BCE)**:

```
L_BCE = -[y·log(p) + (1-y)·log(1-p)]
```

Where:
- `y ∈ {0, 1}` is the ground truth label (0=safe, 1=collision)
- `p = sigmoid(logit)` is the predicted collision probability

### Limitations

1. **No safety guarantees**: The classifier provides probabilities, not hard constraints
2. **Discrete output**: Binary classification doesn't capture safety margins
3. **No temporal consistency**: Each prediction is independent; no guarantee that safe trajectories remain safe

---

## Control Barrier Functions: Theory

### Definition

A Control Barrier Function B(x) is a continuously differentiable function that satisfies:

1. **B(x) ≥ 0** for all x in the safe set Xs
2. **B(x) < 0** for all x in the unsafe set Xu
3. **dB/dt ≥ -α·B(x)** for all x (forward invariance condition)

The third condition is crucial: it ensures that if the system starts in a safe state (B ≥ 0), it will **remain safe for all future time**.

### Intuition

Think of B(x) as a "safety margin":
- Large positive B → far from obstacles (very safe)
- Small positive B → close to obstacles (marginally safe)
- Zero B → at the safety boundary
- Negative B → in collision (unsafe)

The constraint dB/dt ≥ -α·B controls how quickly the safety margin can decrease:
- If B is large, dB/dt can be more negative (faster approach allowed)
- If B is small, dB/dt must be less negative (slower approach required)
- At B = 0, dB/dt ≥ 0 (cannot enter unsafe region)

---

## Mathematical Formulation

### CBF Conditions (CBF1.pdf, Equations 1-3)

```
B(x) ≥ 0    ∀x ∈ Xs (safe states)           ... (1)
B(x) < 0    ∀x ∈ Xu (unsafe states)         ... (2)
Ḃ(x) ≥ -α·B(x)    ∀x ∈ X (CBF constraint)   ... (3)
```

### CBF Loss Function (CBF1.pdf, Equations 6-8)

The neural network CBF is trained using a three-term loss:

```
L = L_safe + L_unsafe + L_cbf
```

**Term 1 - Safe State Loss (Equation 6):**
```
L_safe = Σ max(-B(x), 0)    for x ∈ Xs
```
Penalizes when B(x) < 0 for states that should be safe.

**Term 2 - Unsafe State Loss (Equation 7):**
```
L_unsafe = Σ max(B(x), 0)    for x ∈ Xu
```
Penalizes when B(x) ≥ 0 for states that should be unsafe.

**Term 3 - CBF Constraint Loss (Equation 8):**
```
L_cbf = Σ max(-(Ḃ(x) + α·B(x)), 0)
```
Penalizes when the CBF constraint Ḃ ≥ -α·B is violated.

### Time Derivative Approximation (CBF1.pdf, Equations 11-14)

The time derivative of B is computed using the chain rule:

```
dB/dt = ∂B/∂x · dx/dt = ∇ₓB · ẋ                ... (11-12)
```

For discrete-time implementation (Equation 14):

```
∂B/∂t ≈ (B(x_{t+1}) - B(x_t)) / Δt            ... (14)
```

### Discretized CBF Constraint (CBF1.pdf, Equations 15-18)

Starting from the continuous constraint Ḃ ≥ -α·B:

```
(B(z_{t+1}) - B(z_t)) / Δt ≥ -α·B(z_t)        ... (15)

B(z_{t+1}) - B(z_t) ≥ -α·Δt·B(z_t)            ... (16)

B(z_{t+1}) ≥ B(z_t) - α·Δt·B(z_t)             ... (17)

B(z_{t+1}) ≥ (1 - α·Δt)·B(z_t)                ... (18)
```

This shows that the barrier value at the next state must be at least `(1 - α·Δt)` times the current barrier value.

---

## Latent Safe Update (CBF2.pdf)

### Problem Statement

Given:
- Current state: z_k
- Nominal next state: z^nom_{k+1} (from gradient-based planner)

Find the **minimum correction** to z^nom that satisfies the CBF constraint.

### Solution Structure (Equation A)

```
z^safe_{k+1} = z^nom_{k+1} + λ·d
```

Where:
- `d` is the correction direction
- `λ` is the step size (scalar)

### Safety Constraint (Equation 1)

The corrected state must satisfy:

```
B(z^safe_{k+1}) = (1 - α·Δt)·B(z_k)
```

### Optimization Formulation (Equation 2)

```
z^safe = arg min_z ||z - z^nom||²
         subject to: B(z^safe_{k+1}) = (1 - α·Δt)·B(z_k)
```

This is a **minimum-norm projection** onto the safe manifold.

### Optimal Direction (Equation 3)

The optimal correction direction is the **gradient of the barrier function**:

```
d = ∇B(z^nom)
```

This is because the gradient points in the direction of steepest increase in safety margin.

### Linear Approximation (Equations 4-8)

Using first-order Taylor expansion:

```
B(z^safe_{k+1}) = B(z^nom_{k+1} + λ·d)
                ≈ B(z^nom_{k+1}) + ∇B(z^nom_{k+1})ᵀ · λ·d     ... (5-6)
```

Since d = ∇B(z^nom):

```
∇B(z^nom)ᵀ · λ · ∇B(z^nom) = λ · ||∇B(z^nom)||²              ... (7)
```

Therefore:

```
B(z^safe_{k+1}) ≈ B(z^nom_{k+1}) + λ · ||d||²                ... (8)
```

### Closed-Form Solution for λ (Equation 9)

Setting the approximation equal to the safety target:

```
B(z^nom) + λ · ||d||² = (1 - α·Δt) · B(z_k)
```

Solving for λ:

```
λ = [(1 - α·Δt) · B(z_k) - B(z^nom)] / ||d||²                ... (9)
```

### Non-Negativity Constraint (Equation 10)

```
λ ≥ 0                                                         ... (10)
```

- If λ > 0: Nominal state needs correction (pushed toward safety)
- If λ ≤ 0: Nominal state is already safe enough (no correction needed)

### Complete Algorithm

```python
def latent_safe_update(z_nom, z_current, B_func, alpha, dt):
    # Step 1: Compute gradient
    d = ∇B(z_nom)

    # Step 2: Compute target barrier value
    B_target = (1 - alpha * dt) * B(z_current)

    # Step 3: Compute step size
    B_nom = B(z_nom)
    lambda_step = (B_target - B_nom) / (||d||² + ε)

    # Step 4: Apply correction if needed
    if lambda_step > 0:
        z_safe = z_nom + lambda_step * d
    else:
        z_safe = z_nom  # Already safe

    return z_safe
```

---

## Implementation Changes

### Summary of Changes

| Component | Original | CBF |
|-----------|----------|-----|
| **Output** | P(collision) ∈ [0, 1] | B(x) ∈ ℝ |
| **Output layer** | Linear → Sigmoid | Linear (unbounded) |
| **Loss function** | Binary Cross-Entropy | 3-term CBF loss |
| **Label semantics** | 0=safe, 1=collision | B≥0=safe, B<0=collision |
| **Planning** | Gradient descent + penalty | Gradient descent + safe projection |
| **Safety guarantee** | Soft (probability-based) | Hard (constraint-based) |

### Model Architecture Changes

**Original (`vae_obs.py`):**
```python
def obstacle_collision_classifier(self, z, obs):
    h = self.fc32(torch.cat((z, obs), dim=1))
    for fc in self.fc_obs:
        h = fc(F.elu(h))
    return self.fc42(F.elu(h)).view(-1)  # logit → sigmoid later
```

**CBF (`vae_cbf.py`):**
```python
def barrier_function(self, z, obs):
    h = self.fc_cbf1(torch.cat((z, obs), dim=1))
    for fc in self.fc_cbf:
        h = F.elu(fc(h))
    return self.fc_cbf_out(F.elu(h)).view(-1)  # Raw B(x), no sigmoid
```

### Loss Function Changes

**Original (`train_vae_obs.py`):**
```python
def loss_function_obs(obs_logit, obs_label):
    return F.binary_cross_entropy_with_logits(obs_logit, obs_label)
```

**CBF (`train_vae_cbf.py`):**
```python
def loss_function_cbf(B_current, B_next, labels, alpha, dt):
    safe_mask = (labels == 0)
    unsafe_mask = (labels == 1)

    # Term 1: B(x) ≥ 0 for safe states
    L_safe = torch.mean(F.relu(-B_current[safe_mask]))

    # Term 2: B(x) < 0 for unsafe states
    L_unsafe = torch.mean(F.relu(B_current[unsafe_mask]))

    # Term 3: Ḃ ≥ -α·B (CBF constraint)
    B_dot = (B_next - B_current) / dt
    L_cbf = torch.mean(F.relu(-(B_dot + alpha * B_current)))

    return L_safe + L_unsafe + lambda_cbf * L_cbf
```

### Planning Changes

**Original (`evaluate_planning.py`):**
```python
# Collision loss using learned classifier
logit = model_obs.obstacle_collision_classifier(z, obs)
p_collision = torch.sigmoid(logit / temperature)
L_collision = -torch.log(1 - p_collision + 1e-8)

# Gradient descent
L_total = L_goal + lambda_prior * L_prior + lambda_collision * L_collision
L_total.backward()
optimizer.step()
```

**CBF (`evaluate_planning_cbf.py`):**
```python
# Step 1: Goal-seeking gradient descent
L = L_goal + lambda_prior * L_prior
L.backward()
z_nom = z - lr * z.grad  # Nominal next state

# Step 2: CBF safe projection
d = torch.autograd.grad(B(z_nom), z_nom)[0]  # ∇B
B_target = (1 - alpha * dt) * B(z_current)
lambda_step = (B_target - B(z_nom)) / (||d||² + ε)

if lambda_step > 0:
    z_safe = z_nom + lambda_step * d
else:
    z_safe = z_nom
```

---

## File Descriptions

### New Files Created

#### `src/vae_cbf.py`
**Purpose:** Neural network model with CBF head

**Key Class: `VAECBF`**

```python
class VAECBF(nn.Module):
    """
    VAE with Control Barrier Function head.

    Architecture:
        - Encoder: x → z (same as original)
        - Decoder: z → x' (same as original)
        - CBF head: (z, obs) → B(x) ∈ ℝ (NEW)

    Output semantics:
        B(x) ≥ 0: safe
        B(x) < 0: unsafe
    """
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `encoder(x)` | Maps robot state to latent code z |
| `decoder(z)` | Reconstructs robot state from z |
| `barrier_function(z, obs)` | Computes B(z, obs) ∈ ℝ |
| `get_barrier_with_grad(z, obs)` | B(x) with gradient for planning |

---

#### `src/robot_cbf_dataset.py`
**Purpose:** Dataset providing consecutive state pairs for Ḃ computation

**Key Classes:**

**`RobotCBFDataset`** - Full dataset with trajectory pairs:
```python
def __getitem__(self, index):
    return (
        jpos_ee_xyz,       # Current state
        jpos_ee_xyz_next,  # Next state (for Ḃ)
        obs_xyhr,          # Obstacle parameters
        obs_label,         # Current label
        obs_label_next,    # Next label
        dt                 # Time step
    )
```

**`RobotCBFDatasetSimple`** - Simplified without trajectory:
```python
def __getitem__(self, index):
    return (
        jpos_ee_xyz,   # Robot state
        obs_xyhr,      # Obstacle parameters
        obs_label      # Collision label
    )
```

---

#### `src/train_vae_cbf.py`
**Purpose:** End-to-end training of VAE + CBF

**Key Functions:**

```python
def loss_function_vae(xPrime, x, mu, logVar):
    """VAE reconstruction + KL divergence loss"""

def loss_function_cbf_simple(B, labels):
    """CBF loss without trajectory (Eq. 6-7 only)"""

def loss_function_cbf_full(B_current, B_next, labels, dt, alpha):
    """Full CBF loss with trajectory constraint (Eq. 6-8)"""

def eval_cbf_accuracy(B, labels):
    """Evaluate: B≥0 for safe, B<0 for unsafe"""
```

**Training Loop:**
```
For each epoch:
    For each batch:
        1. Compute VAE loss on free-space data (reconstruction)
        2. Compute CBF loss on collision data (barrier function)
        3. Combined loss = λ_vae * L_vae + λ_cbf * L_cbf
        4. Backpropagate and update all parameters
```

---

#### `src/evaluate_planning_cbf.py`
**Purpose:** CBF-based path planning with latent safe update

**Key Functions:**

```python
def latent_safe_update(z_nom, z_current, model_cbf, obs, alpha, dt, device):
    """
    Project nominal state onto safe manifold.

    Implements CBF2.pdf:
        z_safe = z_nom + λ * ∇B(z_nom)
        λ = [(1 - α·Δt) * B(z_k) - B(z_nom)] / ||∇B||²
    """

def latent_safe_update_multi_obstacle(z_nom, z_current, model_cbf, obs_list, ...):
    """Apply safe update for multiple obstacles (most conservative)"""

def plan_with_cbf(model, model_cbf, q_start, e_start, e_target, obstacles, ...):
    """
    CBF-based path planning.

    At each step:
        1. Gradient descent toward goal → z_nom
        2. CBF safe projection → z_safe
    """
```

---

#### `config/vae_cbf_config/panda_10k.yaml`
**Purpose:** Training configuration for CBF

**Key Parameters:**

```yaml
# Model architecture
input_dim         : 10      # 7 joints + 3 end-effector
latent_dim        : 7
units_per_layer   : 2048
num_hidden_layers : 4

# CBF parameters
alpha_cbf         : 1.0     # CBF decay rate
lambda_vae        : 1.0     # VAE loss weight
lambda_cbf_total  : 1.0     # Total CBF loss weight
lambda_cbf        : 0.1     # CBF constraint term weight
dt                : 0.1     # Time step for Ḃ

# Training
lr                : 0.0001
batch_size        : 4096
epochs            : 16000
```

---

#### `train_cbf.sh`
**Purpose:** Shell script to launch CBF training

```bash
#!/bin/bash
cd src/
python train_vae_cbf.py --c ../config/vae_cbf_config/panda_10k.yaml
```

---

#### `run_evaluation_cbf.sh`
**Purpose:** Shell script to evaluate CBF planning

```bash
#!/bin/bash
python evaluate_planning_cbf.py \
    --checkpoint $VAE_CKPT \
    --checkpoint_cbf $CBF_CKPT \
    --config $CONFIG \
    --num_problems 100 \
    --alpha_cbf 1.0 \
    --dt_cbf 0.1
```

---

## Training Pipeline

### End-to-End Training

The CBF is trained jointly with the VAE:

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    Free-space data ──→ VAE ──→ L_vae (reconstruction)       │
│                          │                                   │
│                          ↓                                   │
│    Collision data ──→ [Encoder → CBF head] ──→ L_cbf        │
│                                                              │
│    Combined: L = λ_vae·L_vae + λ_cbf·L_cbf                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Loss Components

1. **VAE Loss** (ensures good latent representation):
   ```
   L_vae = MSE(x, x') + β·KL(q(z|x) || p(z))
   ```

2. **CBF Loss** (ensures correct barrier function):
   ```
   L_cbf = L_safe + L_unsafe + λ·L_constraint
   ```

### Training Modes

**Simple Mode** (`use_trajectory_data: false`):
- Uses only L_safe and L_unsafe
- No trajectory data required
- Faster training, may have weaker safety guarantees

**Full Mode** (`use_trajectory_data: true`):
- Uses all three loss terms including L_constraint
- Requires trajectory/consecutive state pairs
- Stronger safety guarantees

---

## Planning with CBF

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              CBF Planning Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Planning (Differentiable)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  For each step:                                       │  │
│  │    1. L_goal = ||e_decoded - e_target||              │  │
│  │    2. z_nom = z - lr * ∇L_goal                       │  │
│  │    3. z_safe = CBF_project(z_nom, z_current)         │  │
│  │    4. z = z_safe                                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  Stage 2: Validation (Geometric Ground Truth)               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  For each waypoint:                                   │  │
│  │    Robo3D.check_for_collision(q_degrees, obstacles)  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### CBF Safe Projection Detail

```python
def cbf_project(z_nom, z_current, obstacles):
    """
    Project z_nom onto safe manifold.
    """
    for each obstacle:
        # Get current safety margin
        B_current = model_cbf.barrier_function(z_current, obstacle)

        # Compute gradient of B at nominal state
        z_nom.requires_grad_(True)
        B_nom = model_cbf.barrier_function(z_nom, obstacle)
        grad_B = autograd.grad(B_nom, z_nom)[0]

        # Compute required correction
        B_target = (1 - alpha * dt) * B_current
        lambda_step = (B_target - B_nom) / (||grad_B||² + ε)

        # Apply correction if needed
        if lambda_step > 0:
            z_nom = z_nom + lambda_step * grad_B

    return z_nom
```

---

## Usage Instructions

### Training

```bash
# Navigate to project directory
cd /home/rck/RTLS/5.RTLS_CBFS

# Train CBF model (uses default config)
./train_cbf.sh

# Train with custom epochs
./train_cbf.sh --epochs 8000

# Monitor training
tensorboard --logdir model_params/panda_10k/runs_cbf
```

### Evaluation

```bash
# Run evaluation with defaults
./run_evaluation_cbf.sh

# Custom evaluation
NUM_PROBLEMS=500 NUM_OBSTACLES=2 ./run_evaluation_cbf.sh

# Direct Python invocation
python src/evaluate_planning_cbf.py \
    --checkpoint model_params/panda_10k/snapshots/model.ckpt-015350.pt \
    --checkpoint_cbf model_params/panda_10k/snapshots_cbf/model.ckpt-016000.pt \
    --config model_params/panda_10k/cbf_*-runcmd.json \
    --num_problems 1000 \
    --num_obstacles 1 \
    --alpha_cbf 1.0 \
    --dt_cbf 0.1 \
    --output results.json
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha_cbf` | CBF decay rate (higher = more conservative) | 1.0 |
| `dt_cbf` | Time step for constraint | 0.1 |
| `lambda_cbf` | Weight for CBF constraint term | 0.1 |
| `planning_lr` | Learning rate for planning | 0.03 |
| `max_steps` | Maximum planning iterations | 300 |

---

## Comparison: Original vs CBF

### Theoretical Comparison

| Aspect | Original (BCE) | CBF |
|--------|----------------|-----|
| **Formulation** | Classification | Barrier function |
| **Output space** | [0, 1] (probability) | ℝ (signed distance) |
| **Safety criterion** | P(collision) < threshold | B(x) ≥ 0 |
| **Temporal consistency** | None | Ḃ ≥ -α·B enforced |
| **Safety guarantee** | Probabilistic | Formal (if CBF satisfied) |

### Planning Comparison

| Aspect | Original | CBF |
|--------|----------|-----|
| **Collision avoidance** | Soft penalty in loss | Hard constraint via projection |
| **Update rule** | z -= lr * ∇(L_goal + L_collision) | z = project(z - lr * ∇L_goal) |
| **Guaranteed safe?** | No | Yes (if CBF accurate) |
| **Computational cost** | One backward pass | Two backward passes (goal + CBF) |

### Expected Performance

- **Success rate**: Similar or slightly lower (CBF is more conservative)
- **Collision-free rate**: Higher (formal safety enforcement)
- **Planning time**: Slightly longer (additional gradient computation)
- **Path smoothness**: May be less smooth (projection can cause discontinuities)

---

## References

1. **CBF1.pdf**: Core CBF definitions, loss functions (Equations 1-10), discretization (Equations 11-20)
2. **CBF2.pdf**: Latent safe update derivation (Equations A, 1-10)
3. Original RTLS paper: "Reaching Through Latent Space: From Joint Statistics to Path Planning in Manipulation"

---

## Summary

This implementation transforms the binary collision classifier into a Control Barrier Function by:

1. **Changing output semantics**: From P(collision) ∈ [0,1] to B(x) ∈ ℝ
2. **New loss function**: Three-term CBF loss (L_safe + L_unsafe + L_constraint)
3. **New planning algorithm**: Gradient descent + CBF safe projection
4. **Formal safety**: Ensuring Ḃ ≥ -α·B through training and planning

The result is a system with **formal safety guarantees** that prevents the robot from entering collision states, as long as the learned CBF accurately represents the true safety boundary.
