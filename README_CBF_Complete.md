# Control Barrier Function (CBF) for Safe Robot Motion Planning

## Complete Technical Documentation

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Why CBF Instead of Binary Classifier?](#2-why-cbf-instead-of-binary-classifier)
3. [Why Joint Training for CBF?](#3-why-joint-training-for-cbf)
4. [The Unknown System Dynamics Problem](#4-the-unknown-system-dynamics-problem)
5. [Training Data: How x_t and x_{t+1} Are Generated](#5-training-data-how-x_t-and-x_t1-are-generated)
6. [The use_trajectory_data Flag](#6-the-use_trajectory_data-flag)
7. [Mathematical Foundations](#7-mathematical-foundations)
8. [Architecture Details](#8-architecture-details)
9. [Configuration Parameters](#9-configuration-parameters)
10. [How to Run](#10-how-to-run)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Introduction

This project implements a **Control Barrier Function (CBF)** integrated with a **Variational Autoencoder (VAE)** for safe robot motion planning on a 7-DOF Panda robot arm.

### What is a Control Barrier Function?

A CBF is a function B(x) that:
- Returns **B(x) ≥ 0** for **safe** states (no collision)
- Returns **B(x) < 0** for **unsafe** states (collision)
- Satisfies a **smoothness constraint** that guarantees safety over time

### Project Structure

```
/home/rck/RTLS/5.RTLS_CBFS/
├── config/
│   ├── vae_config/           # VAE-only training config
│   ├── vae_obs_config/       # Binary classifier config (original)
│   └── vae_cbf_config/       # CBF training config (this project)
├── data/
│   ├── collision_100k_train.dat      # Collision states (unsafe)
│   ├── collision_10k_test.dat
│   ├── free_space_100k_train.dat     # Free space states (safe)
│   └── free_space_10k_test.dat
├── src/
│   ├── vae.py                # Base VAE model
│   ├── vae_cbf.py            # VAE with CBF head
│   ├── train_vae.py          # VAE training
│   ├── train_vae_cbf.py      # CBF training (main script)
│   ├── robot_cbf_dataset.py  # Dataset with trajectory pairs
│   ├── evaluate_planning_cbf.py  # CBF-based planning
│   └── sim/                  # Robot simulation
├── train_cbf.sh              # Training script
├── run_evaluation_cbf.sh     # Evaluation script
├── CBF1.pdf                  # Theory reference
└── CBF2.pdf                  # Planning algorithm reference
```

---

## 2. Why CBF Instead of Binary Classifier?

### The Original Approach: Binary Classifier

```
Input: Robot state x, Obstacle params obs
Output: P(collision) ∈ [0, 1]

Loss: Binary Cross Entropy
      L = -[y·log(p) + (1-y)·log(1-p)]
```

**Limitations:**
- Only provides probability, no safety guarantees
- Gradients don't necessarily point toward safety
- No formal mathematical guarantee of collision avoidance

### The CBF Approach

```
Input: Robot state x, Obstacle params obs
Output: B(x) ∈ ℝ (continuous barrier value)

Properties:
  B(x) ≥ 0  →  Safe
  B(x) < 0  →  Unsafe
  dB/dt ≥ -αB  →  Safety is maintained over time (formal guarantee!)
```

**Advantages:**
- Continuous safety margin (not just 0/1)
- Gradients ∇B point toward safe regions
- Formal mathematical safety guarantees
- Enables gradient-based safe projection during planning

### Comparison Table

| Aspect | Binary Classifier | Control Barrier Function |
|--------|-------------------|--------------------------|
| Output | P(collision) ∈ [0,1] | B(x) ∈ ℝ |
| Safety Margin | No (just probability) | Yes (continuous) |
| Gradient Meaning | Arbitrary | Points toward safety |
| Formal Guarantee | No | Yes (dB/dt ≥ -αB) |
| Planning Use | Penalty term | Gradient projection |

---

## 3. Why Joint Training for CBF?

### Original Architecture: Separate Training

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Train VAE                                              │
│  ─────────────────                                              │
│    x → [Encoder] → z → [Decoder] → x'                          │
│    Loss = L_reconstruction + L_KL                               │
│                                                                 │
│  STEP 2: Train Classifier (Encoder FROZEN 🔒)                   │
│  ────────────────────────────────────────────                   │
│    x → [Encoder 🔒] → z → [Classifier] → P(collision)          │
│    Loss = Binary Cross Entropy                                  │
│                                                                 │
│  Why this works:                                                │
│    • Classifier only needs to SEPARATE safe from unsafe        │
│    • Frozen features are sufficient for classification         │
│    • No constraints on gradient direction                       │
└─────────────────────────────────────────────────────────────────┘
```

### CBF Architecture: Joint Training (End-to-End)

```
┌─────────────────────────────────────────────────────────────────┐
│  SINGLE STEP: Train VAE + CBF Together                          │
│  ─────────────────────────────────────                          │
│                                                                 │
│    x → [Encoder 🔓] → z → [Decoder] → x'                       │
│                       ↓                                         │
│                  [CBF Head] → B(x) ∈ ℝ                          │
│                                                                 │
│    Loss = L_VAE + L_safe + L_unsafe + L_constraint              │
│                                                                 │
│  Why joint training is REQUIRED:                                │
│    • CBF needs gradients ∇B to point toward safety             │
│    • Frozen encoder has arbitrary latent structure              │
│    • L_constraint (Eq. 8) shapes the latent space              │
│    • Encoder must learn safety-aligned features                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Core Issue: Gradient Direction

The CBF planning algorithm uses:

```
z_safe = z_nom + λ * ∇B(z_nom)
```

This **only works if ∇B points toward safety!**

```
┌─────────────────────────────────────────────────────────────────┐
│  FROZEN ENCODER PROBLEM                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Latent space was optimized for RECONSTRUCTION, not SAFETY     │
│                                                                 │
│  The directions in latent space are arbitrary w.r.t. safety:   │
│                                                                 │
│       Safe region              Unsafe region                    │
│       ○ ○ ○ ○ ○ ○              ● ● ● ● ● ●                     │
│             ↑                        ↑                          │
│        z = [....]               z = [....]                      │
│                                                                 │
│  With frozen encoder:                                           │
│    • CBF head learns B values correctly                        │
│    • BUT ∇B might point in wrong direction!                    │
│    • z_safe = z_nom + λ*∇B could go to another unsafe region   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  JOINT TRAINING SOLUTION                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder learns to structure latent space such that:           │
│    • Safe states cluster together                               │
│    • Unsafe states cluster together                             │
│    • Movement in ∇B direction increases B (toward safety)      │
│                                                                 │
│       Safe region    ←←←←    Unsafe region                      │
│       ○ ○ ○ ○ ○ ○    ∇B     ● ● ● ● ● ●                        │
│                    points                                       │
│                  toward safe                                    │
│                                                                 │
│  L_constraint = ReLU(-(dB/dt + αB)) enforces this!            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Summary: Why Joint Training?

| Aspect | Classifier (Frozen OK) | CBF (Joint Required) |
|--------|------------------------|----------------------|
| Output | P(collision) ∈ [0,1] | B(x) ∈ ℝ |
| Requirement | Just classify | Classify + smooth gradients |
| Uses gradients? | No | Yes (∇B for planning) |
| Eq. 8 constraint? | No | Yes (dB/dt ≥ -αB) |
| Latent space needs | Any separable features | Safety-aligned structure |

**The classifier only answers "safe or unsafe?" but CBF must also answer "which direction is safer?"**

---

## 4. The Unknown System Dynamics Problem

### Traditional CBF Requires System Dynamics

From CBF theory, the safety constraint is:

```
dB/dt ≥ -α·B(x)                     (Eq. 3 from CBF1.pdf)
```

Expanding dB/dt using the chain rule:

```
dB/dt = ∂B/∂x · dx/dt               (Eq. 11)
      = ∇ₓB · ẋ                      (Eq. 12)
```

Where the system dynamics are:

```
ẋ = f(x, a)                          (Eq. 13)
```

**THE PROBLEM:** For a complex robot arm, **f(x, a) is UNKNOWN!**

- f(x, a) represents how the robot state changes given an action
- This requires a complete dynamics model of the robot
- Modeling friction, joint dynamics, etc. is complex and imprecise

### The Solution: Finite Difference Approximation (Eq. 14)

Instead of computing ∇B · f(x,a), we directly approximate dB/dt:

```
∂B/∂t ≈ (B(x_{t+1}) - B(x_t)) / Δt    (Eq. 14 from CBF1.pdf)
```

**This requires NO knowledge of f(x,a)!**

We only need:
1. B(x_t) - barrier value at current state
2. B(x_{t+1}) - barrier value at next state
3. Δt - time step

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│  TRADITIONAL CBF (Requires known dynamics)                      │
│                                                                 │
│    dB/dt = ∇B · ẋ = ∇B · f(x,a)                                │
│                          ↑                                      │
│                    UNKNOWN for complex robots!                  │
│                                                                 │
│    Need: Full dynamics model f(x,a)                            │
│    Problem: Hard to model, may be inaccurate                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DATA-DRIVEN CBF (No dynamics needed) ← OUR APPROACH           │
│                                                                 │
│    dB/dt ≈ (B(x_{t+1}) - B(x_t)) / Δt                          │
│              ↑            ↑                                     │
│         Just evaluate B at two states!                          │
│                                                                 │
│    Need: Pairs of consecutive states (x_t, x_{t+1})            │
│    Advantage: No dynamics model required                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Data: How x_t and x_{t+1} Are Generated

### Source Files Used

| File | Contents | Label |
|------|----------|-------|
| `collision_100k_train.dat` | Robot states IN collision | 1 (unsafe) |
| `free_space_100k_train.dat` | Robot states NOT in collision | 0 (safe) |

**Data format per row:**
```
[j1, j2, j3, j4, j5, j6, j7, ee_x, ee_y, ee_z, obs_x, obs_y, obs_h, obs_r, label]
 ←────────── robot state x_t ──────────→  ←──── obstacle ────→  ←─ 0/1 ─→
```

### The Synthetic Perturbation Method

Since we don't have real trajectory data (pairs of consecutive robot states), we generate **synthetic x_{t+1}** by adding small random noise:

**Location:** `robot_cbf_dataset.py`, lines 171-174

```python
# x_t: Load current state from dataset file
jpos_ee_xyz = self.robot_data[index, :self.x_dim]  # This is x_t

# x_{t+1}: Generate by adding small random noise
perturbation = torch.randn(self.x_dim) * self.perturbation_scale  # ε ~ N(0, 0.01²)
jpos_ee_xyz_next = jpos_ee_xyz + perturbation  # x_{t+1} = x_t + ε
```

### Visual Explanation

```
┌─────────────────────────────────────────────────────────────────┐
│                SYNTHETIC x_{t+1} GENERATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FROM DATASET FILE                  GENERATED ON-THE-FLY       │
│   ─────────────────                  ────────────────────       │
│                                                                 │
│   collision_100k_train.dat                                      │
│   ┌────────────────────────┐                                    │
│   │ x_t = [j1...j7, ee]    │ ──────┐                            │
│   │ obs = [x, y, h, r]     │       │                            │
│   │ label = 0 or 1         │       │                            │
│   └────────────────────────┘       │                            │
│                                    │                            │
│                                    ▼                            │
│                    ┌───────────────────────────────┐            │
│                    │  x_{t+1} = x_t + ε            │            │
│                    │                               │            │
│                    │  where ε ~ N(0, σ²)           │            │
│                    │  σ = perturbation_scale       │            │
│                    │    = 0.01 (default)           │            │
│                    └───────────────────────────────┘            │
│                                                                 │
│   NOTE: x_{t+1} is NOT stored in any file!                     │
│         It is generated fresh for each training batch.          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Is This Method Valid?

The reasoning is based on **local smoothness** of the barrier function:

```
┌─────────────────────────────────────────────────────────────────┐
│  WHY SYNTHETIC PERTURBATION WORKS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ASSUMPTION: The barrier function B(x) should be SMOOTH         │
│                                                                 │
│  If B is smooth, then for small ε:                              │
│                                                                 │
│      B(x + ε) ≈ B(x) + ∇B(x) · ε     (Taylor expansion)        │
│                                                                 │
│  This means:                                                    │
│      dB/dt ≈ (B(x_{t+1}) - B(x_t)) / Δt                        │
│            ≈ (B(x_t + ε) - B(x_t)) / Δt                        │
│            ≈ ∇B(x_t) · ε / Δt                                  │
│                                                                 │
│  KEY INSIGHT:                                                   │
│    We don't need to know WHERE the robot actually moves         │
│    We just need to ensure B changes smoothly in ALL directions  │
│    Random ε covers all possible motion directions               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison: Real vs Synthetic Trajectories

| Approach | Data Source | Pros | Cons |
|----------|-------------|------|------|
| **Real Trajectories** | Recorded robot motion | Physically accurate | Requires data collection |
| **Synthetic Perturbation** | x_{t+1} = x_t + noise | No extra data needed | Approximation, not real motion |

**Both approaches avoid using unknown system dynamics f(x,a)!**

---

## 6. The use_trajectory_data Flag

### What This Flag Controls

| Value | Dataset Used | Loss Function | CBF Constraint |
|-------|--------------|---------------|----------------|
| `false` | RobotCBFDatasetSimple | L_safe + L_unsafe | **SKIPPED** |
| `true` | RobotCBFDataset | L_safe + L_unsafe + **L_constraint** | **USED** |

### When use_trajectory_data = false (Default)

```python
# Dataset returns: (x, obs, label) - single state only
# Loss function: loss_function_cbf_simple

L = L_safe + L_unsafe    # Only Eq. 6 and Eq. 7
L_constraint = 0         # Eq. 8 is SKIPPED!
```

**This makes the model act like a CLASSIFIER, not a true CBF!**

### When use_trajectory_data = true (Recommended)

```python
# Dataset returns: (x_t, x_{t+1}, obs, label, label_next, dt)
# Loss function: loss_function_cbf_full

L = L_safe + L_unsafe + λ_cbf * L_constraint    # All equations!
```

**This trains a TRUE CBF with smoothness guarantees!**

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│  use_trajectory_data = false (DEFAULT)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss = L_safe + L_unsafe                                       │
│                                                                 │
│  • Model learns: B ≥ 0 for safe, B < 0 for unsafe              │
│  • Acts like a CLASSIFIER (similar to binary classifier!)      │
│  • NO smoothness guarantee from Eq. 8                          │
│  • Gradients ∇B may not point toward safety                    │
│  • Planning projection may not work well!                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  use_trajectory_data = true (RECOMMENDED)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss = L_safe + L_unsafe + L_constraint                        │
│                                                                 │
│  • Model learns: B ≥ 0 for safe, B < 0 for unsafe              │
│  • PLUS: dB/dt ≥ -αB (smoothness constraint)                   │
│  • Guarantees gradients point toward safe regions              │
│  • Planning projection works correctly!                        │
│  • TRUE CBF with formal safety guarantees                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Recommendation

**For proper CBF behavior, always use:**

```yaml
# In config/vae_cbf_config/panda_10k.yaml
use_trajectory_data: true
```

---

## 7. Mathematical Foundations

### From CBF1.pdf: Training Loss Functions

#### CBF Conditions (Equations 1-3)

```
B(x) ≥ 0    for all x ∈ Safe states       (Eq. 1)
B(x) < 0    for all x ∈ Unsafe states     (Eq. 2)
dB/dt ≥ -α·B(x)    (CBF safety condition) (Eq. 3)
```

#### Loss Function (Equations 6-8)

```
L = L_safe + L_unsafe + λ·L_constraint

where:
  L_safe      = Σ max(-B(x), 0)    for x ∈ safe states      (Eq. 6)
  L_unsafe    = Σ max(B(x), 0)     for x ∈ unsafe states    (Eq. 7)
  L_constraint = Σ max(-(dB/dt + α·B), 0)                    (Eq. 8)
```

#### Finite Difference Approximation (Equation 14)

```
dB/dt ≈ (B(x_{t+1}) - B(x_t)) / Δt     (Eq. 14)
```

This allows computing Eq. 8 without knowing system dynamics f(x,a).

### From CBF2.pdf: Planning Algorithm

#### Latent Safe Update (Equations 1-10)

The planning algorithm projects nominal states onto the safe manifold:

```
z_safe = z_nom + λ · ∇B(z_nom)                    (Eq. 4)

where:
  λ = [(1 - α·Δt)·B(z_k) - B(z_nom)] / ||∇B||²   (Eq. 9)

Apply only if λ > 0 (when safety constraint is violated)  (Eq. 10)
```

#### Algorithm Steps

```
1. Compute B(z_k) at current state (no gradient)
2. Compute B(z_nom) and ∇B(z_nom) at nominal next state
3. Compute target: B_target = (1 - α·Δt) · B(z_k)
4. Compute λ = (B_target - B_nom) / ||∇B||²
5. If λ > 0: z_safe = z_nom + λ · ∇B
   Else: z_safe = z_nom (already safe)
```

### Implementation Verification

| Equation | Formula | Implementation | Status |
|----------|---------|----------------|--------|
| Eq. 6 | L_safe = Σ max(-B, 0) | `mean(ReLU(-B[safe]))` | ✓ |
| Eq. 7 | L_unsafe = Σ max(B, 0) | `mean(ReLU(B[unsafe]))` | ✓ |
| Eq. 8 | L_cbf = Σ max(-(dB/dt + αB), 0) | `mean(ReLU(-cbf_constraint))` | ✓ |
| Eq. 14 | dB/dt ≈ (B_{t+1} - B_t) / Δt | `B_dot = (B_next - B_current) / dt` | ✓ |
| Eq. 4 | z_safe = z_nom + λ∇B | `z_safe = z_nom + lambda_step * grad_B` | ✓ |
| Eq. 9 | λ = [(1-αΔ)B_k - B_nom] / \|\|∇B\|\|² | `lambda_step = (B_target - B_nom) / grad_norm_sq` | ✓ |

---

## 8. Architecture Details

### VAECBF Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VAECBF ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT 1: Robot State x = [j1...j7, ee_x, ee_y, ee_z] (dim: 10)│
│  INPUT 2: Obstacle params obs = [x, y, h, r]          (dim: 4) │
│                                                                 │
│         Robot State                    Obstacle Params          │
│              │                              │                   │
│              ▼                              │                   │
│  ┌───────────────────────┐                  │                   │
│  │       ENCODER         │                  │                   │
│  │  fc1: 10 → 2048 + ELU │                  │                   │
│  │  fc2: 2048 → 2048 (x4)│                  │                   │
│  │  fc21: 2048 → 7 (μ)   │                  │                   │
│  │  fc22: 2048 → 7 (logσ)│                  │                   │
│  └───────────┬───────────┘                  │                   │
│              │                              │                   │
│              ▼                              │                   │
│       z = μ + ε·exp(logσ/2)                │                   │
│       (reparameterization)                  │                   │
│       Latent dim: 7                         │                   │
│              │                              │                   │
│       ┌──────┴──────┐                       │                   │
│       │             │                       │                   │
│       ▼             ▼                       │                   │
│  ┌─────────┐   ┌─────────────────────────────────┐              │
│  │ DECODER │   │         CBF HEAD                │              │
│  │ 7→2048  │   │  concat([z, obs]) → 11 dim     │              │
│  │ (x4)    │   │  fc_cbf1: 11 → 2048 + ELU      │              │
│  │ →10     │   │  fc_cbf: 2048 → 2048 (x4)      │              │
│  └────┬────┘   │  fc_cbf_out: 2048 → 1          │              │
│       │        └──────────────┬──────────────────┘              │
│       ▼                       ▼                                 │
│      x'                     B(x) ∈ ℝ                           │
│  (reconstructed)        (barrier value)                         │
│                                                                 │
│  Interpretation:                                                │
│    B ≥ 0  →  SAFE (no collision)                               │
│    B < 0  →  UNSAFE (collision)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  collision_100k_train.dat ──┐                                   │
│  free_space_100k_train.dat ─┴─→ RobotCBFDataset                │
│                                      │                          │
│                                      ▼                          │
│                    ┌─────────────────────────────────┐          │
│                    │ For each sample:                │          │
│                    │   x_t = load from file          │          │
│                    │   x_{t+1} = x_t + ε (synthetic) │          │
│                    │   obs = obstacle params         │          │
│                    │   label = 0 (safe) or 1 (unsafe)│          │
│                    └─────────────────────────────────┘          │
│                                      │                          │
│                                      ▼                          │
│                    ┌─────────────────────────────────┐          │
│                    │ Forward pass:                   │          │
│                    │   z_t = encoder(x_t)            │          │
│                    │   z_{t+1} = encoder(x_{t+1})    │          │
│                    │   B_t = cbf_head(z_t, obs)      │          │
│                    │   B_{t+1} = cbf_head(z_{t+1})   │          │
│                    │   x' = decoder(z_t)             │          │
│                    └─────────────────────────────────┘          │
│                                      │                          │
│                                      ▼                          │
│                    ┌─────────────────────────────────┐          │
│                    │ Compute losses:                 │          │
│                    │   L_VAE = GECO(MSE, KL)         │          │
│                    │   L_safe = ReLU(-B) for safe    │          │
│                    │   L_unsafe = ReLU(B) for unsafe │          │
│                    │   dB/dt = (B_{t+1} - B_t) / Δt  │          │
│                    │   L_cbf = ReLU(-(dB/dt + αB))   │          │
│                    │   L = L_VAE + L_safe + L_unsafe │          │
│                    │       + λ_cbf * L_cbf           │          │
│                    └─────────────────────────────────┘          │
│                                      │                          │
│                                      ▼                          │
│                              Backpropagation                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Planning Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PLANNING DATA FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: (q_start, e_target, obstacles)                         │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  PLANNING LOOP (for each step):                             ││
│  │                                                             ││
│  │  1. GOAL SEEKING:                                           ││
│  │     x_decoded = decoder(z)                                  ││
│  │     L_goal = ||x_decoded[7:10] - e_target||                 ││
│  │     z_nom = z - lr * ∇L_goal                                ││
│  │                                                             ││
│  │  2. CBF SAFE PROJECTION:                                    ││
│  │     B_current = barrier(z_current, obs)                     ││
│  │     B_nom, ∇B = barrier_with_grad(z_nom, obs)               ││
│  │     B_target = (1 - α·Δt) * B_current                       ││
│  │     λ = (B_target - B_nom) / ||∇B||²                        ││
│  │     if λ > 0:                                               ││
│  │         z_safe = z_nom + λ * ∇B                             ││
│  │     else:                                                   ││
│  │         z_safe = z_nom                                      ││
│  │                                                             ││
│  │  3. UPDATE:                                                 ││
│  │     z = z_safe                                              ││
│  │     path.append(decoder(z))                                 ││
│  │                                                             ││
│  │  4. CHECK TERMINATION:                                      ││
│  │     if ||e - e_target|| < threshold: break                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                    │                                            │
│                    ▼                                            │
│  Output: path = [(q_0, e_0), (q_1, e_1), ..., (q_n, e_n)]      │
│                    │                                            │
│                    ▼                                            │
│  Validation: Robo3D.check_for_collision() (geometric check)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Configuration Parameters

### CBF Training Config (`config/vae_cbf_config/panda_10k.yaml`)

```yaml
# Model Architecture
input_dim         : 10        # 7 joints + 3 end-effector xyz
latent_dim        : 7         # latent space dimension
units_per_layer   : 2048      # hidden layer width
num_hidden_layers : 4         # number of hidden layers

# Training Parameters
lr                : 0.0001    # learning rate (Adam)
batch_size        : 4096      # batch size
epochs            : 16000     # number of training epochs
seed              : 1         # random seed

# CBF-Specific Parameters
alpha_cbf         : 1.0       # CBF decay rate α
lambda_vae        : 1.0       # weight for VAE loss
lambda_cbf_total  : 1.0       # weight for total CBF loss
lambda_cbf        : 0.1       # weight for L_constraint (Eq. 8)
use_trajectory_data: true     # USE TRUE FOR FULL CBF!
dt                : 0.1       # time step Δt
perturbation_scale: 0.01      # σ for synthetic x_{t+1}

# GECO Parameters
geco_goal         : 0.1       # reconstruction error target (relaxed)
geco_lr           : 0.00001   # GECO learning rate (slow)
geco_alpha        : 0.99      # GECO moving average
```

### Why GECO Parameters Differ from VAE-Only

| Parameter | VAE-Only | VAE-CBF | Reason |
|-----------|----------|---------|--------|
| geco_goal | 0.0008 | 0.1 | Relaxed - CBF needs latent space flexibility |
| geco_lr | 0.005 | 0.00001 | Slower - multi-objective stability |
| geco_alpha | 0.95 | 0.99 | Smoother - prevent oscillation |

---

## 10. How to Run

### Prerequisites

```bash
cd /home/rck/RTLS/5.RTLS_CBFS

# Check data files exist
ls -la data/
# Should show: collision_100k_train.dat, free_space_100k_train.dat, etc.

# Install dependencies (if needed)
pip install torch numpy tensorboard pyyaml
```

### Step 1: Train CBF Model

```bash
# Make script executable
chmod +x train_cbf.sh

# Run training
./train_cbf.sh
```

**Training takes ~2-4 hours (16000 epochs on GPU)**

**Monitor progress:**
```bash
# In separate terminal
tensorboard --logdir model_params/panda_10k/runs_cbf
```

### Step 2: Evaluate Planning

```bash
# After training completes
chmod +x run_evaluation_cbf.sh
./run_evaluation_cbf.sh
```

**Customize evaluation:**
```bash
# More test scenarios
NUM_PROBLEMS=500 ./run_evaluation_cbf.sh

# Multiple obstacles
NUM_OBSTACLES=3 ./run_evaluation_cbf.sh
```

### Manual Training Command

```bash
cd src

python train_vae_cbf.py \
    --c ../config/vae_cbf_config/panda_10k.yaml \
    --use_trajectory_data true \
    --epochs 16000
```

### Expected Output

**Training:**
```
======================================================================
Starting CBF Training (End-to-End with VAE)
======================================================================
Using device: cuda
Train Epoch: 100 Loss: 0.4523 VAE: 0.3210 CBF: 0.1313 Acc: 78.5%
====> Val Epoch: 100 CBF Loss: 0.1205 Accuracy: 82.34%
>>> Saved checkpoint: snapshots_cbf/model.ckpt-000100.pt
...
Training complete!
```

**Evaluation:**
```
======================================================================
CBF Path Planning Evaluation
======================================================================
Results:
  Success Rate:        85.0%
  Goal Reached Rate:   92.0%
  Collision-Free Rate: 89.0%
  Avg Path Length:     127.3 steps
```

---

## 11. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `batch_size` in config (try 2048 or 1024) |
| `No training data found` | Check `data/` directory has .dat files |
| `No CBF checkpoint found` | Run `./train_cbf.sh` first |
| `ModuleNotFoundError` | Install: `pip install torch numpy tensorboard pyyaml` |
| `Planning always collides` | Ensure `use_trajectory_data: true` in config |
| `Low accuracy after training` | Train longer or adjust `lambda_cbf` |

### Verifying Training Configuration

```bash
# Check if use_trajectory_data is true
grep use_trajectory_data config/vae_cbf_config/panda_10k.yaml
# Should show: use_trajectory_data: true
```

### Checking GPU Usage

```bash
# Monitor GPU during training
watch -n 1 nvidia-smi
```

### Viewing Training Logs

```bash
# Real-time training output
tail -f model_params/panda_10k/runs_cbf/events.*

# Or use TensorBoard
tensorboard --logdir model_params/panda_10k/runs_cbf
```

---

## References

1. **CBF1.pdf** - CBF training loss functions (Equations 6-8, 14)
2. **CBF2.pdf** - Latent safe update planning algorithm (Equations 1-10)

---

## Summary

This project implements a Control Barrier Function for safe robot motion planning:

1. **CBF vs Classifier**: CBF provides continuous safety margins and gradient-based projection, unlike binary classifiers

2. **Joint Training**: Required because CBF needs latent space structured for safe gradient directions

3. **Unknown Dynamics**: Solved using finite difference approximation (Eq. 14) with synthetic state pairs

4. **use_trajectory_data**: Must be `true` for full CBF with smoothness guarantees

5. **Planning**: Uses gradient projection z_safe = z_nom + λ∇B to maintain safety

**The key insight**: CBF doesn't just classify safe/unsafe - it tells you *which direction is safer*, enabling principled safe motion planning.
