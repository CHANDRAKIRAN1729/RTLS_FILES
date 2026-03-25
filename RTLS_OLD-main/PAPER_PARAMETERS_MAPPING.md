# Paper Parameters Mapping to Implementation

This document maps the parameters mentioned in the paper to their implementation in the config files and source code.

---

## 📋 Parameter Mapping Table

| # | Paper Parameter | Symbol | Config File | Parameter Name | Current Value |
|---|----------------|---------|-------------|----------------|---------------|
| 1 | Learning rate (AM) | α<sub>AM</sub> | `vae_config/panda_10k.yaml` | `am_lr` | **0.03** |
| 2 | Learning rate (GECO) | α<sub>GECO</sub> | `vae_config/panda_10k.yaml` | `g_lr` | **0.005** |
| 3 | Max number of planning steps | T | `vae_config/panda_10k.yaml` | `am_steps` | **300** |
| 4 | Reaching distance threshold | γ | ❌ **Not explicitly configured** | - | - |
| 5 | GECO prior loss target | τ<sub>prior</sub> | `vae_config/panda_10k.yaml` | `g_goal` | **0.0008** |
| 6 | Moving average factor for prior loss | α<sub>prior</sub><sup>ma</sup> | `vae_config/panda_10k.yaml` | `g_alpha` | **0.95** |
| 7 | GECO obstacle loss target | τ<sub>obs</sub><sup>goal</sup> | ❌ **Not explicitly configured** | - | - |
| 8 | Moving average factor for obstacle loss | α<sub>obs</sub><sup>ma</sup> | ❌ **Not explicitly configured** | - | - |

---

## 📝 Detailed Parameter Analysis

### ✅ **1. Learning Rate (AM) - α<sub>AM</sub>**

**File**: `config/vae_config/panda_10k.yaml`
```yaml
# am_params:
    am_lr: 0.03
```

**Usage**: In `src/train_vae.py`, line 265
```python
optimiser_am = optim.Adam([z], lr=args.am_lr)
```

**Purpose**: Learning rate for gradient descent in latent space during Active Modeling (AM) path planning evaluation. Controls how quickly the latent code is updated to reach goal positions.

**Current Value**: **0.03**

---

### ✅ **2. Learning Rate (GECO) - α<sub>GECO</sub>**

**File**: `config/vae_config/panda_10k.yaml`
```yaml
# geco_params:
    g_lr: 0.005
```

**Usage**: In `src/train_vae.py`, line 73
```python
geco = GECO(args.g_goal, args.g_lr, args.g_alpha, args.g_init, args.g_min, args.g_max, args.g_s)
```

And in `src/geco.py`, line 18-21
```python
def __init__(self, goal, step_size, alpha=0.95, ...):
    self.step_size = step_size  # This is g_lr
```

**Purpose**: Step size for updating the GECO Lagrange multiplier (λ). Controls how aggressively GECO adjusts the balance between reconstruction and KL divergence.

**Current Value**: **0.005**

---

### ✅ **3. Max Number of Planning Steps - T**

**File**: `config/vae_config/panda_10k.yaml`
```yaml
# am_params:
    am_steps: 300
```

**Usage**: In `src/train_vae.py`, line 268
```python
# gradient descent in latent space
for _ in range(1, args.am_steps + 1):
```

**Purpose**: Maximum number of gradient descent iterations in latent space when optimizing to reach a goal position during AM evaluation.

**Current Value**: **300**

---

### ❌ **4. Reaching Distance Threshold - γ**

**Status**: **NOT EXPLICITLY CONFIGURED**

**Expected Usage**: Threshold distance to determine if a goal has been successfully reached during path planning.

**Evidence from code**: In `src/train_vae.py`, there's AUC calculation with `am_auc_max`:
```yaml
# am_params:
    am_auc_max: 0.1  # 0.1 meters = 100mm
```

This might serve as an implicit threshold for success evaluation. The AUC calculation in `train_vae.py` line 308-320 uses this to determine success rates across distance thresholds.

**Likely implicit value**: **0.1 meters (100mm)** based on `am_auc_max`

**Note**: The paper's γ threshold may be implicitly used in evaluation metrics rather than as a hard-coded stopping criterion.

---

### ✅ **5. GECO Prior Loss Target - τ<sub>prior</sub>**

**File**: `config/vae_config/panda_10k.yaml`
```yaml
# geco_params:
    g_goal: 0.0008
```

**Usage**: In `src/geco.py`, line 22
```python
self.goal = goal  # This is g_goal (τ_prior)
```

And in the loss function, line 51:
```python
constraint = err - self.goal  # Constraint tries to keep err ≤ g_goal
```

**Purpose**: Target reconstruction error (MSE) that GECO tries to maintain. The algorithm adjusts λ to keep reconstruction loss close to this value while maximizing KL divergence for better regularization.

**Current Value**: **0.0008** (8e-4)

---

### ✅ **6. Moving Average Factor for Prior Loss - α<sub>prior</sub><sup>ma</sup>**

**File**: `config/vae_config/panda_10k.yaml`
```yaml
# geco_params:
    g_alpha: 0.95
```

**Usage**: In `src/geco.py`, line 24 and 55
```python
def __init__(self, ..., alpha=0.95, ...):
    self.alpha = alpha

def loss(self, err, kld):
    # Compute moving average of constraint
    self.cma = (1.0 - self.alpha) * constraint + self.alpha * self.cma
```

**Purpose**: Exponential moving average (EMA) smoothing factor for the reconstruction error. High α (0.95) means more weight on past values, creating a smooth constraint signal that prevents wild oscillations in GECO λ updates.

**Formula**: `CMA = (1 - α) × current_error + α × previous_CMA`

**Current Value**: **0.95**

---

### ❌ **7. GECO Obstacle Loss Target - τ<sub>obs</sub><sup>goal</sup>**

**Status**: **NOT EXPLICITLY CONFIGURED**

**Expected Usage**: Target loss for the obstacle classifier network during training.

**Analysis**: The obstacle classifier training in `src/train_vae_obs.py` uses standard Binary Cross-Entropy (BCE) loss without GECO constraint optimization:

```python
# Line 120-123 in train_vae_obs.py
def loss_function_obs(obs_logit, obs_label):
    bce = F.binary_cross_entropy_with_logits(obs_logit, obs_label, reduction='sum')
    bce /= len(obs_logit)
    return bce
```

**Conclusion**: The implementation does **NOT** use GECO for obstacle classifier training. Instead, it uses standard supervised learning with BCE loss and a fixed learning rate (`lr_obs: 0.0001`).

**Why**: The obstacle classifier is a binary classification task, not a VAE, so the reconstruction-KL trade-off constraint doesn't apply.

---

### ❌ **8. Moving Average Factor for Obstacle Loss - α<sub>obs</sub><sup>ma</sup>**

**Status**: **NOT EXPLICITLY CONFIGURED**

**Expected Usage**: Moving average smoothing for obstacle loss (if GECO was used).

**Analysis**: Since the obstacle classifier doesn't use GECO (see #7), there's no moving average mechanism for its loss.

**Conclusion**: This parameter is **NOT IMPLEMENTED** because the obstacle classifier uses standard BCE loss optimization, not GECO constrained optimization.

---

## 📊 Additional GECO Parameters Found

While analyzing the code, several additional GECO parameters were found that aren't in your list but are part of the GECO algorithm:

| Parameter | Config Name | Value | Purpose |
|-----------|-------------|-------|---------|
| Initial λ | `g_init` | **1** | Starting value for GECO Lagrange multiplier |
| Min λ | `g_min` | **0.00001** (1e-5) | Lower bound for λ to prevent numerical issues |
| Max λ | `g_max` | **100000** (1e5) | Upper bound for λ to prevent explosion |
| Speedup | `g_s` | **1** | GECO speedup factor (not used in current implementation) |

---

## 🔍 Summary

### **Parameters Found in Config** ✅
1. ✅ α<sub>AM</sub> = 0.03 (AM learning rate)
2. ✅ α<sub>GECO</sub> = 0.005 (GECO learning rate)
3. ✅ T = 300 (Max planning steps)
4. ✅ τ<sub>prior</sub> = 0.0008 (GECO prior loss target)
5. ✅ α<sub>prior</sub><sup>ma</sup> = 0.95 (Prior loss moving average)

### **Parameters Not Explicitly Configured** ❌
6. ❌ γ (Reaching threshold) - **Implicitly ~0.1m from `am_auc_max`**
7. ❌ τ<sub>obs</sub><sup>goal</sup> (Obstacle loss target) - **Not used; BCE loss instead**
8. ❌ α<sub>obs</sub><sup>ma</sup> (Obstacle moving average) - **Not needed; no GECO for obstacle classifier**

### **Key Findings**

1. **VAE Training**: Uses GECO with all constrained optimization parameters properly configured
2. **Obstacle Classifier**: Uses standard supervised learning (BCE loss), not GECO
3. **Path Planning**: AM parameters properly configured for latent space optimization
4. **Reaching Threshold**: Not a hard cutoff but used implicitly in AUC evaluation

---

## 🎯 Recommendations

If you want to implement the missing parameters:

### **For γ (Reaching Threshold)**:
Add to `vae_config/panda_10k.yaml`:
```yaml
# am_params:
    am_reaching_threshold: 0.05  # 50mm success threshold
```

### **For Obstacle GECO Parameters** (if desired):
Add to `vae_obs_config/panda_10k.yaml`:
```yaml
# geco_obs_params:
    g_obs_goal: 0.01    # Target BCE loss
    g_obs_lr: 0.005     # GECO learning rate for obstacle loss
    g_obs_alpha: 0.95   # Moving average factor
```

Then modify `train_vae_obs.py` to use GECO instead of direct BCE optimization (would require significant code changes).

However, the current implementation works well without these, suggesting the paper's implementation may have evolved or these parameters are paper-specific experimental settings.
