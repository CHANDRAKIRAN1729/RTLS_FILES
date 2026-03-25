## 2. Architecture Diagram

---

### 🏋️ Training Phase

---

#### Stage 1 — VAE Training

**Input Data:** `free_space_100k_train.dat`
- Each row: 13 floats → `[j1..j7, ex, ey, ez, gx, gy, gz]`
- Dataset returns: `x` = first 10 dims, `g` = last 3 dims

```
Input x (10-dim)
  [j1  j2  j3  j4  j5  j6  j7  |  ex  ey  ez]
              │
              ▼  normalize: (x − μ) / σ
              │
     ┌────────────────┐
     │    ENCODER     │
     │  10 → 2048     │
     │     → 2048     │
     │     → 2048     │
     │     → 2048     │
     │    ↙       ↘   │
     │  μ(7)  logVar(7)│
     └────────────────┘
          │
          ▼  z = μ + ε · √(softplus(h) + 1e-5)   [Reparameterization]
          │
     ┌────────────────┐
     │    DECODER     │
     │   7 → 2048     │
     │     → 2048     │
     │     → 2048     │
     │     → 2048     │
     │     → 10       │
     └────────────────┘
          │
          ▼  denormalize: x' = x_norm · σ + μ
          │
Reconstructed x' (10-dim)
  [j1' j2' j3' j4' j5' j6' j7'  |  ex' ey' ez']
```

**Loss Function:**
$$\mathcal{L} = \text{KL}(q(z|x) \,\|\, p(z)) + \lambda_{\text{geco}} \cdot \text{MSE}(x,\, x')$$

> `λ_geco` auto-adapts so that MSE → target goal of **0.0008**

**Training Config:** 16,000 epochs · batch size 4,096 · lr 0.0001

---

#### Stage 2 — Collision Classifier Training

**Input Data:** `collision_100k_train.dat`
- Each row: 15 floats → `[j1..j7, ex, ey, ez, ox, oy, oh, or, collision_label]`
- Dataset returns: `x[0:10]`, `obs[10:14]`, `label[14]`

```
   x (10-dim)               obs (4-dim)
[j1..j7  ex ey ez]       [ox  oy  oh  or]
        │                       │
        │ normalize with         │ normalize with
        │ free-space μ, σ        │ collision dataset μ, σ
        ▼                        │
 ┌─────────────────┐             │
 │  FROZEN ENCODER  │            │
 │  (from Stage 1)  │            │
 └─────────────────┘             │
        │                        │
      z (7-dim)                  │
        │                        │
        └──────────┬─────────────┘
                   ▼
           concat(z, obs) → 11-dim
         ┌──────────────────────┐
         │   CLASSIFIER HEAD    │
         │  11 → 2048 → ELU     │
         │     → 2048 → ELU     │
         │     → 2048 → ELU     │
         │     → 2048 → ELU     │
         │     →    1           │
         └──────────────────────┘
                   │
                   ▼
             logit (scalar)
          sigmoid(logit) → P(collision)
```

**Loss Function:** Binary Cross-Entropy `BCE(logit, label)`

> ⚠️ Only the **classifier head** is trained. Encoder + Decoder weights are **frozen**.

**Training Config:** 16,000 epochs · batch size 4,096 · lr 0.0001

---

### 🔍 Inference Phase — Path Planning via Latent Optimization

**Inputs:**
- `q_start` — 7 joint angles
- `e_target` — 3D end-effector goal position
- `obstacles` — list of `[ox, oy, oh, or]`

**Algorithm:**

```
1.  Encode q_start  ──►  z_init   (via frozen Encoder)

2.  Gradient descent on z:

      x'         =  Decoder(z)               → decoded configuration
      e'         =  x'[7:10]                 → predicted EE position

      L_goal     =  ‖e' − e_target‖          → reach the goal
      L_prior    =  0.5 · ‖z‖²              → stay in latent distribution
      L_collision=  Σ −log(1 − σ(classifier(z, obs)))   → avoid obstacles

      L_total    =  L_goal  +  λ_p · L_prior  +  λ_c · L_collision

      z  ←  z − lr · ∇L_total

3.  Stop when  ‖e' − e_target‖  <  10 mm

4.  Decode final z  →  joint angles for execution
```

---

### 📐 Component Summary

| Component | Architecture | Role |
|---|---|---|
| **Encoder** | 10 → 2048 × 4 → μ(7) + logVar(7) | Maps configs to latent space |
| **Decoder** | 7 → 2048 × 4 → 10 | Reconstructs configs from latent |
| **Classifier Head** | 11 → 2048 × 4 → 1 | Predicts collision probability |
| **Latent Space** | 7-dim Gaussian | Continuous config manifold |
