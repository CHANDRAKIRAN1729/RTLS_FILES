# Classifier vs CBF: Why Joint Training Matters

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Fundamental Question](#the-fundamental-question)
3. [Architecture Comparison](#architecture-comparison)
4. [Why Frozen Encoders Work for Classifiers](#why-frozen-encoders-work-for-classifiers)
5. [Why CBFs Require Joint Training](#why-cbfs-require-joint-training)
6. [The Gradient Direction Problem](#the-gradient-direction-problem)
7. [Mathematical Foundation](#mathematical-foundation)
8. [Visual Intuition](#visual-intuition)
9. [Implementation Implications](#implementation-implications)
10. [Summary Table](#summary-table)

---

## Executive Summary

This document explains a critical architectural difference between training a **collision classifier** and training a **Control Barrier Function (CBF)** on top of a Variational Autoencoder (VAE):

| Approach | Encoder Training | Why? |
|----------|------------------|------|
| **Classifier** | Frozen after VAE training | Only needs to separate safe from unsafe |
| **CBF** | Joint (end-to-end) training | Must shape latent space for correct gradients |

The key insight: **A classifier only answers "safe or unsafe?" but a CBF must also answer "which direction is safer?"** — and that requires reshaping the latent space through joint training.

---

## The Fundamental Question

> *Why can't we simply freeze the VAE encoder and train a CBF head on top, like we do with a collision classifier?*

This question reveals a fundamental difference between:
- **Binary classification** (does this state collide?)
- **Barrier functions** (how safe is this state AND which direction leads to safety?)

---

## Architecture Comparison

### Original Architecture: Separate Training (Classifier)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLASSIFIER: VAE + Collision Classifier (Separate Training)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: Train VAE (encoder frozen after this)                             │
│  ──────────────────────────────────────────────                             │
│                                                                             │
│    x ─────────→ [Encoder] ──────→ z ──────→ [Decoder] ──────→ x'           │
│                                                                             │
│    Loss = L_reconstruction + β * L_KL                                       │
│                                                                             │
│    • Train until convergence                                                │
│    • Encoder learns good latent representation                              │
│    • After this step: FREEZE encoder weights                                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  STEP 2: Train Classifier (encoder FROZEN)                                  │
│  ──────────────────────────────────────────                                 │
│                                                                             │
│    x ─────────→ [Encoder 🔒] ──────→ z ──────→ [Classifier] ──────→ P(col) │
│                   FROZEN            │              │                        │
│                                     │              ↓                        │
│                                     └──→ obs ──→ concat                     │
│                                                                             │
│    Loss = Binary Cross Entropy = -[y·log(p) + (1-y)·log(1-p)]              │
│                                                                             │
│    • Encoder weights are NOT updated                                        │
│    • Only classifier head learns                                            │
│    • Output: P(collision) ∈ [0, 1]                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CBF Architecture: Joint Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CBF: VAE + CBF Head (Joint / End-to-End Training)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SINGLE STEP: Train everything together                                     │
│  ──────────────────────────────────────                                     │
│                                                                             │
│    x ─────────→ [Encoder 🔓] ──────→ z ──────→ [Decoder] ──────→ x'        │
│                  TRAINABLE          │                                       │
│                                     │                                       │
│                                     ↓                                       │
│                               [CBF Head] ──────→ B(x) ∈ ℝ                  │
│                                     ↑                                       │
│                                     │                                       │
│                                   obs                                       │
│                                                                             │
│    Loss = L_VAE + λ_safe·L_safe + λ_unsafe·L_unsafe + λ_cbf·L_constraint   │
│                                                            ↑                │
│                                                   This is the key!          │
│                                                                             │
│    • Encoder weights ARE updated                                            │
│    • Latent space is shaped for safety constraints                          │
│    • Output: B(x) ∈ ℝ (unbounded real value)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Frozen Encoders Work for Classifiers

A binary classifier has a simple job: **separate two classes**. Any decision boundary that correctly separates safe from unsafe states is equally valid.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLASSIFIERS: Only Care About Separation                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Task: Distinguish safe states (○) from unsafe states (●)                  │
│                                                                             │
│         Latent Space (z)                                                    │
│         ┌────────────────────────────────────────────────────┐              │
│         │                                                    │              │
│         │     ○ ○ ○ ○             ● ● ●                     │              │
│         │       ○ ○ ○ ○       │   ● ● ● ●                   │              │
│         │         ○ ○ ○       │     ● ● ●                   │              │
│         │           ○ ○       │       ● ●                   │              │
│         │             ○       │         ●                   │              │
│         │                     │                              │              │
│         │        SAFE         │       UNSAFE                 │              │
│         │                     │                              │              │
│         │               Decision Boundary                    │              │
│         │               (any shape works!)                   │              │
│         └────────────────────────────────────────────────────┘              │
│                                                                             │
│  Requirements:                                                              │
│  ─────────────                                                              │
│    ✓ Separate safe and unsafe correctly                                    │
│    ✗ No requirement on decision boundary shape                             │
│    ✗ No requirement on gradient direction                                  │
│    ✗ No requirement on output magnitude                                    │
│                                                                             │
│  Why frozen encoder works:                                                  │
│  ─────────────────────────                                                  │
│    • Encoder learns features during VAE training                           │
│    • These features ARE informative about robot configuration              │
│    • Classifier head only needs to find ANY separating hyperplane          │
│    • As long as safe/unsafe clusters are distinguishable, it works         │
│                                                                             │
│  Mathematical guarantee:                                                    │
│  ───────────────────────                                                    │
│    If ∃ features that distinguish safe from unsafe                         │
│    → A classifier CAN learn on frozen features                             │
│    → Because classification only needs P(safe | features)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: Classification is a "What" Question

The classifier answers: **"What is this state?"** (safe or unsafe)

This is purely about labeling — any function that maps states to correct labels works. The frozen encoder provides features, and the classifier head learns a decision boundary in that feature space.

---

## Why CBFs Require Joint Training

A Control Barrier Function has a much harder job: it must not only classify states but also provide **gradients that point toward safety**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CBFs: Care About Classification AND Gradient Direction                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Task: B(x) ≥ 0 for safe, B(x) < 0 for unsafe                              │
│        AND ∇B must point toward safety!                                    │
│                                                                             │
│         Latent Space (z)                                                    │
│         ┌────────────────────────────────────────────────────┐              │
│         │                                                    │              │
│         │   B > 0     B > 0     B = 0      B < 0     B < 0  │              │
│         │   (safe)    (safe)   (boundary)  (unsafe)  (unsafe)│              │
│         │                                                    │              │
│         │     ○ ○ ○ ○ ○ ○ ○ ──────────── ● ● ● ● ● ● ●      │              │
│         │                    ↑                               │              │
│         │              B = 0 boundary                        │              │
│         │        ←←←←←←←←←←←←                               │              │
│         │        ∇B must point THIS way (toward safe)       │              │
│         │                                                    │              │
│         └────────────────────────────────────────────────────┘              │
│                                                                             │
│  Requirements:                                                              │
│  ─────────────                                                              │
│    ✓ B(x) ≥ 0 for safe states                                              │
│    ✓ B(x) < 0 for unsafe states                                            │
│    ✓ ∇B must point toward safe regions          ← NEW REQUIREMENT!         │
│    ✓ B must be smooth for gradient-based planning                          │
│    ✓ Must satisfy: dB/dt ≥ -α·B (Equation 8)   ← NEW REQUIREMENT!         │
│                                                                             │
│  Why frozen encoder FAILS:                                                  │
│  ─────────────────────────                                                  │
│    • Latent space was optimized for RECONSTRUCTION, not SAFETY             │
│    • Directions in latent space are arbitrary w.r.t. safety                │
│    • CBF head learns B values, but gradients may point wrong way           │
│    • Cannot reshape latent space to fix gradient directions                │
│                                                                             │
│  Why joint training WORKS:                                                  │
│  ─────────────────────────                                                  │
│    • Encoder learns to structure latent space for safety                   │
│    • Safe states cluster together naturally                                 │
│    • Unsafe states cluster together                                         │
│    • ∇B naturally points from unsafe → safe                                │
│    • L_constraint loss enforces correct gradient structure                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: CBF is Both a "What" AND "Where" Question

The CBF answers:
1. **"What is this state?"** (safe or unsafe) — via B(x) sign
2. **"Where is safety?"** (which direction leads to safe states) — via ∇B direction

The second question requires the latent space itself to be organized around safety concepts.

---

## The Gradient Direction Problem

This is the core issue that makes joint training necessary for CBFs.

### How Planning Uses the CBF

During planning, we use the CBF gradient to project unsafe states toward safety:

```
z_safe = z_nom + λ · ∇B
```

Where:
- `z_nom` is the nominal (possibly unsafe) next state from goal-seeking
- `∇B` is the gradient of the barrier function
- `λ > 0` is a step size computed from the CBF constraint
- `z_safe` is the corrected safe state

**This ONLY works if ∇B points toward safety!**

### The Frozen Encoder Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHY FROZEN ENCODER FAILS FOR CBF                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Scenario: Frozen encoder trained for reconstruction                        │
│                                                                             │
│  Latent Space Structure (from VAE training):                               │
│  ────────────────────────────────────────────                               │
│                                                                             │
│      The encoder learned to organize z for RECONSTRUCTION:                 │
│      • Similar joint configurations → similar z                            │
│      • Good for decoder to reconstruct                                      │
│      • NO CONSIDERATION OF SAFETY!                                          │
│                                                                             │
│         ┌────────────────────────────────────────────────────┐              │
│         │            Reconstruction-Optimized Space          │              │
│         │                                                    │              │
│         │   z1 ●    z3 ○              z5 ○                  │              │
│         │                                                    │              │
│         │        z2 ○        z4 ●                 z7 ●      │              │
│         │                                                    │              │
│         │             z6 ○                    z8 ○          │              │
│         │                                                    │              │
│         │   ○ = safe          ● = unsafe                    │              │
│         │   (randomly scattered based on arm configuration) │              │
│         └────────────────────────────────────────────────────┘              │
│                                                                             │
│  Problem: Safe and unsafe states are INTERLEAVED                           │
│  ────────────────────────────────────────────────                           │
│                                                                             │
│    • The latent space wasn't optimized to cluster safe/unsafe              │
│    • It was optimized to preserve reconstruction information               │
│    • When CBF head learns B, the gradients are misdirected                 │
│                                                                             │
│  Example of failure:                                                        │
│  ───────────────────                                                        │
│                                                                             │
│    z_nom = [0.5, 0.3, ...]     (unsafe state, B(z_nom) < 0)                │
│    ∇B = [0.1, -0.2, ...]       (gradient computed by CBF head)             │
│                                                                             │
│    Planning step:                                                           │
│    z_safe = z_nom + λ · ∇B = [0.6, 0.1, ...]                               │
│                                                                             │
│    Expected: z_safe should be safer (B(z_safe) > B(z_nom))                 │
│    Actual: z_safe might be ANOTHER unsafe state!                           │
│                                                                             │
│    Why? Because ∇B in the frozen latent space doesn't necessarily         │
│    point toward the safe cluster — it could point anywhere.                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Joint Training Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHY JOINT TRAINING FIXES THE PROBLEM                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Joint training reshapes latent space for SAFETY:                          │
│                                                                             │
│         ┌────────────────────────────────────────────────────┐              │
│         │            Safety-Optimized Space                  │              │
│         │                                                    │              │
│         │   ○ ○ ○ ○ ○                     ● ● ● ● ●         │              │
│         │     ○ ○ ○ ○                       ● ● ● ●         │              │
│         │       ○ ○ ○         ←←←←←         ● ● ●           │              │
│         │         ○ ○          ∇B            ● ●            │              │
│         │           ○                          ●            │              │
│         │                                                    │              │
│         │      SAFE CLUSTER   │   B = 0   │ UNSAFE CLUSTER  │              │
│         │       (B > 0)       │ boundary  │    (B < 0)      │              │
│         └────────────────────────────────────────────────────┘              │
│                                                                             │
│  The L_constraint loss enforces the gradient structure:                    │
│  ───────────────────────────────────────────────────────                    │
│                                                                             │
│    L_constraint = ReLU(-(dB/dt + α·B))         (Equation 8)                │
│                                                                             │
│    This loss term:                                                          │
│      • Backpropagates through the CBF head AND through the encoder         │
│      • Encoder learns to place states such that ∇B points correctly       │
│      • Creates smooth B function with well-behaved gradients               │
│                                                                             │
│  Result after joint training:                                               │
│  ────────────────────────────                                               │
│                                                                             │
│    z_nom = [0.5, 0.3, ...]     (unsafe state, B(z_nom) < 0)                │
│    ∇B = [−0.4, 0.2, ...]       (gradient points TOWARD safe cluster)      │
│                                                                             │
│    Planning step:                                                           │
│    z_safe = z_nom + λ · ∇B = [0.1, 0.5, ...]                               │
│                                                                             │
│    Now z_safe IS actually safer because:                                    │
│    • Latent space was shaped for safety during training                    │
│    • ∇B reliably points from unsafe → safe region                         │
│    • Planning projection works as intended                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### Classifier: Binary Cross-Entropy

The classifier loss only cares about correct labeling:

```
L_BCE = -[y · log(σ(f(z))) + (1-y) · log(1 - σ(f(z)))]

Where:
  y ∈ {0, 1}     : Ground truth (0=safe, 1=unsafe)
  f(z)           : Classifier output (logit)
  σ              : Sigmoid function
```

**Key properties:**
- Loss depends only on output VALUE at each point
- No constraint on gradients or smoothness
- Any decision boundary that separates the classes works
- No temporal or trajectory considerations

### CBF: Three-Term Loss

The CBF loss enforces both classification AND gradient structure:

```
L_CBF = L_safe + L_unsafe + λ · L_constraint

Where:

  L_safe = Σ max(-B(x), 0)          for x ∈ X_safe    (Equation 6)
           Penalizes B < 0 for safe states

  L_unsafe = Σ max(B(x), 0)         for x ∈ X_unsafe  (Equation 7)
             Penalizes B ≥ 0 for unsafe states

  L_constraint = Σ max(-(dB/dt + α·B), 0)             (Equation 8)
                  ↑
                  This term enforces gradient structure!
```

**The L_constraint term is crucial:**

```
dB/dt + α·B ≥ 0

Expanding dB/dt using chain rule:
∇_z B · dz/dt + α·B ≥ 0

This constraint:
  • Links the gradient ∇_z B to temporal evolution dz/dt
  • Ensures that moving along trajectories preserves safety
  • Forces the encoder to learn features where ∇B points toward safety
  • Cannot be satisfied by just changing the CBF head — encoder must adapt
```

### Why The Constraint Requires Joint Training

Consider what happens during backpropagation of L_constraint:

```
L_constraint = max(-(dB/dt + α·B), 0)

Backpropagation path:
─────────────────────

  L_constraint
       │
       ↓
  dB/dt = (B(z_{t+1}) - B(z_t)) / dt
       │
       ├───────→ ∂L/∂B(z_{t+1})
       │              │
       │              ↓
       │         B(z_{t+1}) = CBF_head(z_{t+1}, obs)
       │              │
       │              ├────→ ∂L/∂CBF_head_weights
       │              │
       │              └────→ ∂L/∂z_{t+1}
       │                          │
       │                          ↓
       │                     z_{t+1} = Encoder(x_{t+1})
       │                          │
       │                          └────→ ∂L/∂Encoder_weights  ← FROZEN in classifier!
       │                                           ↑
       └───────→ Similar path for z_t              │
                                                   │
                                    Joint training allows this gradient
                                    to update encoder weights, reshaping
                                    the latent space for safety.
```

**With frozen encoder:** Gradients stop at z, only CBF head updates
**With joint training:** Gradients flow to encoder, reshaping latent space

---

## Visual Intuition

### Classifier Decision Boundaries

```
┌───────────────────────────────────────────────────────────────────┐
│  CLASSIFIERS: Multiple Valid Decision Boundaries                  │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│    Boundary A          Boundary B          Boundary C             │
│    (linear)            (curved)            (complex)              │
│                                                                   │
│   ○ ○ ○ │ ● ●        ○ ○ ○ ╮ ● ●        ○ ○ ○ ╱╲ ● ●           │
│   ○ ○ ○ │ ● ● ●      ○ ○ ○  ╰─ ● ●      ○ ○ ○ ╱  ╲● ● ●        │
│   ○ ○ ○ │ ● ● ●      ○ ○ ○    ╰● ●      ○ ○ ○╱    ╲ ● ●         │
│   ○ ○   │   ● ●      ○ ○       ╲● ●      ○ ○ ╲    ╱ ● ●          │
│   ○     │     ●      ○          ╲●       ○    ╲╱     ●           │
│                                                                   │
│   All three boundaries achieve 100% accuracy!                     │
│   The classifier doesn't care about the shape.                    │
│   Any separating surface works equally well.                      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### CBF Level Sets

```
┌───────────────────────────────────────────────────────────────────┐
│  CBF: Level Sets and Gradient Flow                               │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  The CBF defines a smooth function with specific level sets:      │
│                                                                   │
│         B = +2      B = +1      B = 0       B = -1      B = -2   │
│           │           │           │           │           │       │
│           ↓           ↓           ↓           ↓           ↓       │
│                                                                   │
│      ╭─────╮      ╭─────╮      ╭─────╮      ╭─────╮               │
│     ╱       ╲    ╱       ╲    ╱       ╲    ╱       ╲              │
│    │  VERY   │  │  SAFE   │  │BOUNDARY│  │ UNSAFE  │  │ VERY    │
│    │  SAFE   │  │         │  │        │  │         │  │ UNSAFE  │
│     ╲       ╱    ╲       ╱    ╲       ╱    ╲       ╱              │
│      ╰─────╯      ╰─────╯      ╰─────╯      ╰─────╯               │
│                                                                   │
│   Gradients (∇B) must point perpendicular to level sets           │
│   and ALWAYS toward regions of higher B (more safety):            │
│                                                                   │
│      ←──────── ←──────── ←──────── ←──────── ←────────            │
│                          ∇B direction                             │
│                                                                   │
│   This structure CANNOT be achieved with frozen encoder!         │
│   The encoder must learn to arrange features so that ∇B          │
│   consistently points from unsafe toward safe regions.            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Planning Projection Comparison

```
┌───────────────────────────────────────────────────────────────────┐
│  PLANNING: How CBF Projection Works (and Why It Needs Joint)     │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Goal-seeking gives z_nom (possibly unsafe)               │
│  Step 2: CBF projection gives z_safe (guaranteed safe)            │
│                                                                   │
│                         correct ∇B                                │
│                       (joint training)                            │
│                                                                   │
│         SAFE REGION          │          UNSAFE REGION             │
│                              │                                    │
│                       z_nom ─┼─→ ← z_safe                         │
│                              │  ↖                                 │
│                              │    ∇B (toward safety)              │
│                              │                                    │
│                              │                                    │
│              B = 0 boundary ─┘                                    │
│                                                                   │
│  ─────────────────────────────────────────────────────────────   │
│                                                                   │
│                        WRONG ∇B                                   │
│                     (frozen encoder)                              │
│                                                                   │
│         SAFE REGION          │          UNSAFE REGION             │
│                              │                                    │
│                       z_nom ─┼─→                                  │
│                              │    ↘                               │
│                              │      ∇B (wrong direction!)         │
│                              │         ↘                          │
│                              │           z_safe ← STILL UNSAFE!   │
│                                                                   │
│   With frozen encoder, ∇B may not point toward safety because    │
│   the latent space wasn't structured for that purpose.           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Implementation Implications

### Training Code Differences

**Classifier Training (Frozen Encoder):**

```python
# Step 1: Train VAE
vae_optimizer = Adam(vae.parameters())
for epoch in range(vae_epochs):
    for batch in free_space_data:
        loss = reconstruction_loss(vae(batch))
        loss.backward()
        vae_optimizer.step()

# Step 2: Freeze encoder
for param in vae.encoder.parameters():
    param.requires_grad = False  # ← FREEZE

# Step 3: Train classifier
classifier_optimizer = Adam(classifier.parameters())  # Only classifier
for epoch in range(classifier_epochs):
    for batch in collision_data:
        x, obs, label = batch
        with torch.no_grad():  # ← Encoder not updated
            z = vae.encoder(x)
        logit = classifier(z, obs)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        loss.backward()
        classifier_optimizer.step()
```

**CBF Training (Joint):**

```python
# Single training loop: VAE + CBF together
all_params = list(vae.parameters()) + list(cbf_head.parameters())
optimizer = Adam(all_params)  # ← ALL parameters

for epoch in range(epochs):
    # VAE loss on reconstruction data
    for batch in free_space_data:
        loss_vae = reconstruction_loss(vae(batch))

    # CBF loss on collision data
    for batch in collision_data:
        x_t, x_t1, obs, label = batch

        # Encoder IS updated through this path
        z_t = vae.encoder(x_t)    # ← Gradients flow here
        z_t1 = vae.encoder(x_t1)  # ← And here

        B_t = cbf_head(z_t, obs)
        B_t1 = cbf_head(z_t1, obs)

        loss_safe = F.relu(-B_t[safe_mask]).mean()
        loss_unsafe = F.relu(B_t[unsafe_mask]).mean()

        # This term requires encoder gradients!
        dB_dt = (B_t1 - B_t) / dt
        loss_constraint = F.relu(-(dB_dt + alpha * B_t)).mean()

        loss_cbf = loss_safe + loss_unsafe + lambda_cbf * loss_constraint

    # Combined loss updates EVERYTHING
    total_loss = loss_vae + loss_cbf
    total_loss.backward()  # ← Gradients to encoder AND cbf_head
    optimizer.step()
```

### Key Differences Table

| Aspect | Classifier | CBF |
|--------|------------|-----|
| **Training phases** | 2 sequential | 1 joint |
| **Encoder updates** | Only in phase 1 | Throughout training |
| **Optimizer scope** | Phase 1: VAE; Phase 2: Classifier | All parameters together |
| **Gradient flow** | Stops at z for classifier | Flows through z to encoder |
| **Loss dependency** | Output only | Output + gradient structure |

---

## Summary Table

| Aspect | Classifier (Frozen OK) | CBF (Joint Required) |
|--------|------------------------|----------------------|
| **Output** | P(collision) ∈ [0, 1] | B(x) ∈ ℝ |
| **Output activation** | Sigmoid | None (unbounded) |
| **What it answers** | "Safe or unsafe?" | "Safe or unsafe?" + "Where is safety?" |
| **Requirement** | Just classify correctly | Classify + smooth gradients |
| **Uses gradients in planning?** | No | Yes (∇B for projection) |
| **Equation 8 constraint?** | No | Yes (dB/dt ≥ -α·B) |
| **Latent space needs** | Any separable features | Safety-aligned structure |
| **Encoder training** | Frozen after VAE | Must be trainable |
| **Loss terms** | 1 (BCE) | 3 (safe + unsafe + constraint) |
| **Temporal data needed?** | No | Yes (consecutive pairs) |
| **Why it works/fails** | Separation = classification | Need ∇B → safe direction |

---

## Conclusion

The fundamental difference between classifiers and CBFs lies in what each must provide:

1. **Classifier**: Answers "Is this state safe or unsafe?"
   - This is a **static, point-wise question**
   - Any decision boundary works
   - Frozen encoder features suffice

2. **CBF**: Answers "Is this state safe or unsafe?" AND "Which direction leads to safety?"
   - This is a **dynamic, directional question**
   - The gradient ∇B must point toward safety
   - Requires reshaping the latent space through joint training

**Bottom line**: A classifier only needs to label states correctly. A CBF must label states correctly AND provide gradients that enable safe planning. This second requirement fundamentally changes what the latent space must represent, which is why joint training is essential.

---

## References

1. **Equation 6-8**: CBF loss terms from "Learning Control Barrier Functions from Expert Demonstrations"
2. **Safe Projection**: `z_safe = z_nom + λ · ∇B` from CBF-based planning literature
3. Original RTLS: "Reaching Through Latent Space: From Joint Statistics to Path Planning in Manipulation"

---

*Document created: 2024*
*Last updated: 2024*
*Project: RTLS-CBF (Reaching Through Latent Space with Control Barrier Functions)*
