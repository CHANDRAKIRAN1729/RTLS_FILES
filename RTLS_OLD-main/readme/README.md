
# Comprehensive Analysis: RISE vs. Reaching Through Latent Space

## 1. RISE: Architecture, Data Flow & Implementation

### **Architecture Overview**

RISE consists of **three main components**:

```
┌─────────────────────────────────────────────────────────────┐
│                        RISE System                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   VAE for    │    │   Policy     │    │   Expert     │  │
│  │   Obstacle   │───▶│   Network    │◀───│ Demonstrator │  │
│  │   Encoding   │    │   π(s,g,z)   │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                    │                               │
│         │                    │                               │
│    Obstacle Data        Action Output                        │
│  (x,y,vx,vy,r)                                               │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

1. **VAE (Variational Autoencoder)**
   - **Input dimension**: 5D (obs_x, obs_y, obs_vx, obs_vy, obs_radius)
   - **Encoder**: Maps obstacle params → latent distribution (μ, σ)
   - **Latent dimension**: Learned (typically 2-4D)
   - **Decoder**: Reconstructs obstacle params from z
   - **Architecture**: 2-layer FC network with ReLU, 128 hidden units

2. **Policy Network**
   - **Input**: Concatenated [state, goal, latent_z]
   - **Output**: Action (linear velocity, angular velocity for ground vehicle; end-effector displacement for Franka)
   - **Architecture**: 2-layer FC, 128 units, ReLU

3. **Expert Demonstrator** (offline)
   - MPC, CBF, or other safe controller
   - Generates demonstrations with obstacle parameters

---

### **Data Flow: RISE**

#### **Phase 1: VAE Training (Offline)**

**Input:**
```
Dataset D_obs = {obs_i}_{i=1}^N
where obs_i = (x, y, vx, vy, radius) ∈ ℝ^5
```

**Process:**
```python
# Stage 1: Train VAE on obstacle parameters only
for epoch in 1...E_VAE:
    for batch {obs} in D_obs:
        # Encode
        μ, σ = Encoder_φ(obs)
        
        # Reparameterization trick
        ϵ ~ N(0, I)
        z = μ + σ ⊙ ϵ
        
        # Decode
        obs_hat = Decoder_ψ(z)
        
        # ELBO loss
        L = -log p_ψ(obs|z) + β·KL[q_φ(z|obs) || N(0,I)]
        
        # Update VAE
        Update φ, ψ
```

**Output:**
- Trained encoder: `q_φ(z|obs)` that maps obstacle params to latent distribution
- Trained decoder: `p_ψ(obs|z)` for reconstruction
- **VAE weights frozen after this stage**

---

#### **Phase 2: Policy Training (Offline)**

**Input:**
```
Expert demonstrations: D = {(s_t, a_t, g_t, obs_t)}
where:
  s_t = robot state (position, velocity, orientation)
  a_t = expert action
  g_t = goal position
  obs_t = obstacle parameters (5D)
```

**Process:**
```python
# Stage 2: Train policy with frozen VAE
Freeze VAE parameters: φ, ψ

for epoch in 1...E_policy:
    for batch {(s, a, g, obs)} in D:
        # Encode obstacle (no gradients to VAE)
        μ, σ = Encoder_φ(obs)  # frozen
        
        # Sample M latent perturbations
        L_policy = 0
        for m in 1...M:  # M = number of samples (e.g., 10)
            ϵ^(m) ~ N(0, I)
            z^(m) = μ + σ ⊙ ϵ^(m)
            
            # Policy forward pass
            a_pred^(m) = Policy_θ(s, g, z^(m))
            
            # Imitation loss
            L_policy += ||a_pred^(m) - a||²
        
        L_policy = L_policy / M
        
        # Update only policy
        Update θ
```

**Key Innovation**: By sampling **M different z values** from the same (μ, σ), the policy sees multiple "virtual" obstacle configurations during training, even though the actual obstacle is fixed. This is **stochastic data augmentation in latent space**.

**Output:**
- Trained policy: `π_θ(s, g, z)` that conditions on latent obstacle representation

---

#### **Phase 3: Deployment (Online)**

**Input:**
```
Current state: s_t
Goal: g_t
Noisy obstacle measurement: obs_noisy
```

**Process:**
```python
# At each timestep during execution
while not reached_goal:
    # Get noisy obstacle reading
    obs_noisy = get_sensor_reading()  # Contains measurement noise
    
    # Encode to latent (captures uncertainty)
    μ, σ = Encoder_φ(obs_noisy)
    
    # Sample ONE latent vector
    ϵ ~ N(0, I)
    z = μ + σ ⊙ ϵ
    
    # Get action from policy
    a = Policy_θ(s_current, g_target, z)
    
    # Execute action
    execute(a)
    
    # Update state
    s_current = get_new_state()
```

**Output:**
- Safe trajectory that reaches goal while avoiding obstacles
- Robustness to sensor noise (σ captures uncertainty)

---

### **Implementation Details: RISE**

**Key Files Structure:**
```
RISE/
├── vae_obstacle.py          # VAE for obstacle encoding
├── policy_network.py        # Imitation policy
├── train_vae.py            # Stage 1 training
├── train_policy.py         # Stage 2 training
├── expert_generator.py     # Generate demonstrations
└── deployment.py           # Online inference
```

**Hyperparameters:**
- VAE latent dim: 2-4
- Policy hidden units: 128
- Batch size: 256
- M (sampling during training): 10
- Expert demonstrations: 10,000

---

## 2. Reaching Through Latent Space: Architecture, Data Flow & Implementation

### **Architecture Overview**

RTLS consists of **three main components** used in **sequence**:

```
┌────────────────────────────────────────────────────────────────┐
│              Reaching Through Latent Space System              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │  VAE for      │    │  Collision   │    │  Activation    │ │
│  │  Joint State  │    │  Classifier  │    │  Maximization  │ │
│  │  (q,e) → z    │───▶│  p(c|z,o)    │───▶│  Optimizer     │ │
│  └───────────────┘    └──────────────┘    └────────────────┘ │
│         ▲                    ▲                     │           │
│         │                    │                     ▼           │
│    Robot States         Obstacle Info         Path in z       │
│    (q, e)              (x,y,h,r)                               │
└────────────────────────────────────────────────────────────────┘
```

**Components:**

1. **VAE (Variational Autoencoder)**
   - **Input dimension**: 10D (7 joint angles + 3 end-effector positions)
   - **Encoder**: Maps (q, e) → latent distribution (μ, σ)
   - **Latent dimension**: 7D (matches robot DoF)
   - **Decoder**: Reconstructs (q̂, ê) from z
   - **Architecture**: 4 hidden layers × 2048 units, ELU activation

2. **Collision Classifier**
   - **Input**: [latent z (7D), obstacle o (4D)] = 11D
   - **Output**: p(collision | z, o) via sigmoid
   - **Architecture**: 4 hidden layers × 2048 units, ELU

3. **Activation Maximization (AM)**
   - **Not a neural network** - it's an **optimization algorithm**
   - Performs gradient descent in latent space
   - Uses frozen VAE + classifier

---

### **Data Flow: RTLS**

#### **Phase 1: VAE Training (Offline)**

**Input:**
```
Dataset D_robot = {(q_i, e_i)}_{i=1}^N
where:
  q_i = joint angles ∈ ℝ^7
  e_i = end-effector position ∈ ℝ^3 (computed via FK)
  N = 100,000 random valid configurations
```

**Process:**
```python
# Stage 1: Train VAE on joint statistics
for epoch in 1...16000:
    for batch {(q, e)} in D_robot:
        # Concatenate into 10D input
        x = concat(q, e)
        
        # Encode
        μ, σ = Encoder_φ(x)
        
        # Reparameterization
        ϵ ~ N(0, I)
        z = μ + σ ⊙ ϵ
        
        # Decode
        x_hat = Decoder_θ(z)
        q_hat, e_hat = x_hat[:7], x_hat[7:]
        
        # GECO loss (adaptive β)
        reconstruction_error = ||x - x_hat||²
        KL_term = KL[q_φ(z|x) || N(0,I)]
        
        constraint = reconstruction_error - τ  # τ = 0.0008
        
        # Update Lagrange multiplier
        λ = λ · exp(α_GECO · moving_avg(constraint))
        
        L_GECO = KL_term + λ · constraint
        
        # Update VAE
        Update φ, θ
```

**Key Metric: Sample Consistency**
```python
# Measure how well decoded states respect FK
δ = ||e_hat - FK(q_hat)||²

# Model selection: choose model with lowest δ
# Good model: δ < 5mm for 95% of samples
```

**Output:**
- Trained encoder: `q_φ(z|q,e)` 
- Trained decoder: `p_θ(q,e|z)`
- Latent space where **high-likelihood regions correspond to feasible robot states**
- **VAE weights frozen after this stage**

---

#### **Phase 2: Collision Classifier Training (Offline)**

**Input:**
```
Dataset D_collision = {(q_i, e_i, o_i, c_i)}_{i=1}^N
where:
  q_i, e_i = robot state
  o_i = (x, y, h, r) = cylindrical obstacle
  c_i ∈ {0, 1} = collision label (from MoveIt)
  
  50% collision, 50% no collision (balanced)
```

**Process:**
```python
# Stage 2: Train collision classifier
Freeze VAE: φ, θ

for epoch in 1...16000:
    for batch {(q, e, o, c)} in D_collision:
        # Encode to latent (no gradients)
        x = concat(q, e)
        z = Encoder_φ(x)  # frozen, detached
        
        # Concatenate latent with obstacle
        input_classifier = concat(z, o)  # 11D
        
        # Classifier forward
        logit = Classifier_ϑ(input_classifier)
        p_collision = sigmoid(logit)
        
        # Binary cross-entropy
        L_BCE = -[c·log(p_collision) + (1-c)·log(1-p_collision)]
        
        # Update only classifier
        Update ϑ
```

**Output:**
- Trained classifier: `p_ϑ(collision | z, o)`
- **Provides differentiable gradients** for discrete collision constraint

---

#### **Phase 3: Path Planning via Latent Space Optimization (Online)**

**Input:**
```
Start state: (q_0, e_0)
Goal position: e_target ∈ ℝ^3
Obstacles: {o_1, ..., o_K}
```

**Process:**
```python
# Stage 3: Plan by optimizing in latent space
# Initialize
z_0 = Encoder_φ(q_0, e_0)  # Start in latent space
z_current = z_0.clone().requires_grad_(True)

trajectory = []

# GECO parameters for planning (separate from training!)
λ_prior = init_value
λ_obs = init_value

for t in 1...T_max (300 steps):
    # Decode current latent to robot state
    x_hat = Decoder_θ(z_current)
    q_hat, e_hat = x_hat[:7], x_hat[7:]
    
    trajectory.append((q_hat, e_hat))
    
    # Check termination
    if ||e_hat - e_target||² < γ:  # γ = 0.01m
        break
    
    # Compute loss components
    L_target = ||e_hat - e_target||²
    
    L_prior = -log p(z_current)  # = 0.5·||z_current||²
    
    L_obstacle = 0
    for obstacle_i in obstacles:
        p_coll_i = Classifier_ϑ(z_current, obstacle_i)
        L_obstacle += -log(1 - p_coll_i)
    
    # Combined loss (weighted by GECO multipliers)
    L_AM = L_target + λ_prior·L_prior + λ_obs·L_obstacle
    
    # Gradient descent in latent space
    grad = ∇_{z_current} L_AM
    z_current = z_current - α_AM · grad  # α_AM = 0.03
    
    # Update GECO multipliers (Algorithm 2 in paper)
    λ_prior = update_GECO(λ_prior, L_prior, τ_prior_goal)
    λ_obs = update_GECO(λ_obs, L_obstacle, τ_obs_goal)
```

**Output:**
- Latent trajectory: $(z_0, z_1, ..., z_T)$
- Joint-space path: $(q_0, q_1, ..., q_T)$ (decoded)
- End-effector path: $(e_0, e_1, ..., e_T)$ (decoded)
- **Path is kinematically feasible** (from VAE prior)
- **Path avoids obstacles** (from classifier gradients)

---

#### **Phase 4: Execution (Online)**

**Input:**
- Planned joint trajectory: $(q_0, ..., q_T)$

**Process:**
```python
# Execute on real robot
for q_t in trajectory:
    robot.move_to_joint_angles(q_t)
    wait_for_completion()
```

**Output:**
- Robot successfully reaches goal while avoiding obstacles

---

### **Implementation Details: RTLS**

**Key Files:**
```
Reaching Through Latent Space/
├── src/
│   ├── vae.py                    # VAE model
│   ├── vae_obs.py               # Collision classifier
│   ├── geco.py                  # GECO optimizer
│   ├── train_vae.py             # Phase 1
│   ├── train_vae_obs.py         # Phase 2
│   ├── [planning script]        # Phase 3 (not in repo)
│   └── sim/                     # FK/IK utilities
├── config/
│   ├── vae_config/              # Training params
│   └── vae_obs_config/          # Classifier params
└── data/
    ├── free_space_100k_train.dat   # Phase 1 data
    └── collision_100k_train.dat    # Phase 2 data
```

**Hyperparameters:**
- VAE latent dim: 7
- Hidden units: 2048 × 4 layers
- Batch size: 256
- Training epochs: 16,000
- Planning steps T: 300
- AM learning rate: 0.03

---

## 3. Problems Addressed & Goals

### **RISE: Problem & Goals**

**Problem:**
- **Sensor noise and measurement uncertainty** in obstacle perception
- Traditional imitation learning (BC) fails when deployed with noisy sensors
- Real-world obstacle positions/velocities are imprecise
- Safety-critical: robot must avoid obstacles despite uncertainty

**Specific Scenario:**
```
Training: Expert demonstrates safe navigation with perfect obstacle knowledge
          Obstacle at (x=2.0, y=3.0, vx=0.5, vy=0.0, r=0.3)

Deployment: Sensor reads noisy values
           Obstacle at (x=2.1±0.2, y=2.9±0.15, vx=0.45±0.1, ...) ← Uncertainty!

Standard BC: Policy trained on (s, a, obs_exact) fails with obs_noisy
RISE: Policy trained with latent z that captures uncertainty distribution
```

**Goals:**
1. ✅ Learn robust policies from demonstrations without noise injection
2. ✅ Handle realistic sensor noise at deployment
3. ✅ Maintain safety (collision avoidance) under uncertainty
4. ✅ Preserve goal-reaching performance
5. ✅ Avoid expensive online rollouts or reward engineering (IL, not RL)

---

### **RTLS: Problem & Goals**

**Problem:**
- **High-dimensional path planning** is computationally expensive
- **Inverse kinematics** for redundant manipulators is ill-posed (multimodal)
- **Non-differentiable constraints** (collision detection) prevent gradient-based optimization
- Sampling-based planners (RRT) slow in constrained spaces
- Classical optimizers (CHOMP) require explicit signed distance fields

**Specific Scenario:**
```
Input: 7-DoF robot, goal e_target in R³, multiple obstacles

Classical approach:
1. Sample random joint config q
2. Check FK: e = FK(q), collision check
3. Repeat 100,000+ times (slow!)
4. Connect valid samples (RRT)

RTLS approach:
1. Optimize in 7D latent space (lower-dim than 10D input)
2. Implicit FK (decoder learned it)
3. Differentiable collision (learned classifier)
4. 300 gradient steps → done!
```

**Goals:**
1. ✅ Learn latent representation of feasible robot configurations
2. ✅ Enable gradient-based planning in latent space
3. ✅ Handle non-differentiable constraints via learned proxies
4. ✅ Avoid explicit IK/FK computations
5. ✅ Competitive performance with classical planners
6. ✅ Task-agnostic training (no demonstrations needed)

---

### **Common Goals**

| Goal | RISE | RTLS |
|------|------|------|
| **Learn structured latent representations** | ✅ Obstacle space | ✅ Robot state space |
| **Avoid explicit constraint modeling** | ✅ (no CBF needed) | ✅ (no SDF needed) |
| **Handle uncertainty/noise** | ✅ Sensor noise | ✅ Implicit in prior |
| **Enable gradient-based methods** | ✅ Policy training | ✅ Path planning |
| **Data-driven approach** | ✅ From demos | ✅ From random samples |
| **Safety-critical tasks** | ✅ Collision avoidance | ✅ Collision avoidance |

---

## 4. Benefits of Each Approach

### **RISE Benefits**

**1. Robustness to Sensor Noise**
```
Standard BC: Trained on perfect obstacle readings
            → Fails with ±20% sensor noise
            → 40% collision rate

RISE: Trained with latent uncertainty
     → Robust to ±20% sensor noise
     → 90% safety rate
```

**2. No Noise Injection Required**
- Traditional: Must collect demonstrations with added noise (expensive, dangerous)
- RISE: Train on clean data, robustness emerges from latent sampling

**3. Interpretable Uncertainty**
- σ in latent distribution represents obstacle uncertainty
- Can visualize "plausible obstacle locations" from latent samples

**4. Computational Efficiency**
- No online replanning needed
- Single forward pass: ~1-5ms per action
- Suitable for real-time control (50 Hz)

**5. Generalizes to Unseen Noise Patterns**
- Trained without noise
- Generalizes to various noise distributions (Gaussian, uniform, outliers)
- Virtual data augmentation in latent space

**6. Easy to Extend**
- Add more obstacle types: just retrain VAE on new data
- Add velocity constraints: extend latent dimension
- Works for any differentiable policy architecture

---

### **RTLS Benefits**

**1. Task-Agnostic Learning**
```
Classical planners: Must retune for each task/robot
RTLS: Train once on random configs → works for any goal
```

**2. Implicit Inverse Kinematics**
- No need to solve $q = FK^{-1}(e_{target})$
- Handles multimodal IK solutions naturally
- Optimization in latent space automatically finds valid q

**3. Handles Non-Differentiable Constraints**
```
Classical: Collision = discrete {0,1} → no gradient
RTLS: p(collision | z, o) → smooth gradient
     ∇_z (-log(1-p)) pushes away from obstacles
```

**4. Lower-Dimensional Optimization**
- Search space: 7D latent (vs. 10D input or 7D joint with constraints)
- Structured space: smooth, feasible by design

**5. Competitive Performance**
```
Success Rate (1 obstacle):
  RRT-Connect: 84.9%
  RTLS: 85.8%

Planning Time:
  RRT-Connect: 128ms
  RTLS: 180ms (consistent, low variance)
```

**6. No Geometric Models Required**
- Classical: Need accurate 3D models, signed distance fields
- RTLS: Learned from data (collision labels from MoveIt)

**7. Composable Constraints**
- Want to add new constraint? Add new loss term!
```python
L_AM = L_target + λ_prior·L_prior + λ_obs·L_obs + λ_new·L_new
```

**8. Sample Consistency Metric**
- Novel evaluation: $\delta = ||ê - FK(\hat{q})||$
- Predictor of downstream performance
- Better than ELBO for model selection

---

### **Common Benefits**

| Benefit | RISE | RTLS |
|---------|------|------|
| **Avoids hand-engineered features** | ✅ No manual uncertainty modeling | ✅ No SDF, no IK solver |
| **Data-driven safety** | ✅ From expert demos | ✅ From collision labels |
| **Scales to complex domains** | ✅ Ground vehicle, Franka | ✅ 7-DoF manipulator |
| **Learns implicit relationships** | ✅ Obs params → safety | ✅ (q,e) → FK constraint |
| **Enables gradient-based methods** | ✅ Policy optimization | ✅ Latent optimization |

---

## 5. Detailed Comparison

### **Architecture Comparison**

| Aspect | RISE | RTLS |
|--------|------|------|
| **VAE Input** | Obstacle params (5D) | Robot states (10D: q+e) |
| **VAE Output** | Obstacle reconstruction | Robot state reconstruction |
| **Latent Dim** | 2-4D (smaller) | 7D (matches DoF) |
| **VAE Purpose** | Capture obstacle uncertainty | Capture state manifold |
| **Network Size** | Small (2×128) | Large (4×2048) |
| **Secondary Model** | None (policy is primary) | Collision classifier |
| **Planning Method** | Direct policy inference | Optimization (AM) |

---

### **Training Comparison**

| Phase | RISE | RTLS |
|-------|------|------|
| **Stage 1** | Train VAE on obstacle data | Train VAE on robot states |
| **Stage 2** | Train policy with M samples per batch | Train collision classifier |
| **Stage 3** | None (policy ready to use) | Online planning (300 steps) |
| **Data Required** | 10k expert demonstrations | 100k random configs + labels |
| **Training Time** | ~2-4 hours | ~60-80 hours (much longer!) |
| **Expertise Needed** | Expert controller for demos | Just collision checker |

---

### **Inference Comparison**

| Aspect | RISE | RTLS |
|--------|------|------|
| **Online Computation** | 1 forward pass (~1ms) | 300 gradient steps (~180ms) |
| **Real-time?** | ✅ Yes (50 Hz) | ⚠️ Moderate (5-10 Hz) |
| **Latent Sampling** | Once per timestep | Once per planning query |
| **Output** | Single action | Full trajectory |
| **Replanning** | Every timestep (reactive) | As needed (goal changes) |

---

### **Use Case Comparison**

| Scenario | RISE | RTLS |
|----------|------|------|
| **Dynamic obstacles** | ✅ Excellent (reactive policy) | ❌ Poor (static planning) |
| **Sensor noise** | ✅ Designed for this | ⚠️ Not addressed |
| **Complex kinematics** | ⚠️ Limited by demos | ✅ Excellent (7-DoF arm) |
| **Long-horizon planning** | ❌ Greedy (myopic) | ✅ Plans full trajectory |
| **Real-time control** | ✅ <5ms | ⚠️ ~200ms |
| **Training data** | ❌ Needs expert demos | ✅ Random sampling |

---

## 6. Similarities & Differences

### **Key Similarities**

**1. Core Philosophy: Latent Space Learning**
- Both learn structured latent representations
- Both leverage smoothness of latent space
- Both use VAE with Gaussian prior

**2. Two-Stage Training**
- Stage 1: Learn latent representation (VAE)
- Stage 2: Learn task-specific component
  - RISE: Policy
  - RTLS: Collision classifier

**3. Frozen VAE After Training**
- VAE trained once, then frozen
- No end-to-end gradient flow during Stage 2

**4. Stochastic Encoding**
- Both use reparameterization trick: $z = \mu + \sigma \odot \epsilon$
- Both sample from learned distributions

**5. Safety-Critical Applications**
- Both handle obstacle avoidance
- Both deployed on physical robots (Franka Panda)

**6. Data-Driven Constraint Handling**
- RISE: Implicitly from demonstrations
- RTLS: Explicitly via classifier
- Both avoid hand-coded constraint functions

---

### **Key Differences**

**1. What Gets Encoded?**
```
RISE: Encodes ENVIRONMENTAL parameters
      obstacle params → latent z

RTLS: Encodes ROBOT STATES  
      (q, e) → latent z
```

**2. Purpose of Latent Space**
```
RISE: Captures UNCERTAINTY in obstacle perception
      σ = how uncertain are we about obstacle location?

RTLS: Captures FEASIBILITY manifold of robot
      High p(z) = configuration is valid/reachable
```

**3. Sampling Strategy**
```
RISE: Multiple samples (M=10) during TRAINING
      Single sample during DEPLOYMENT
      → Stochastic data augmentation

RTLS: Single sample during VAE TRAINING (reconstruction)
      Continuous optimization during PLANNING (not sampling!)
      → Gradient descent in latent space
```

**4. How Actions Are Generated**
```
RISE: Direct policy mapping
      π(s, g, z) → action
      Learned end-to-end from demos

RTLS: Optimization-based planning
      argmin_z L(z) → trajectory
      No policy network!
```

**5. Training Data Requirements**
```
RISE: Expert demonstrations (expensive)
      - Need safe controller
      - Need diverse scenarios
      - 10,000 trajectories

RTLS: Random sampling (cheap)
      - Just sample random q
      - Compute FK(q) = e
      - Check collision
      - 100,000 states
```

**6. Computational Pattern**
```
RISE: 
  Training: SLOW (M samples per batch)
  Inference: FAST (1 forward pass)

RTLS:
  Training: SLOW (16k epochs)
  Inference: MODERATE (300 optimization steps)
```

**7. Type of Problem**
```
RISE: CONTROL problem
      - Reactive
      - Closed-loop
      - Continuous decision making

RTLS: PLANNING problem
      - Deliberative
      - Open-loop (plan then execute)
      - One-shot trajectory
```

**8. Handling Uncertainty**
```
RISE: EXPLICIT uncertainty modeling
      - VAE σ represents sensor noise
      - Policy trained to be robust to uncertainty
      - Designed for noisy observations

RTLS: IMPLICIT feasibility constraint
      - Prior loss keeps in valid region
      - Not designed for sensor noise
      - Assumes accurate state observations
```

---

## 7. Extending RISE with RTLS Ideas

### **Idea 1: Joint State-Obstacle Encoding** ⭐⭐⭐⭐⭐

**Current RISE:**
```python
# Separate encoding
z_obs = VAE_obs(obstacle_params)
policy(state, goal, z_obs)
```

**RTLS-Inspired Extension:**
```python
# Joint encoding (like RTLS encodes q+e together)
x = concat(state, obstacle_params, goal)  # e.g., 15D
z = VAE_joint(x)  # Learn joint distribution p(s, obs, g)
policy(z)  # Simpler policy, works in latent space
```

**Benefits:**
- Captures **correlations** between robot state and obstacle positions
- More expressive: "when robot is at position X, obstacle at Y matters differently"
- Could improve performance in tight spaces

**Implementation:**
```python
# Modified architecture
class JointVAE(nn.Module):
    def __init__(self):
        self.encoder = MLP(input_dim=15, latent_dim=8)
        self.decoder = MLP(latent_dim=8, output_dim=15)
    
    def forward(self, state, obs, goal):
        x = torch.cat([state, obs, goal], dim=-1)
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_recon = self.decoder(z)
        return x_recon, mu, sigma

# Training remains similar
for batch in data:
    s, obs, g, a = batch
    recon, mu, sigma = joint_vae(s, obs, g)
    # ELBO loss on reconstruction
```

---

### **Idea 2: Latent Space Path Planning for RISE** ⭐⭐⭐⭐

**Current RISE:**
- Policy is myopic (greedy)
- Plans one step at a time
- Can get stuck in local minima

**RTLS-Inspired Extension:**
Perform **trajectory optimization in latent space** before executing!

```python
# New planning layer for RISE
def plan_in_latent_space(s_start, g_target, obstacles):
    # Encode initial state + obstacles
    z_0 = encoder(s_start, obstacles)
    z_current = z_0.requires_grad_(True)
    
    trajectory = []
    for t in range(T_lookahead):  # e.g., T=50 steps
        # Decode to get predicted state
        s_pred = decoder(z_current)
        
        # Loss: how close to goal?
        L = ||s_pred - g_target||²
        
        # Gradient step in latent space
        grad = ∇_z L
        z_next = z_current - α * grad
        
        trajectory.append(z_next)
        z_current = z_next
    
    # Now have a latent trajectory
    # Execute first action from policy
    a_0 = policy(s_start, g_target, z_trajectory[0])
    return a_0
```

**Benefits:**
- **Look-ahead planning** (not greedy)
- Smooth trajectories in latent space
- Can escape local minima

**Hybrid Approach:**
```python
# Combine RTLS planning with RISE policy
if close_to_goal:
    # Use reactive RISE policy (fast)
    a = policy(s, g, z_obs)
else:
    # Use RTLS-style planning (deliberate)
    a = plan_in_latent_space(s, g, obstacles)
```

---

### **Idea 3: Sample Consistency Metric for RISE** ⭐⭐⭐

**RTLS Innovation:**
```python
# Measure FK consistency
δ = ||e_decoded - FK(q_decoded)||²
# Use δ for model selection (better than ELBO)
```

**Apply to RISE:**
```python
# Measure obstacle prediction consistency
obs_decoded = decoder(z)
obs_true = get_true_obstacle_state()  # from simulator/sensor

δ_obs = ||obs_decoded - obs_true||²

# Use for VAE model selection
# Lower δ_obs → better obstacle representation → safer policy
```

**Benefits:**
- Better VAE training: optimize for what matters (obstacle accuracy)
- Could correlate with safety rate
- Interpretable metric

---

### **Idea 4: GECO for RISE Training** ⭐⭐⭐⭐

**Current RISE:** Uses standard β-VAE
```python
L = reconstruction_loss + β * KL_divergence
# β is hard to tune!
```

**RTLS-Inspired:** Use GECO
```python
# Interpretable constraint on reconstruction
L_GECO = KL_divergence + λ * (reconstruction_loss - τ)

# τ = max acceptable obstacle reconstruction error (e.g., 0.1m²)
# λ automatically adjusted
```

**Benefits:**
- **Interpretable** τ parameter (meters of error)
- Better training stability
- Avoid posterior collapse

**Implementation:**
```python
from geco import GECO

# During VAE training
geco = GECO(goal=0.01, step_size=0.005, alpha=0.95)

for epoch in epochs:
    for batch in data:
        obs_recon, mu, sigma = vae(obs)
        
        recon_error = ||obs - obs_recon||²
        kl_loss = KL[q(z|obs) || N(0,I)]
        
        loss = geco.loss(recon_error, kl_loss)
        loss.backward()
```

---

### **Idea 5: Collision Classifier for RISE** ⭐⭐⭐⭐⭐

**Current RISE:**
- Policy implicitly learns safety from expert demos
- No explicit collision avoidance signal

**RTLS-Inspired:**
Train a **collision classifier** on latent space!

```python
# New component for RISE
class CollisionClassifierRISE(nn.Module):
    def __init__(self):
        self.fc = MLP(input_dim=latent_dim + state_dim, output_dim=1)
    
    def forward(self, state, z_obstacle):
        x = torch.cat([state, z_obstacle], dim=-1)
        logit = self.fc(x)
        return torch.sigmoid(logit)

# Training
for batch in collision_dataset:
    s, obs, collision_label = batch
    
    z_obs = vae_encoder(obs)  # frozen
    p_collision = classifier(s, z_obs)
    
    loss = BCE(p_collision, collision_label)
    loss.backward()

# During policy training - add safety term!
for batch in expert_data:
    s, a_expert, obs = batch
    
    z_obs = vae_encoder(obs)
    a_pred = policy(s, goal, z_obs)
    
    # Original imitation loss
    L_imitation = ||a_pred - a_expert||²
    
    # NEW: Safety loss (penalize collision-prone actions)
    s_next = dynamics_model(s, a_pred)  # predicted next state
    p_collision = classifier(s_next, z_obs)
    L_safety = p_collision  # want this to be LOW
    
    # Combined loss
    L_total = L_imitation + λ_safety * L_safety
```

**Benefits:**
- **Explicit safety** signal
- Can train on more diverse data (not just expert demos)
- Can add new obstacles without retraining policy (just classifier)
- Better generalization to unseen scenarios

---

### **Idea 6: Multi-Sample Planning (RTLS → RISE)** ⭐⭐⭐

**RTLS:** Deterministic optimization (single trajectory)

**RISE-Inspired Extension:** Use RISE's sampling strategy for planning

```python
# Sample-based planning in RTLS
def sample_based_planning(s_start, g_target, obstacles):
    z_0 = encoder(s_start)
    
    best_trajectory = None
    best_cost = float('inf')
    
    # Sample M candidate trajectories
    for m in range(M):
        z_current = z_0 + noise()  # Add noise for diversity
        
        trajectory = []
        for t in range(T):
            # Optimize
            z_next = optimize_step(z_current, g_target, obstacles)
            trajectory.append(z_next)
            z_current = z_next
        
        # Evaluate trajectory
        cost = compute_cost(trajectory)
        if cost < best_cost:
            best_trajectory = trajectory
            best_cost = cost
    
    return best_trajectory
```

**Benefits:**
- Avoid local minima (multiple initializations)
- Find diverse solutions
- More robust to initialization

---

### **Idea 7: Hybrid Architecture** ⭐⭐⭐⭐⭐

**Ultimate Extension:** Combine both approaches!

```python
class HybridRISE_RTLS:
    def __init__(self):
        # RTLS components
        self.vae_robot = VAE_Robot()  # Encodes (q, e)
        
        # RISE components  
        self.vae_obstacle = VAE_Obstacle()  # Encodes obstacles
        self.policy = Policy()
        
        # Shared component
        self.collision_classifier = Classifier()
    
    def plan(self, s_start, g_target, obstacles):
        # Stage 1: RTLS-style global planning
        z_robot_start = self.vae_robot.encode(s_start)
        
        # Optimize in robot latent space
        z_trajectory = []
        z_current = z_robot_start
        
        for t in range(T_global):
            # Decode
            s_decoded = self.vae_robot.decode(z_current)
            
            # Loss
            L = ||s_decoded - g_target||²
            
            # Add obstacle awareness
            z_obs = self.vae_obstacle.encode(obstacles)
            p_coll = self.collision_classifier(z_current, z_obs)
            L += λ * p_coll
            
            # Update
            z_current = z_current - α * ∇L
            z_trajectory.append(z_current)
        
        # Stage 2: RISE-style reactive control
        for z_waypoint in z_trajectory:
            s_target = self.vae_robot.decode(z_waypoint)
            
            # Use policy for local control
            while ||s_current - s_target|| > threshold:
                z_obs = self.vae_obstacle.encode(get_obstacles())
                a = self.policy(s_current, s_target, z_obs)
                execute(a)
                s_current = get_state()
```

**Benefits:**
- **Global planning** from RTLS (deliberative)
- **Local control** from RISE (reactive)
- Best of both worlds!
- Handles:
  - Sensor noise ✅ (RISE policy)
  - Complex kinematics ✅ (RTLS VAE)
  - Dynamic obstacles ✅ (RISE reactivity)
  - Long-horizon tasks ✅ (RTLS planning)

---

## Summary Table: Extension Ideas

| Idea | Difficulty | Impact | Priority |
|------|-----------|--------|----------|
| 1. Joint State-Obstacle Encoding | Medium | High | ⭐⭐⭐⭐⭐ |
| 2. Latent Space Planning | Medium | High | ⭐⭐⭐⭐ |
| 3. Sample Consistency Metric | Easy | Medium | ⭐⭐⭐ |
| 4. GECO Training | Easy | Medium | ⭐⭐⭐⭐ |
| 5. Collision Classifier | Medium | Very High | ⭐⭐⭐⭐⭐ |
| 6. Multi-Sample Planning | Easy | Medium | ⭐⭐⭐ |
| 7. Hybrid Architecture | Hard | Very High | ⭐⭐⭐⭐⭐ |

**Recommended Priority:**
1. **Collision Classifier** (biggest safety improvement)
2. **GECO Training** (better VAE, easy to implement)
3. **Joint Encoding** (better representation)
4. **Hybrid Architecture** (long-term goal)
