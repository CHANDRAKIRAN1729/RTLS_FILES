# Control Barrier Functions (CBF): Complete Technical Deep Dive

---

# PART 1: REFERENCE CONTROLLER (Base/Nominal Controller)

---

## 1.1 Detailed Architecture

### What is a Reference Controller?

The **reference controller** (also called nominal/base controller) is any controller that achieves your primary task (e.g., goal reaching) but **DOES NOT consider safety/obstacles**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REFERENCE CONTROLLER ARCHITECTURE                        │
│                    (from reference_controller.py)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │              ProportionalController (Single Integrator)               │ │
│  │                                                                       │ │
│  │   INPUTS:                                                             │ │
│  │     x_current [2] - Current position [x, y]                          │ │
│  │     x_goal [2]    - Goal position [x_goal, y_goal]                   │ │
│  │                                                                       │ │
│  │   CONTROL LAW:                                                        │ │
│  │     ┌──────────────────────────────────────────────────────────┐     │ │
│  │     │                                                          │     │ │
│  │     │   u = -K × (x_current - x_goal)                          │     │ │
│  │     │                                                          │     │ │
│  │     │   Where:                                                 │     │ │
│  │     │     K = Proportional gain (typically 1.0)                │     │ │
│  │     │     u = Velocity command [vx, vy]                        │     │ │
│  │     │                                                          │     │ │
│  │     └──────────────────────────────────────────────────────────┘     │ │
│  │                          │                                            │ │
│  │                          ▼                                            │ │
│  │   SATURATION:                                                         │ │
│  │     ┌──────────────────────────────────────────────────────────┐     │ │
│  │     │   IF ||u|| > u_max:                                      │     │ │
│  │     │       u = u × (u_max / ||u||)                            │     │ │
│  │     └──────────────────────────────────────────────────────────┘     │ │
│  │                          │                                            │ │
│  │                          ▼                                            │ │
│  │   OUTPUT: u [2] - Saturated velocity command                         │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │           DoubleIntegratorPDController (Double Integrator)            │ │
│  │                                                                       │ │
│  │   INPUTS:                                                             │ │
│  │     state [4]    - Current state [x, y, vx, vy]                      │ │
│  │     goal_pos [2] - Goal position [x_goal, y_goal]                    │ │
│  │     goal_vel [2] - Goal velocity (default: [0, 0])                   │ │
│  │                                                                       │ │
│  │   CONTROL LAW (PD Control):                                          │ │
│  │     ┌──────────────────────────────────────────────────────────┐     │ │
│  │     │                                                          │     │ │
│  │     │   u = -Kp × (pos - goal_pos) - Kd × (vel - goal_vel)     │     │ │
│  │     │                                                          │     │ │
│  │     │   Where:                                                 │     │ │
│  │     │     Kp = Position gain (typically 1.0)                   │     │ │
│  │     │     Kd = Velocity/damping gain (typically 2.0)           │     │ │
│  │     │     u = Acceleration command [ax, ay]                    │     │ │
│  │     │                                                          │     │ │
│  │     │   Note: Kd = 2√Kp for critical damping                   │     │ │
│  │     │                                                          │     │ │
│  │     └──────────────────────────────────────────────────────────┘     │ │
│  │                          │                                            │ │
│  │                          ▼                                            │ │
│  │   SATURATION: Same as above (||u|| ≤ u_max)                          │ │
│  │                          │                                            │ │
│  │                          ▼                                            │ │
│  │   OUTPUT: u [2] - Saturated acceleration command                     │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Controller Parameters

| Controller | Parameter | Default | Description |
|------------|-----------|---------|-------------|
| Proportional | `K` | 1.0 | Proportional gain |
| Proportional | `u_max` | 2.0 | Maximum velocity magnitude |
| PD | `Kp` | 1.0 | Position gain |
| PD | `Kd` | 2.0 | Damping gain (velocity feedback) |
| PD | `u_max` | 2.0 | Maximum acceleration magnitude |

### Key Properties

| Property | Reference Controller |
|----------|---------------------|
| Goal Reaching | ✅ Yes - drives toward goal |
| Obstacle Avoidance | ❌ No - ignores obstacles |
| Formal Safety | ❌ No guarantees |
| Simplicity | ✅ Very simple |

---

## 1.2 Pipeline: Reference Controller Only

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     REFERENCE CONTROLLER PIPELINE                           │
│                        (NO SAFETY FILTER)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────┐      ┌───────────────────┐      ┌───────────────────┐      │
│   │ Current   │      │   Proportional    │      │      Robot        │      │
│   │ Position  │─────▶│    Controller     │─────▶│     (System)      │      │
│   │    x      │      │                   │      │                   │      │
│   └───────────┘      │  u = -K(x - goal) │      │  ẋ = u            │      │
│                      └───────────────────┘      └───────────────────┘      │
│         ▲                                                │                  │
│         │                                                │                  │
│         └────────────────────────────────────────────────┘                  │
│                           Feedback Loop                                     │
│                                                                             │
│   PROBLEM: This drives STRAIGHT toward goal, through obstacles!             │
│                                                                             │
│         Start                    Obstacle                    Goal           │
│           ●━━━━━━━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━▶★              │
│                               COLLISION!                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

---

# PART 2: CBF SAFETY FILTER

---

## 2.1 Detailed Architecture

### What is a Control Barrier Function?

A **Control Barrier Function (CBF)** is a mathematical tool that acts as a **safety filter** on top of any reference controller. It modifies control inputs to guarantee the system stays in a "safe set."

### Neural Network? No!

**CBF is NOT a neural network.** It is:
- A **mathematical function** h(x) that defines safety
- A **Quadratic Program (QP)** solver that finds safe controls
- An **optimization-based filter** with formal guarantees

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CBF SAFETY FILTER ARCHITECTURE                       │
│                         (from cbf_safety_filter.py)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: x [2] or [4] - Current state                                       │
│         u_ref [2]    - Reference control (from base controller)            │
│         obstacles    - List of (center, radius, margin)                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CBF SINGLE INTEGRATOR                           │   │
│  │                                                                     │   │
│  │   System Dynamics: ẋ = u  (velocity control)                       │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │             STEP 1: Compute CBF Value h(x)                  │   │   │
│  │   │                                                             │   │   │
│  │   │   For each obstacle i:                                      │   │   │
│  │   │                                                             │   │   │
│  │   │     h_i(x) = ||x - x_obs_i||² - (r_i + margin_i)²          │   │   │
│  │   │                                                             │   │   │
│  │   │   Interpretation:                                           │   │   │
│  │   │     h_i > 0 : Robot OUTSIDE obstacle (SAFE)                │   │   │
│  │   │     h_i = 0 : Robot ON boundary                            │   │   │
│  │   │     h_i < 0 : Robot INSIDE obstacle (UNSAFE)               │   │   │
│  │   │                                                             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │             STEP 2: Compute CBF Gradient ∇h(x)              │   │   │
│  │   │                                                             │   │   │
│  │   │     ∇h_i(x) = 2 × (x - x_obs_i)                             │   │   │
│  │   │                                                             │   │   │
│  │   │   This points AWAY from the obstacle center                 │   │   │
│  │   │                                                             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │             STEP 3: Compute Class K Function α(h)           │   │   │
│  │   │                                                             │   │   │
│  │   │     α(h_i) = α × h_i   (linear class K function)            │   │   │
│  │   │                                                             │   │   │
│  │   │   Properties of extended class K:                           │   │   │
│  │   │     • Strictly increasing                                   │   │   │
│  │   │     • α(0) = 0                                              │   │   │
│  │   │                                                             │   │   │
│  │   │   Effect of α:                                              │   │   │
│  │   │     • Large α: More aggressive (allows closer approach)     │   │   │
│  │   │     • Small α: More conservative (keeps more distance)      │   │   │
│  │   │                                                             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │             STEP 4: Formulate CBF-QP                        │   │   │
│  │   │                                                             │   │   │
│  │   │   QUADRATIC PROGRAM:                                        │   │   │
│  │   │   ┌───────────────────────────────────────────────────┐     │   │   │
│  │   │   │                                                   │     │   │   │
│  │   │   │   minimize   ||u - u_ref||²                       │     │   │   │
│  │   │   │      u                                            │     │   │   │
│  │   │   │                                                   │     │   │   │
│  │   │   │   subject to:                                     │     │   │   │
│  │   │   │     ∇h_i(x) · u ≥ -α(h_i)   ∀ obstacles i        │     │   │   │
│  │   │   │     ||u|| ≤ u_max           (control limit)       │     │   │   │
│  │   │   │                                                   │     │   │   │
│  │   │   └───────────────────────────────────────────────────┘     │   │   │
│  │   │                                                             │   │   │
│  │   │   Meaning:                                                  │   │   │
│  │   │     • Objective: Stay as close to u_ref as possible         │   │   │
│  │   │     • Constraint: Control must satisfy safety condition     │   │   │
│  │   │                                                             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │             STEP 5: Solve QP (using cvxpy)                  │   │   │
│  │   │                                                             │   │   │
│  │   │     u = cp.Variable(2)                                      │   │   │
│  │   │     objective = cp.Minimize(cp.sum_squares(u - u_ref))      │   │   │
│  │   │     constraints = [...]                                     │   │   │
│  │   │     problem = cp.Problem(objective, constraints)            │   │   │
│  │   │     problem.solve()  # Uses CLARABEL solver                 │   │   │
│  │   │                                                             │   │   │
│  │   │     u_safe = u.value                                        │   │   │
│  │   │                                                             │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: u_safe [2] - Safe control (closest to u_ref that is safe)        │
│          info (dict) - CBF values, constraint status, QP status           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Double Integrator CBF (Relative Degree 2)

The double integrator requires special treatment because the control (acceleration) doesn't directly affect position.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CBF DOUBLE INTEGRATOR ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   System Dynamics:                                                          │
│     ẋ = v      (position derivative = velocity)                            │
│     v̇ = u      (velocity derivative = acceleration = control)             │
│                                                                             │
│   State: [x, y, vx, vy] = [position, velocity]                             │
│   Control: u = [ax, ay] (acceleration)                                     │
│                                                                             │
│   PROBLEM: h(x) depends on position, but control affects acceleration      │
│            This is "relative degree 2" - need to differentiate h twice     │
│                                                                             │
│   SOLUTION: Exponential CBF (Higher-Order CBF)                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    EXPONENTIAL CBF FORMULATION                      │  │
│   │                                                                     │  │
│   │   Primary barrier:                                                  │  │
│   │     h(x) = ||p - p_obs||² - r²                                      │  │
│   │                                                                     │  │
│   │   First derivative (depends on velocity, not control):              │  │
│   │     ḣ = ∇h · ẋ = 2(p - p_obs) · v                                   │  │
│   │                                                                     │  │
│   │   Composite CBF (exponential CBF):                                  │  │
│   │     ψ(x) = ḣ + λ × h                                                │  │
│   │          = 2(p - p_obs) · v + λ × (||p - p_obs||² - r²)             │  │
│   │                                                                     │  │
│   │   Now enforce:                                                      │  │
│   │     ψ̇ ≥ -α(ψ)                                                       │  │
│   │                                                                     │  │
│   │   Derivative of ψ:                                                  │  │
│   │     ψ̇ = 2v·v + 2(p - p_obs)·u + 2λ(p - p_obs)·v                    │  │
│   │        = A + B·u                                                    │  │
│   │                                                                     │  │
│   │   Where:                                                            │  │
│   │     A = 2||v||² + 2λ(p - p_obs)·v    (drift term)                  │  │
│   │     B = 2(p - p_obs)                  (control coefficient)         │  │
│   │                                                                     │  │
│   │   CBF Constraint:                                                   │  │
│   │     A + B·u ≥ -α×ψ                                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   QP formulation same as single integrator, but with different constraint  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Exponential CBF?

| Problem | Solution |
|---------|----------|
| h depends on position only | Need to consider velocity (momentum) |
| Robot approaching obstacle fast | ḣ could be very negative (approaching fast) |
| Need to brake in time | ψ = ḣ + λh combines both position AND velocity |
| λ controls braking | Larger λ = earlier braking |

---

## 2.3 Mathematical Background (No Dataset - Pure Math!)

### CBF Does NOT Use Training Data

Unlike neural networks, CBF is **completely analytical**:

| Component | Source |
|-----------|--------|
| h(x) function | Hand-designed (geometric formula) |
| Gradient ∇h(x) | Analytical derivative |
| Class K function | Design choice (linear, polynomial, etc.) |
| QP solver | Mathematical optimization (cvxpy/CLARABEL) |

### Control-Affine Systems

CBFs work with control-affine systems:

```
ẋ = f(x) + g(x)·u
```

| System | f(x) | g(x) | State | Control |
|--------|------|------|-------|---------|
| Single Integrator | 0 | I (identity) | [x, y] | velocity |
| Double Integrator | [v; 0] | [0; I] | [x, y, vx, vy] | acceleration |

### CBF Condition (Lie Derivatives)

For h(x) to be a valid CBF:

```
Lf h(x) + Lg h(x) · u ≥ -α(h(x))
```

Where:
- `Lf h(x) = ∇h(x) · f(x)` : Lie derivative along drift
- `Lg h(x) = ∇h(x) · g(x)` : Lie derivative along control

### Single Integrator Derivation

```
System: ẋ = u

h(x) = ||x - x_obs||² - r²
∇h(x) = 2(x - x_obs)

Lf h(x) = ∇h · f = ∇h · 0 = 0
Lg h(x) = ∇h · g = ∇h · I = 2(x - x_obs)

CBF Constraint: 2(x - x_obs) · u ≥ -α·h(x)
```

### Double Integrator Derivation

```
System: ṗ = v, v̇ = u
State: x = [p; v]

h(x) = ||p - p_obs||² - r²
ḣ = 2(p - p_obs) · v

ψ = ḣ + λh  (exponential CBF)
ψ̇ = 2v·v + 2(p - p_obs)·u + 2λ(p - p_obs)·v
   = [2||v||² + 2λ(p - p_obs)·v] + [2(p - p_obs)]·u
   =           A                 +        B      ·u

CBF Constraint: A + B·u ≥ -α·ψ
```

---

## 2.4 Code and Config Files

### File Structure

```
Control_Barrier_Functions/
├── cbf_safety_filter.py      # Core CBF implementation
├── reference_controller.py   # Base controllers
├── simulation.py             # Demos and visualization
├── animation.py              # Animated demos
├── README.md                 # Documentation
└── learningcbfs.txt          # Original paper (reference)
```

### Main Classes

| File | Class | Purpose |
|------|-------|---------|
| cbf_safety_filter.py | `Obstacle` | Circular obstacle dataclass |
| cbf_safety_filter.py | `CBFSingleIntegrator` | CBF for ẋ = u systems |
| cbf_safety_filter.py | `CBFDoubleIntegrator` | CBF for ẍ = u systems |
| cbf_safety_filter.py | `MultiObstacleCBF` | Unified interface |
| reference_controller.py | `ProportionalController` | P-controller (single) |
| reference_controller.py | `DoubleIntegratorPDController` | PD-controller (double) |
| reference_controller.py | `PotentialFieldController` | APF method (comparison) |

### Parameters (NOT config file - hardcoded)

```python
# CBF Parameters
alpha = 1.0           # Class K function coefficient
                      # - Larger: more aggressive (closer to obstacles)
                      # - Smaller: more conservative (stays farther)

lambda_cbf = 1.0      # Exponential CBF coefficient (double integrator)
                      # - Larger: earlier braking
                      # - Smaller: later braking

u_max = 2.0           # Maximum control magnitude

safety_margin = 0.1   # Extra buffer around obstacles

# Reference Controller Parameters
K = 1.0               # Proportional gain (single integrator)
Kp = 1.0              # Position gain (double integrator)
Kd = 2.0              # Damping gain (double integrator)

# Simulation Parameters
dt = 0.02             # Time step
max_steps = 500       # Maximum iterations
```

### Dependencies

```python
# Required
numpy                 # Numerical computation
cvxpy                 # Convex optimization (QP solving)
matplotlib            # Visualization

# Solver (comes with cvxpy)
CLARABEL              # Default solver (supports SOCP constraints)
```

---

## 2.5 Full CBF Algorithm

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CBF SAFETY FILTER ALGORITHM                          ║
║                          (Single Integrator Version)                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT:                                                                      ║
║    - x [2]: Current position                                                ║
║    - u_ref [2]: Reference control from base controller                      ║
║    - obstacles: List of Obstacle(center, radius, margin)                    ║
║    - α: Class K function coefficient                                        ║
║    - u_max: Control magnitude limit                                         ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                        │ ║
║  │  FOR each obstacle i:                                                  │ ║
║  │                                                                        │ ║
║  │     # STEP 1: Compute CBF value                                        │ ║
║  │     r_safe = radius_i + margin_i                                       │ ║
║  │     h_i = ||x - center_i||² - r_safe²                                  │ ║
║  │                                                                        │ ║
║  │     # STEP 2: Compute CBF gradient                                     │ ║
║  │     grad_h_i = 2 × (x - center_i)                                      │ ║
║  │                                                                        │ ║
║  │     # STEP 3: Compute class K function value                           │ ║
║  │     alpha_h_i = α × h_i                                                │ ║
║  │                                                                        │ ║
║  │     # STEP 4: Add constraint                                           │ ║
║  │     constraints.append(grad_h_i @ u >= -alpha_h_i)                     │ ║
║  │                                                                        │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  # STEP 5: Add control limit constraint                                     ║
║  constraints.append(||u|| <= u_max)                                         ║
║                                                                              ║
║  # STEP 6: Define QP                                                        ║
║  u = cp.Variable(2)                                                         ║
║  objective = cp.Minimize(||u - u_ref||²)                                    ║
║  problem = cp.Problem(objective, constraints)                               ║
║                                                                              ║
║  # STEP 7: Solve QP                                                         ║
║  problem.solve()                                                            ║
║                                                                              ║
║  # STEP 8: Extract safe control                                             ║
║  IF status == 'optimal':                                                    ║
║      u_safe = u.value                                                       ║
║  ELSE:                                                                      ║
║      u_safe = [0, 0]  # Emergency stop                                      ║
║                                                                              ║
║  OUTPUT: u_safe [2] - Safe control closest to u_ref                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2.6 CBF Safety Filter Mind Map

```
                                    ┌──────────────────────────────────────────────────┐
                                    │             CBF SAFETY FILTER                     │
                                    │        (What vs How vs Why)                       │
                                    └──────────────────────────────────────────────────┘
                                                         │
              ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
              │                                          │                                          │
              ▼                                          ▼                                          ▼
┌─────────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────────┐
│         WHAT            │               │          HOW            │               │          WHY            │
│  (Components)           │               │   (Mechanism)           │               │    (Purpose)            │
└─────────────────────────┘               └─────────────────────────┘               └─────────────────────────┘
              │                                          │                                          │
    ┌─────────┴─────────┐                      ┌─────────┴─────────┐                      ┌─────────┴─────────┐
    │                   │                      │                   │                      │                   │
    ▼                   ▼                      ▼                   ▼                      ▼                   ▼
┌───────────┐     ┌───────────┐          ┌───────────┐     ┌───────────┐          ┌───────────┐     ┌───────────┐
│ Reference │     │   CBF     │          │  Compute  │     │  Solve    │          │  Formal   │     │  Minimal  │
│ Controller│     │ Function  │          │ Constraint│     │    QP     │          │  Safety   │     │ Deviation │
│           │     │  h(x)     │          │           │     │           │          │ Guarantee │     │           │
│ u_ref     │     │           │          │ ∇h·u ≥    │     │ min       │          │           │     │ Stay close│
│ (unsafe)  │     │ Defines   │          │  -α(h)    │     │ ||u-u_ref||│          │ Never     │     │ to u_ref  │
│           │     │ safe set  │          │           │     │           │          │ violate   │     │           │
│           │     │           │          │           │     │ s.t.      │          │ h(x) ≥ 0  │     │           │
└───────────┘     └───────────┘          └───────────┘     │ constraints│          └───────────┘     └───────────┘
                                                           └───────────┘


                                    ┌──────────────────────────────────────────────────┐
                                    │              SYSTEM TYPES                        │
                                    └──────────────────────────────────────────────────┘
                                                         │
                        ┌────────────────────────────────┴────────────────────────────────┐
                        │                                                                 │
                        ▼                                                                 ▼
          ┌─────────────────────────────┐                               ┌─────────────────────────────┐
          │     SINGLE INTEGRATOR       │                               │     DOUBLE INTEGRATOR       │
          │         ẋ = u               │                               │      ẋ = v, v̇ = u           │
          └─────────────────────────────┘                               └─────────────────────────────┘
                        │                                                                 │
          ┌─────────────┴─────────────┐                               ┌─────────────────┴─────────────┐
          │                           │                               │                               │
          ▼                           ▼                               ▼                               ▼
    ┌───────────┐             ┌───────────┐                     ┌───────────┐                 ┌───────────┐
    │ State     │             │ Control   │                     │ State     │                 │ Control   │
    │ [x, y]    │             │ velocity  │                     │ [x,y,vx,vy]                 │ acceleration
    │           │             │           │                     │           │                 │           │
    │ Position  │             │ Directly  │                     │ Position +│                 │ Affects   │
    │ only      │             │ affects   │                     │ Velocity  │                 │ velocity  │
    │           │             │ position  │                     │           │                 │ (indirect)│
    └───────────┘             └───────────┘                     └───────────┘                 └───────────┘
                                    │                                                               │
                                    ▼                                                               ▼
                              ┌───────────┐                                                   ┌───────────┐
                              │ Direct    │                                                   │Exponential│
                              │ CBF h(x)  │                                                   │ CBF ψ(x)  │
                              │           │                                                   │           │
                              │ ∇h·u ≥-αh │                                                   │ψ = ḣ + λh │
                              │           │                                                   │ψ̇ ≥ -αψ   │
                              └───────────┘                                                   └───────────┘


                                    ┌──────────────────────────────────────────────────┐
                                    │            GEOMETRY OF CBF                       │
                                    └──────────────────────────────────────────────────┘

                              Safe Set (h > 0)
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
            ●─────────────────────●───────────────●
          Robot              Boundary           Obstacle
        (safe here)          (h = 0)           (h < 0)
            │                    │                │
            ▼                    ▼                ▼
       Can move            Cannot cross      NEVER enter
       freely            (CBF prevents)      (guaranteed)

```

---

---

# PART 3: FULL CBF SYSTEM (Reference + Filter)

---

## 3.1 Detailed Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE CBF PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────┐                                                             │
│   │ SENSORS   │                                                             │
│   │           │                                                             │
│   │ Robot     │─────┐                                                       │
│   │ Position  │     │                                                       │
│   │           │     │                                                       │
│   │ Obstacle  │     │                                                       │
│   │ Locations │     │                                                       │
│   └───────────┘     │                                                       │
│                     ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     LAYER 1: REFERENCE CONTROLLER                   │  │
│   │                                                                     │  │
│   │   INPUT: x (current position), x_goal (goal position)              │  │
│   │                                                                     │  │
│   │   COMPUTE: u_ref = -K × (x - x_goal)  [+ saturation]               │  │
│   │                                                                     │  │
│   │   OUTPUT: u_ref (desired control - may be UNSAFE!)                 │  │
│   │                                                                     │  │
│   │   PROPERTY: Reaches goal but IGNORES obstacles                     │  │
│   │                                                                     │  │
│   └─────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     LAYER 2: CBF SAFETY FILTER                      │  │
│   │                                                                     │  │
│   │   INPUT: x (current state), u_ref (reference control),             │  │
│   │          obstacles (list of Obstacle)                              │  │
│   │                                                                     │  │
│   │   FOR EACH obstacle:                                               │  │
│   │     1. Compute h(x) = distance² - radius²                          │  │
│   │     2. Compute ∇h(x) = 2(x - x_obs)                                │  │
│   │     3. Add constraint: ∇h·u ≥ -α×h                                 │  │
│   │                                                                     │  │
│   │   SOLVE QP:                                                        │  │
│   │     minimize ||u - u_ref||²                                        │  │
│   │     subject to all CBF constraints + ||u|| ≤ u_max                 │  │
│   │                                                                     │  │
│   │   OUTPUT: u_safe (closest safe control to u_ref)                   │  │
│   │                                                                     │  │
│   │   PROPERTY: GUARANTEES safety while staying close to u_ref         │  │
│   │                                                                     │  │
│   └─────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     LAYER 3: SYSTEM DYNAMICS                        │  │
│   │                                                                     │  │
│   │   Single Integrator: x_new = x + u_safe × dt                       │  │
│   │                                                                     │  │
│   │   Double Integrator: pos_new = pos + vel×dt + 0.5×u_safe×dt²       │  │
│   │                      vel_new = vel + u_safe × dt                   │  │
│   │                                                                     │  │
│   └─────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                          │
│                                 │ Feedback                                 │
│                                 ▼                                          │
│                           ┌───────────┐                                    │
│                           │   ROBOT   │                                    │
│                           │           │                                    │
│                           │ New state │───────────────────┐                │
│                           │           │                   │                │
│                           └───────────┘                   │                │
│                                                           │                │
│                               ▲                           │                │
│                               │                           │                │
│                               └───────────────────────────┘                │
│                                     State feedback                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Full Simulation Algorithm

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      FULL CBF SIMULATION ALGORITHM                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT:                                                                      ║
║    - start [2]: Initial position                                            ║
║    - goal [2]: Target position                                              ║
║    - obstacles: List of Obstacle(center, radius, margin)                    ║
║    - dt: Time step (e.g., 0.02s)                                            ║
║    - max_steps: Maximum iterations (e.g., 500)                              ║
║                                                                              ║
║  INITIALIZE:                                                                 ║
║    - x = start.copy()                                                       ║
║    - trajectory = [x.copy()]                                                ║
║    - ref_controller = ProportionalController(K=1.5, u_max=2.0)              ║
║    - cbf = CBFSingleIntegrator(obstacles, alpha=1.0, u_max=2.0)             ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ FOR step = 0 to max_steps-1:                                           │ ║
║  │                                                                        │ ║
║  │   # CHECK: Goal reached?                                               │ ║
║  │   IF ||x - goal|| < 0.1:                                               │ ║
║  │       BREAK (success!)                                                 │ ║
║  │                                                                        │ ║
║  │   # STEP 1: Reference Controller (ignores obstacles)                   │ ║
║  │   u_ref = ref_controller.compute(x, goal)                              │ ║
║  │   # u_ref = -K × (x - goal), saturated                                 │ ║
║  │                                                                        │ ║
║  │   # STEP 2: CBF Safety Filter                                          │ ║
║  │   u_safe, info = cbf.safe_control(x, u_ref)                            │ ║
║  │   # u_safe = closest safe control to u_ref                             │ ║
║  │   # Solves QP with CBF constraints                                     │ ║
║  │                                                                        │ ║
║  │   # STEP 3: Apply control to system                                    │ ║
║  │   x = x + u_safe × dt                                                  │ ║
║  │                                                                        │ ║
║  │   # STEP 4: Record trajectory                                          │ ║
║  │   trajectory.append(x.copy())                                          │ ║
║  │                                                                        │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  # POST-PROCESS                                                              ║
║  trajectory = np.array(trajectory)                                          ║
║  min_dist = min distance to any obstacle surface                            ║
║  collision = (min_dist < 0)                                                 ║
║  goal_reached = ||trajectory[-1] - goal|| < threshold                       ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    - trajectory: Array of positions [N × 2]                                 ║
║    - collision: Boolean (should always be False with CBF!)                  ║
║    - goal_reached: Boolean                                                  ║
║    - min_dist: Closest approach to obstacle (should be ≥ 0)                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3.3 Complete System Mind Map

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CONTROL BARRIER FUNCTION - COMPLETE SYSTEM                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
          ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
          │                                         │                                         │
          ▼                                         ▼                                         ▼
┌───────────────────────┐               ┌───────────────────────┐               ┌───────────────────────┐
│   REFERENCE LAYER     │               │     CBF LAYER         │               │    SYSTEM LAYER       │
│  (Goal Reaching)      │               │  (Safety Filter)      │               │   (Robot Dynamics)    │
└───────────────────────┘               └───────────────────────┘               └───────────────────────┘
          │                                         │                                         │
  ┌───────┴───────┐                       ┌─────────┴─────────┐                     ┌─────────┴─────────┐
  │               │                       │                   │                     │                   │
  ▼               ▼                       ▼                   ▼                     ▼                   ▼
┌─────────┐ ┌─────────┐             ┌─────────┐         ┌─────────┐           ┌─────────┐         ┌─────────┐
│Proport- │ │ PD      │             │ h(x)    │         │ QP      │           │ Single  │         │ Double  │
│ional    │ │ Control │             │ Barrier │         │ Solver  │           │ Integr- │         │ Integr- │
│         │ │         │             │ Function│         │         │           │ ator    │         │ ator    │
│u=-K(x-g)│ │u=-Kp×e  │             │         │         │         │           │         │         │         │
│         │ │ -Kd×ė   │             │||x-obs||²│         │min|u-   │           │ ẋ = u   │         │ ẋ = v   │
│         │ │         │             │  - r²   │         │  u_ref|²│           │         │         │ v̇ = u   │
│         │ │         │             │         │         │         │           │         │         │         │
│ Single  │ │ Double  │             │ h > 0   │         │s.t. CBF │           │velocity │         │accel    │
│ integr  │ │ integr  │             │ = safe  │         │constraint│          │control  │         │control  │
└─────────┘ └─────────┘             └─────────┘         └─────────┘           └─────────┘         └─────────┘
     │           │                       │                   │                     │                   │
     └─────┬─────┘                       └─────────┬─────────┘                     └─────────┬─────────┘
           │                                       │                                         │
           ▼                                       ▼                                         ▼
    ┌─────────────┐                       ┌─────────────┐                           ┌─────────────┐
    │   OUTPUT:   │                       │   OUTPUT:   │                           │   OUTPUT:   │
    │   u_ref     │─────────────────────▶│   u_safe    │─────────────────────────▶│   x_new     │
    │  (unsafe)   │                       │   (safe)    │                           │  (state)    │
    └─────────────┘                       └─────────────┘                           └─────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    WHY CBF WORKS                                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

          ┌────────────────────┐                                    ┌────────────────────┐
          │   WITHOUT CBF      │                                    │    WITH CBF        │
          └────────────────────┘                                    └────────────────────┘
                   │                                                         │
                   ▼                                                         ▼
          
          Start ●                                                   Start ●
                 \                                                          \
                  \                                                          \
                   \                                                          \
                    \                                                          ╲
                     \    ┌───┐                                                 ╲  ┌───┐
                      \   │OBS│ ← Collision!                                     ╰─│OBS│─╮
                       \  │   │                                                    │   │ │
                        \ └───┘                                                    └───┘ │
                         \                                                               │
                          ▼                                                              ▼
                      Goal ★                                                         Goal ★
          
          Trajectory: Straight line                         Trajectory: Curves around obstacle
          Result: COLLISION                                 Result: SAFE + reaches goal


┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                CBF vs OTHER METHODS                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│     Method        │   Safety Guarantee  │   Local Minima      │   Computation       │
├───────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Potential Fields  │   ❌ No formal      │   ❌ Gets stuck     │   ✅ Fast           │
│                   │   guarantee         │   in local minima   │                     │
├───────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ MPC with bounds   │   ✅ Yes            │   ✅ Global opt     │   ❌ Slow          │
│                   │                     │   (if convex)       │   (heavy)           │
├───────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Collision Check   │   ❌ Reactive only  │   ✅ N/A            │   ✅ Fast           │
│ (stop on detect)  │   (no forward-look) │                     │                     │
├───────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ CBF               │   ✅ FORMAL        │   ❌ Can deadlock   │   ✅ Fast (QP)      │
│                   │   GUARANTEE        │   if goal behind    │   (~ms)             │
│                   │                     │   large obstacle    │                     │
└───────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    KEY INSIGHT: CBF IS A FILTER                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                     ┌─────────────────────────┐
                                     │   CBF is NOT a Planner  │
                                     └─────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
          ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
          │  Does NOT find  │         │  Does NOT plan  │         │  CAN deadlock   │
          │  global paths   │         │  around objects │         │  if goal behind │
          │                 │         │                 │         │  large obstacle │
          └─────────────────┘         └─────────────────┘         └─────────────────┘
                                                │
                                                ▼
                                     ┌─────────────────────────┐
                                     │ CBF is a REACTIVE       │
                                     │ SAFETY FILTER that      │
                                     │ modifies control AT     │
                                     │ EACH TIMESTEP           │
                                     └─────────────────────────┘
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          │                                           │
                          ▼                                           ▼
              ┌───────────────────────┐                   ┌───────────────────────┐
              │ Works well when goal  │                   │ May need higher-level │
              │ is reachable without  │                   │ planner for complex   │
              │ going "through"       │                   │ obstacle arrangements │
              │ obstacles             │                   │                       │
              └───────────────────────┘                   └───────────────────────┘
```

---

## 3.4 Comparison: CBF vs RTLS Collision Classifier

| Aspect | CBF (This Project) | RTLS Collision Classifier |
|--------|-------------------|---------------------------|
| **Type** | Mathematical function | Neural network |
| **Input** | State + obstacle geometry | Latent code z + obstacle params |
| **Output** | Safe control u_safe | P(collision) probability |
| **Training** | None (analytical) | Supervised learning (BCE) |
| **Guarantee** | Formal safety certificate | Statistical (learned) |
| **Computation** | QP solver (~1ms) | Neural net forward pass |
| **Adaptation** | Change h(x) formula | Retrain network |
| **Obstacle shape** | Must be known analytically | Can learn arbitrary shapes |

### Future Integration

The CBF approach could be integrated into RTLS:
1. **Replace classifier** with CBF-based safety evaluation
2. **CBF in latent space** for trajectory optimization
3. **Learn CBF** (as in the paper) instead of hand-crafting it

---

# APPENDIX: Quick Reference

## CBF Formula Sheet

### Single Integrator (ẋ = u)

```
h(x) = ||x - x_obs||² - r²
∇h(x) = 2(x - x_obs)
Constraint: ∇h(x) · u ≥ -α × h(x)
```

### Double Integrator (ẋ = v, v̇ = u)

```
h(x) = ||p - p_obs||² - r²
ḣ = 2(p - p_obs) · v
ψ = ḣ + λ × h
A = 2||v||² + 2λ(p - p_obs)·v
B = 2(p - p_obs)
Constraint: A + B·u ≥ -α × ψ
```

### QP Structure

```
minimize   ||u - u_ref||²
subject to ∇h_i · u ≥ -α(h_i)  ∀ obstacles
           ||u|| ≤ u_max
```

## Code Quick Start

```python
import numpy as np
from reference_controller import ProportionalController
from cbf_safety_filter import CBFSingleIntegrator, Obstacle

# Setup
obstacles = [Obstacle(center=np.array([2.0, 1.0]), radius=0.5, safety_margin=0.2)]
ref = ProportionalController(K=1.5, u_max=2.0)
cbf = CBFSingleIntegrator(obstacles, alpha=1.0, u_max=2.0)

# Simulation loop
x = np.array([0.0, 0.0])
goal = np.array([5.0, 1.5])
dt = 0.02

while np.linalg.norm(x - goal) > 0.1:
    u_ref = ref.compute(x, goal)           # Layer 1: Goal reaching
    u_safe, _ = cbf.safe_control(x, u_ref) # Layer 2: Safety filter
    x = x + u_safe * dt                     # Layer 3: Dynamics
```
