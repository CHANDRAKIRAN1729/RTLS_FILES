# Control Barrier Functions (CBF) - Tutorial Implementation

This project implements **Control Barrier Functions** for safe robot navigation based on the paper:

> **"Learning Control Barrier Functions from Expert Demonstrations"**  
> by Alexander Robey, Haimin Hu, Lars Lindemann, et al.  
> [Paper Link](https://people.kth.se/~dimos/pdfs/learningcbfs-7.pdf)

## Table of Contents
1. [What are Control Barrier Functions?](#what-are-control-barrier-functions)
2. [Key Concepts Explained](#key-concepts-explained)
3. [Architecture Overview](#architecture-overview)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Mathematical Background](#mathematical-background)
7. [File Structure](#file-structure)

---

## What are Control Barrier Functions?

**CBFs are safety filters** that modify control inputs to guarantee that a system stays safe (e.g., avoids collisions) while staying as close as possible to a desired behavior.

### The Problem
You have a robot that needs to:
1. **Reach a goal** (navigation task)
2. **Avoid obstacles** (safety requirement)

### Traditional Approaches
- **Potential Fields**: Can get stuck in local minima
- **MPC with constraints**: Computationally expensive, may not find solutions
- **Pure collision checking**: Reactive, no formal guarantees

### CBF Approach
CBFs provide **formal mathematical guarantees** that the robot will never enter unsafe regions, while minimally modifying the control commands.

---

## Key Concepts Explained

### 1. Reference Controller (Nominal/Base Controller)

The **reference controller** is any controller that achieves your primary goal (e.g., reaching a target position) but **does NOT consider safety/obstacles**.

Example: A simple proportional controller
```python
u_ref = -K * (current_position - goal_position)
```

This controller drives straight toward the goal, potentially **through obstacles**.

### 2. CBF as a Safety Filter

The CBF sits **on top of** the reference controller and modifies the control input to ensure safety:

```
┌─────────────────────┐
│ Reference Controller │  → u_ref (desired control, may be unsafe)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  CBF Safety Filter  │  → Solves QP to find safe control
│  (Quadratic Program)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   u_safe (output)    │  → Safe control, close to u_ref
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│     Robot/System     │
└─────────────────────┘
```

### 3. The CBF Function h(x)

The barrier function `h(x)` defines what "safe" means:

| h(x) Value | Meaning |
|------------|---------|
| h(x) > 0   | Inside safe set (SAFE) |
| h(x) = 0   | On the boundary |
| h(x) < 0   | Outside safe set (UNSAFE) |

**For obstacle avoidance:**
```
h(x) = ||x - x_obstacle||² - r²
```
where `r` is the safety radius (obstacle radius + margin).

### 4. CBF Constraint

The key insight of CBFs is the **derivative constraint**:

```
ḣ(x) ≥ -α(h(x))
```

This ensures that:
- If you're safe (h > 0), you can approach the boundary slowly
- If you're on the boundary (h = 0), you cannot cross into unsafe regions
- The function α (class K function) controls how aggressively safety is enforced

### 5. CBF-QP (Quadratic Program)

The magic happens in the QP solver:

```
minimize ||u - u_ref||²           ← Stay close to reference
subject to: ∇h(x)·f(x,u) ≥ -α(h)  ← Safety constraint
```

This finds the **closest safe control** to the desired control.

---

## Architecture Overview

```
cbf_implementation/
├── reference_controller.py   # Goal-reaching controllers (ignore safety)
├── cbf_safety_filter.py      # CBF implementation (enforces safety)
├── simulation.py             # Demo simulations
└── README.md                 # This file
```

### Layer 1: Reference Controller
```python
# Simple goal reaching - no safety consideration
from reference_controller import ProportionalController

ref_ctrl = ProportionalController(K=1.0)
u_ref = ref_ctrl.compute(current_pos, goal_pos)
```

### Layer 2: CBF Safety Filter  
```python
# Make the control safe
from cbf_safety_filter import CBFSingleIntegrator, Obstacle

obstacles = [Obstacle(center=np.array([2,2]), radius=0.5)]
cbf = CBFSingleIntegrator(obstacles, alpha=1.0)

u_safe, info = cbf.safe_control(current_pos, u_ref)
# u_safe is guaranteed to avoid obstacles!
```

---

## Installation

```bash
# Required packages
pip install numpy matplotlib cvxpy

# Optional for better QP solving
pip install osqp
```

## Usage

### Run the Demo
```bash
cd cbf_implementation
python simulation.py
```

### Basic Usage Example

```python
import numpy as np
from reference_controller import ProportionalController
from cbf_safety_filter import CBFSingleIntegrator, Obstacle

# Setup
obstacles = [
    Obstacle(center=np.array([2.0, 1.5]), radius=0.5, safety_margin=0.1),
]

ref_controller = ProportionalController(K=1.0)
cbf = CBFSingleIntegrator(obstacles, alpha=1.0, u_max=2.0)

# Simulation loop
position = np.array([0.0, 0.0])
goal = np.array([5.0, 3.0])
dt = 0.05

while np.linalg.norm(position - goal) > 0.1:
    # Step 1: Get reference control (goal reaching)
    u_ref = ref_controller.compute(position, goal)
    
    # Step 2: Apply CBF safety filter
    u_safe, info = cbf.safe_control(position, u_ref)
    
    # Step 3: Apply safe control
    position = position + u_safe * dt
    
    print(f"Position: {position}, CBF values: {info['cbf_values']}")
```

---

## Mathematical Background

### Control-Affine Systems

CBFs work with systems of the form:
```
ẋ = f(x) + g(x)·u
```

| System Type | f(x) | g(x) | Control u |
|-------------|------|------|-----------|
| Single Integrator | 0 | I (identity) | Velocity |
| Double Integrator | [v; 0] | [0; I] | Acceleration |

### CBF Condition (Formal)

For a function h(x) to be a valid CBF, there must exist a control u such that:

```
Lf h(x) + Lg h(x) · u ≥ -α(h(x))
```

Where:
- `Lf h(x) = ∇h(x) · f(x)` (Lie derivative along f)
- `Lg h(x) = ∇h(x) · g(x)` (Lie derivative along g)
- `α` is an extended class K function (e.g., α(h) = αh)

### Single Integrator (Our Implementation)

For ẋ = u (velocity control):
- f(x) = 0, g(x) = I
- h(x) = ||x - x_obs||² - r²
- ∇h(x) = 2(x - x_obs)
- Lf h = 0
- Lg h = 2(x - x_obs)

**CBF Constraint:**
```
2(x - x_obs) · u ≥ -α·h(x)
```

### Double Integrator (Advanced)

For ẋ = v, v̇ = u (acceleration control):

Since h(x) depends only on position but control affects acceleration (relative degree 2), we use the **exponential CBF**:

```
ψ(x) = ḣ(x) + λ·h(x)
```

Then enforce: ψ̇(x) ≥ -α(ψ(x))

---

## File Structure

### `reference_controller.py`
Contains goal-reaching controllers that **ignore safety**:
- `ProportionalController`: Simple P-controller for single integrator
- `PotentialFieldController`: APF method (for comparison)
- `DoubleIntegratorPDController`: PD controller for double integrator

### `cbf_safety_filter.py`
Core CBF implementation:
- `Obstacle`: Circular obstacle dataclass
- `CBFSingleIntegrator`: CBF for velocity control systems
- `CBFDoubleIntegrator`: CBF for acceleration control systems
- `MultiObstacleCBF`: Unified interface for multiple obstacles

### `simulation.py`
Demonstration scripts:
- Single integrator demo
- Double integrator demo
- Challenging narrow passage scenario

---

## Key Takeaways

1. **CBF is a FILTER**, not a planner. It works on top of any reference controller.

2. **Minimal modification**: CBF finds the control closest to your desired control that is still safe.

3. **Formal guarantees**: Unlike heuristic methods, CBF provides mathematical safety certificates.

4. **Real-time**: CBF-QP is a convex optimization, solvable in milliseconds.

5. **Modular**: You can swap out the reference controller without changing the CBF.

---

## Next Steps: Integration with RTLS

After understanding this standalone CBF implementation, the next step is to integrate it into the `Reaching_Through_Latent_Space` codebase:

1. **Replace Binary Collision Classifier** with CBF-based safety evaluation
2. **Use CBF in latent space** for trajectory optimization
3. **Learn the CBF** (as in the paper) instead of hand-crafting it

---

## References

1. Robey et al., "Learning Control Barrier Functions from Expert Demonstrations"
2. Ames et al., "Control Barrier Functions: Theory and Applications"
3. Ames et al., "Control Barrier Function Based Quadratic Programs for Safety Critical Systems"
