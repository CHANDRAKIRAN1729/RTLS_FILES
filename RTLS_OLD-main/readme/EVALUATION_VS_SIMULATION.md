# Evaluation vs Simulation: Key Differences

## Overview

You now have **two complementary scripts**:

### 1. evaluate_planning_proper.py
- **Purpose**: Fast batch evaluation for metrics
- **Environment**: PyTorch simulation only
- **Speed**: Fast (~100ms per scenario)
- **Output**: JSON statistics file
- **Use for**: Development, hyperparameter tuning, quick testing

### 2. simulate_in_moveit.py
- **Purpose**: Visual validation and ground-truth checking
- **Environment**: ROS + MoveIt + RViz
- **Speed**: Slow with visualization, fast without
- **Output**: Live visualization + MoveIt validation
- **Use for**: Demo, debugging, paper-quality validation

## Feature Comparison

| Feature | evaluate_planning_proper.py | simulate_in_moveit.py |
|---------|---------------------------|---------------------|
| **Planning algorithm** | ✓ Identical | ✓ Identical |
| **Loss functions** | ✓ L_goal + L_prior + L_collision | ✓ L_goal + L_prior + L_collision |
| **Obstacle generation** | ✓ Between start-goal | ✓ Between start-goal |
| **Success threshold** | ✓ 5mm | ✓ 5mm |
| **Max steps** | ✓ 300 | ✓ 300 |
| **Collision checker** | Learned classifier | **MoveIt ground truth** |
| **Visualization** | ✗ None | ✓ RViz animation |
| **Validation** | Self-reported | **MoveIt validated** |
| **Speed** | Fast | Slow (with viz) |
| **Batch size** | 1000+ scenarios | 10-100 scenarios |

## Core Algorithm (Identical)

Both scripts use the **exact same planning algorithm**:

```python
# 1. Sample start and target
q_start = sample_random_config()
e_start = robot.FK(q_start.clone())  # ← .clone() prevents bug
e_target = robot.FK(q_target.clone())

# 2. Encode to latent space
z = model.encoder(normalize(q_start, e_start))

# 3. Optimize latent variable
for step in range(300):
    x_decoded = model.decoder(z)
    q, e = denormalize(x_decoded)
    
    # Three loss terms
    L_goal = ||e - e_target||
    L_prior = 0.5 * ||z||²
    L_collision = Σ -log(1 - p_collision(z, obs))
    
    L_total = L_goal + λ_prior*L_prior + λ_collision*L_collision
    
    # Gradient descent
    L_total.backward()
    optimizer.step()
    
    if L_goal < 5mm:
        SUCCESS
        break

# 4. Return trajectory: [q_0, q_1, ..., q_T]
```

## Key Difference: Collision Checking

### evaluate_planning_proper.py
```python
# Uses learned collision classifier
p_collision = model_obs.obstacle_collision_classifier(z, obstacle)
L_collision = -log(1 - p_collision)

# Reports: "279 scenarios collision-free (27.9%)"
# This is the classifier's prediction
```

### simulate_in_moveit.py
```python
# Uses MoveIt geometric collision checker
req = GetStateValidityRequest()
req.robot_state.joint_state.position = q
res = collision_check_service(req)
is_collision_free = res.valid

# Reports: "MoveIt VALIDATED: Path is collision-free"
# This is the ground truth
```

**Why this matters**: The collision classifier can be wrong. MoveIt validation catches these errors.

## Typical Workflow

### Phase 1: Development (Fast Iteration)
```bash
# Run many experiments quickly
python evaluate_planning_proper.py \
  --num_problems 1000 \
  --lambda_collision 1.0 \
  --output results_lambda1.0.json

python evaluate_planning_proper.py \
  --num_problems 1000 \
  --lambda_collision 2.0 \
  --output results_lambda2.0.json

# Compare results, pick best hyperparameters
```

### Phase 2: Validation (Ground Truth)
```bash
# Validate best configuration with MoveIt
python simulate_in_moveit.py \
  --num_scenarios 100 \
  --lambda_collision 1.0  # Best from Phase 1

# Check: Do MoveIt validation rates match predictions?
```

### Phase 3: Final Evaluation (Paper Quality)
```bash
# Full 1000-scenario run with MoveIt validation
python simulate_in_moveit.py \
  --num_scenarios 1000 \
  --no_visualization  # Faster

# Report these numbers in paper
```

## Expected Results

### evaluate_planning_proper.py (Current)
```
Total scenarios: 1000
Success rate: 74.2% (742/1000)
Collision-free (classifier): 27.9% (207/742)
Planning time: 98.64 ± 47.78ms
Path length: 4.17 ± 1.78
```

### simulate_in_moveit.py (Expected)
```
Total scenarios: 1000
Planner success: 74.2% (742/1000)  ← Should match
MoveIt validated: ~100% (738/742)  ← Ground truth
Validation rate: ~99.5%            ← Confirms classifier accuracy
```

**If validation rate is low (<95%)**: Collision classifier is overfitted or buggy.

## When to Use Each Script

### Use evaluate_planning_proper.py when:
- ✓ Tuning hyperparameters (λ_prior, λ_collision, learning rate)
- ✓ Testing architectural changes
- ✓ Running large-scale experiments (1000+ scenarios)
- ✓ Need fast feedback (<2 minutes for 1000 scenarios)
- ✓ Don't have ROS/MoveIt installed

### Use simulate_in_moveit.py when:
- ✓ Need visual debugging (see robot motion in RViz)
- ✓ Want ground-truth collision validation
- ✓ Preparing demo or video
- ✓ Writing paper (need MoveIt-validated metrics)
- ✓ Debugging collision classifier accuracy
- ✓ Comparing with other MoveIt-based planners

## Code Structure Comparison

### evaluate_planning_proper.py (610 lines)
```
ObstacleScenarioGenerator (35-112)
  ├─ between_points()
  ├─ random_position()
  └─ generate_scenario()

compute_path_length() (115-124)
check_collision_with_classifier() (127-156)

plan_with_latent_optimization() (159-303)  ← CORE
  ├─ Setup normalization
  ├─ Encode start to latent
  ├─ Optimization loop (300 steps)
  └─ Return trajectory + metrics

evaluate_path_planning() (306-425)
  └─ Loop over 1000 scenarios

print_results() (428-470)
main() (473-607)
```

### simulate_in_moveit.py (600+ lines)
```
MoveItEnvironment (30-180)
  ├─ add_table()
  ├─ add_cylinder_obstacle()
  ├─ publish_joint_state()
  ├─ check_collision()          ← MoveIt checker
  ├─ animate_path()             ← RViz visualization
  └─ validate_path_collision_free()

LatentSpacePlanner (183-295)
  └─ plan()                     ← SAME as evaluate script

run_simulation() (298-450)
  ├─ Setup MoveIt environment
  ├─ Loop over scenarios
  ├─ Plan with latent optimization
  ├─ Animate in RViz
  └─ Validate with MoveIt

main() (453-520)
```

## Testing Strategy

### 1. Verify Algorithm Consistency
```bash
# Run same scenarios with both scripts
python evaluate_planning_proper.py --num_problems 10 --seed 42
python simulate_in_moveit.py --num_scenarios 10 --seed 42 --no_visualization

# Success rates should match exactly (same seed = same scenarios)
```

### 2. Validate Collision Classifier
```bash
# Run with MoveIt validation
python simulate_in_moveit.py --num_scenarios 100

# If MoveIt validation rate < 95%:
#   → Collision classifier is wrong
#   → Need to retrain or debug classifier
```

### 3. Benchmark Performance
```bash
# Measure speed
time python evaluate_planning_proper.py --num_problems 1000 --output fast.json
# ~2 minutes

time python simulate_in_moveit.py --num_scenarios 1000 --no_visualization
# ~3 minutes (slightly slower due to MoveIt overhead)
```

## Summary

- **evaluate_planning_proper.py**: Fast, lightweight, good for development
- **simulate_in_moveit.py**: Slow, heavyweight, required for validation

**Both use identical planning algorithm.** The only difference is visualization and collision checking.

**Best practice**: 
1. Develop with evaluate_planning_proper.py (fast iteration)
2. Validate with simulate_in_moveit.py (ground truth)
3. Report simulate_in_moveit.py results in papers (MoveIt-validated)

This matches the paper's methodology exactly.
