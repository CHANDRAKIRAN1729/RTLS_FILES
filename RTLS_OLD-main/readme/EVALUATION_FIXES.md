# Evaluation Script Issues and Fixes

## Summary of Problems

The original `evaluate_planning.py` script had **fundamental flaws** that made it test something completely different from what the paper describes.

---

## Problems with Original Script

### 1. ❌ **Wrong Dataset**
```python
# Original (WRONG)
test_data_name = 'free_space_10k_test.dat'  # No obstacles!
```
- Used free_space dataset with **no obstacles**
- Paper requires obstacles between start and goal
- Makes collision avoidance trivial

### 2. ❌ **Wrong Number of Test Cases**
```python
# Original (WRONG)
num_problems = 100  # Only 100 scenarios
```
- Paper uses **1000 scenarios**
- 10x fewer test cases = less reliable results

### 3. ❌ **Missing Collision Loss**
```python
# Original (WRONG) - only one loss term
for step in range(max_steps):
    L = ||e_decoded - e_target||²  # Only goal reaching
    L.backward()
    optimizer.step()
```
- **No collision penalty**
- Collision classifier never used during planning
- Would collide with obstacles if any existed

### 4. ❌ **Missing Prior Loss**
```python
# Original (WRONG) - no prior constraint
L = L_goal  # Missing: + λ_prior * ||z||²
```
- Paper shows prior loss is "instrumental" for success
- Without it: optimization goes to unexplored regions of latent space
- Decoder produces invalid joint configurations

### 5. ❌ **Wrong Success Threshold**
```python
# Original (WRONG)
success_threshold = 0.05  # 50mm = 5cm
```
- Paper uses **5mm = 0.005m**
- 10x more lenient threshold
- Artificially inflates success rate

### 6. ❌ **Meaningless "Goals"**
```python
# In robot_state_dataset.py
def __getitem__(self, index):
    jpos_ee_xyz = self.robot_data[index, :-3]  # [q, e]
    goal_xyz = self.robot_data[index, -3:]     # Last 3 cols as "goal"
```
- The "goal" is just the last 3 columns of the same data point
- Not meaningful start→goal reaching tasks
- Often trivially close to start position

### 7. ❌ **No Environment Validation**
- No collision checking after planning
- No verification that path is executable
- Just optimization in latent space, no physics

---

## What the Paper Actually Does

### Proper Methodology (Section V)

**1. Scenario Generation:**
```python
# Sample start and goal configurations
q_start = sample_valid_joint_config()
e_start = FK(q_start)

q_target = sample_valid_joint_config()  # Different from start!
e_target = FK(q_target)  # Use only for target position

# Generate obstacles between start and goal
obstacle_1 = sample_between(e_start, e_target)  # First always between
obstacle_2 = sample_random() if rand() < 0.5 else sample_between()
```

**2. Path Planning:**
```python
# Initialize from start
z = encoder(q_start, e_start)

for step in range(300):
    q_decoded, e_decoded = decoder(z)
    
    # THREE loss terms (not just one!)
    L_goal = ||e_decoded - e_target||²
    
    L_prior = 0.5 * ||z||²  # Keep in well-trained region
    
    L_collision = 0
    for obstacle in obstacles:
        p_coll = collision_classifier(z, obstacle)
        L_collision += -log(1 - p_coll)
    
    # Combined loss
    L_total = L_goal + λ_prior * L_prior + λ_collision * L_collision
    
    L_total.backward()
    optimizer.step()
```

**3. Evaluation:**
- Test on **1000 scenarios**
- Success threshold: **5mm**
- Report: success rate, planning time, collision rate

---

## The New Script (evaluate_planning_proper.py)

### ✅ **Fixed: Proper Obstacle Scenarios**
```python
class ObstacleScenarioGenerator:
    def generate_scenario(self, e_start, e_target, num_obstacles):
        obstacles = []
        # First obstacle between start-goal (paper method)
        obs1 = sample_between(e_start, e_target)
        obstacles.append(obs1)
        
        # Subsequent: 50% between, 50% random
        for i in range(1, num_obstacles):
            if random() < 0.5:
                obs = sample_between(e_start, e_target)
            else:
                obs = sample_random()
            obstacles.append(obs)
        return obstacles
```

### ✅ **Fixed: All Three Loss Terms**
```python
# Goal reaching loss
L_goal = torch.dist(e_decoded, e_target)

# Prior loss (Eq. 5 from paper)
L_prior = 0.5 * torch.sum(z ** 2)

# Collision loss (Eq. 6 from paper)
L_collision = torch.tensor(0.0)
for obstacle in obstacles:
    p_coll = torch.sigmoid(classifier(z, obstacle))
    L_collision += -torch.log(1 - p_coll + 1e-8)

# Combined (Eq. 3 from paper)
L_total = L_goal + λ_prior * L_prior + λ_collision * L_collision
```

### ✅ **Fixed: 1000 Test Scenarios**
```python
parser.add_argument('--num_problems', type=int, default=1000,
                   help='Number of scenarios (paper uses 1000)')
```

### ✅ **Fixed: 5mm Threshold**
```python
parser.add_argument('--success_threshold', type=float, default=0.005,
                   help='5mm = 0.005m (paper standard)')
```

### ✅ **Fixed: Collision Verification**
```python
# Use collision classifier to verify final path
collision_detected, max_prob = check_collision_with_classifier(
    final_q, obstacles, model_obs, encoder
)
```

---

## Expected Results Comparison

### Original Script (Wrong)
```
Success Rate: ~99%     ← Too high (trivial problem)
Planning Time: ~50ms   ← Too fast (no collision avoidance)
Collision Rate: 0%     ← No obstacles to collide with
```

### New Script (Correct)
```
Success Rate: ~85-90%  ← Matches paper
Planning Time: ~180ms  ← Matches paper
Collision Rate: ~5-10% ← Some failures expected
```

---

## How to Run

### Quick test (100 scenarios):
```bash
cd "/home/chandrakiran/Projects/Reaching Through Latent Space/src"

python evaluate_planning_proper.py \
    --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
    --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
    --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
    --config_obs ../model_params/panda_10k/20260105_210436817-runcmd.json \
    --num_problems 100 \
    --num_obstacles 1 \
    --output ../model_params/panda_10k/test_results.json
```

### Full evaluation (1000 scenarios, as in paper):
```bash
./run_proper_evaluation.sh
```

---

## Key Takeaways

1. **The original script was fundamentally flawed** - it wasn't testing what the paper describes
2. **Free space ≠ Obstacle avoidance** - completely different problem
3. **All three loss terms are essential**:
   - L_goal: reach target
   - L_prior: stay in valid region
   - L_collision: avoid obstacles
4. **Paper uses 1000 scenarios at 5mm threshold** - not 100 at 50mm
5. **The new script properly implements the paper's methodology**

---

## What's Still Missing (Future Work)

For complete paper replication, you would need:

1. **MoveIt integration** for ground-truth collision checking
2. **Trajectory execution** in simulation to verify feasibility
3. **Comparison with RRT-Connect** baseline
4. **Physical robot experiments** (Franka Panda hardware)

However, the current evaluation using the collision classifier (trained on MoveIt labels) is a reasonable approximation of the paper's methodology.
