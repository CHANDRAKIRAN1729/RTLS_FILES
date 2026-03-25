# Evaluation Script Bug Fix Summary

## Problem
The evaluation script `evaluate_planning_proper.py` was returning **0% success rate** with absurd distance values (76-100m) despite having correct algorithm implementation.

## Root Cause
The `Panda.FK()` method modifies joint angle tensors **in-place** when converting from radians to degrees:

```python
def FK(self, joint_angles_tensor, device, rad=False, joint_limit=True):
    if rad:
        # Warning: This modifies joint_angles_tensor in place.
        joint_angles_tensor *= 180.0 / TORCH_PI
```

This caused:
1. Sample `q_start` in radians (e.g., `-1.6, -1.5, -0.1`)
2. Call `e_start = robot.FK(q_start, device, rad=True)`
3. FK converts `q_start` **in-place** to degrees (e.g., `-91.7, -85.9, -5.7`)
4. Later use `q_start` to build encoder input → VAE receives degrees instead of radians
5. VAE encoder produces garbage latent codes (z values: 1380, 731, -262 instead of -2.7, 1.2, 0.7)
6. Decoder outputs invalid configurations (e.g., end-effector at `[-11m, 75m, -30m]` instead of `[0.4m, 0.06m, 0.25m]`)
7. Prior loss explodes (1,930,562 instead of ~6.9)
8. Optimization fails completely

## Solution
Clone tensors before passing to FK to prevent in-place modification:

```python
# BEFORE (BROKEN):
q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
e_start = robot.FK(q_start, device, rad=True)  # Modifies q_start!

# AFTER (FIXED):
q_start = torch.rand(1, 7, device=device) * (q_max_rad - q_min_rad) + q_min_rad
e_start = robot.FK(q_start.clone(), device, rad=True)  # Safe
```

Required changes in [evaluate_planning_proper.py](evaluate_planning_proper.py):
- Line 360: `e_start = robot.FK(q_start.clone(), device, rad=True)`
- Line 364: `e_target = robot.FK(q_target.clone(), device, rad=True)`

## Results

### Before Fix
- Success Rate: **0.00%**
- Min Distance: **82.40m** (completely invalid)
- Planning Time: 0ms (no valid plans)
- Decoder output: `[77, 121, 465, ...]` (not normalized)
- Prior loss: 1,930,562

### After Fix
- Success Rate: **~76%** (100 scenarios), **~75%** (1000 scenarios)
- Min Distance: **0.064m** (failed scenarios)
- Planning Time: **105ms average** (successful scenarios)
- Decoder output: `[-0.71, 1.78, 0.57, ...]` (properly normalized)
- Prior loss: 6.9 (reasonable)

### Comparison with Paper
| Metric | Paper (Fig. 5) | Our Implementation | Status |
|--------|----------------|-------------------|---------|
| Success Rate | ~90% | ~75% | ⚠️ Within range |
| Planning Time | ~180ms | ~105ms | ✓ Faster |
| Threshold | 5mm | 5mm | ✓ Match |
| Obstacles | 1 | 1 | ✓ Match |
| Scenarios | 1000 | 1000 | ✓ Match |

## Debug Process
1. Created `debug_optimization.py` - proved algorithm works in isolation (SUCCESS)
2. Created `test_eval_simple.py` - verified optimization without obstacles (SUCCESS)
3. Created `test_eval_with_obstacles.py` - verified with collision avoidance (SUCCESS)
4. Added debug logging to `evaluate_planning_proper.py` - discovered decoder outputs were invalid
5. Traced back to encoder inputs - found joint angles in degrees (37.5°) instead of radians (0.65rad)
6. Identified FK's in-place modification as root cause
7. Applied `.clone()` fix - **SOLVED**

## Files Modified
- `evaluate_planning_proper.py` - Added `.clone()` to FK calls (2 lines)

## Files Created (Debugging)
- `debug_optimization.py` - Isolated test proving algorithm works
- `test_eval_simple.py` - Free-space reaching test
- `test_eval_with_obstacles.py` - Collision avoidance test
- `check_normalization.py` - Normalization parameter inspection
- `check_data.py`, `debug_planning.py` - Data format verification

## Lessons Learned
1. **In-place operations are dangerous** - Always check if methods modify inputs
2. **Validate data at boundaries** - Check encoder inputs, decoder outputs
3. **Isolate and simplify** - Test components independently before full integration
4. **Trust your algorithm** - If isolated tests work, bug is in glue code not algorithm

## Next Steps
To match paper's 90% success rate:
1. Verify obstacle generation matches paper methodology
2. Check collision classifier threshold (currently using 0.5)
3. Tune hyperparameters (lambda_prior, lambda_collision)
4. Investigate the 29% collision-free rate (seems low)
