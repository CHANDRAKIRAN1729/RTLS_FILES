# Proper Evaluation Script Usage Guide

## Overview

The new `evaluate_planning_proper.py` script implements the correct evaluation methodology from the "Reaching Through Latent Space" paper.

## Key Differences from Old Script

| Aspect | Old Script (Wrong) | New Script (Correct) |
|--------|-------------------|---------------------|
| **Test scenarios** | 100 | 1000 (paper standard) |
| **Environment** | free_space (no obstacles) | Obstacles between start-goal |
| **Loss terms** | Only L_goal | L_goal + L_collision + L_prior |
| **Collision classifier** | NOT used | Used during planning |
| **Success threshold** | 0.05m (50mm) | 0.005m (5mm) - paper standard |
| **Obstacles** | None | 1-3 obstacles per scenario |

## Requirements

### Models Needed

1. **VAE checkpoint**: Trained on free_space data
   - Path: `../model_params/panda_10k/model.ckpt-003000.pt`
   
2. **Collision classifier checkpoint**: Trained on collision data
   - Path: `../model_params/panda_10k/model_obs.ckpt-XXXXX.pt`
   - **⚠️ YOU NEED TO TRAIN THIS!** (see instructions below)

3. **Config files**:
   - VAE config: `../model_params/panda_10k/20260105_210436817-runcmd.json`
   - Collision classifier config: `../model_params/panda_10k/XXXXX-runcmd.json`

## Training the Collision Classifier (Required!)

You currently only have the VAE trained. You need to train the collision classifier:

```bash
# Navigate to src directory
cd "/home/chandrakiran/Projects/Reaching Through Latent Space/src"

# Train collision classifier using the VAE you already trained
python train_vae_obs.py \
    --c ../config/vae_obs_config/panda_10k.yaml \
    --pretrained_checkpoint_path ../model_params/panda_10k/model.ckpt-003000.pt \
    --vae_run_cmd_path ../model_params/panda_10k/20260105_210436817-runcmd.json
```

This will:
- Load your trained VAE encoder (frozen)
- Train the collision classifier on collision_100k_train.dat
- Save checkpoints to `../model_params/panda_10k/snapshots_obs/`
- Take approximately 20-40 hours on GPU

## Running the Evaluation

### Example 1: Basic evaluation (1 obstacle, paper settings)

```bash
python evaluate_planning_proper.py \
    --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
    --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-XXXXX.pt \
    --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
    --config_obs ../model_params/panda_10k/XXXXX-runcmd.json \
    --num_problems 1000 \
    --num_obstacles 1 \
    --success_threshold 0.005 \
    --max_steps 300 \
    --lambda_prior 0.01 \
    --lambda_collision 1.0 \
    --output ../model_params/panda_10k/evaluation_results.json
```

### Example 2: Quick test (100 scenarios for debugging)

```bash
python evaluate_planning_proper.py \
    --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
    --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-XXXXX.pt \
    --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
    --config_obs ../model_params/panda_10k/XXXXX-runcmd.json \
    --num_problems 100 \
    --num_obstacles 1 \
    --output ../model_params/panda_10k/test_results.json
```

### Example 3: Multiple obstacles (harder scenarios)

```bash
python evaluate_planning_proper.py \
    --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
    --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-XXXXX.pt \
    --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
    --config_obs ../model_params/panda_10k/XXXXX-runcmd.json \
    --num_problems 1000 \
    --num_obstacles 3 \
    --output ../model_params/panda_10k/evaluation_3obs_results.json
```

## Expected Results (from paper)

### Free Space Reaching (no obstacles)
- Success rate: > 90% (at 5mm threshold)
- Planning time: ~180ms

### With Obstacles (1 obstacle)
- Success rate: ~85% (paper reports 85.8%)
- Planning time: ~180ms
- Collision-free rate: > 95%

## Output Files

The evaluation produces two files:

1. **Summary file** (`evaluation_results.json`):
   - Overall statistics
   - Success rates
   - Planning times
   - Collision rates

2. **Detailed file** (`evaluation_results_detailed.json`):
   - Per-scenario results
   - Obstacle positions
   - Decoded paths
   - Collision probabilities

## Interpreting Results

### Success Rate
- **High (>90%)**: Model is working well
- **Medium (70-90%)**: Acceptable but could improve
- **Low (<70%)**: Something is wrong (check model training)

### Collision-Free Rate
- Should be > 90% if collision classifier is working
- If low, check:
  - Collision classifier training
  - Lambda_collision weight (try increasing)
  - Max steps (may need more iterations)

### Planning Time
- Paper reports ~180ms per scenario
- Your results may vary based on hardware
- Faster is better, but success rate is more important

## Ablation Studies (for paper analysis)

### Effect of Prior Loss
```bash
# Without prior loss
python evaluate_planning_proper.py ... --lambda_prior 0.0

# With prior loss (paper default)
python evaluate_planning_proper.py ... --lambda_prior 0.01
```

Expected: Prior loss improves success rate by ~10-15%

### Effect of Collision Loss
```bash
# Without collision loss (will collide!)
python evaluate_planning_proper.py ... --lambda_collision 0.0

# With collision loss (paper default)
python evaluate_planning_proper.py ... --lambda_collision 1.0
```

Expected: Without collision loss, collision rate should be very high

## Troubleshooting

### Error: "Collision classifier checkpoint not found"
**Solution**: You need to train the collision classifier first (see above)

### Error: "Config file not found"
**Solution**: Make sure the collision classifier training completed and saved the config

### Low success rate (<50%)
**Possible causes**:
1. VAE not trained well (check sample_consistency metric)
2. Collision classifier not trained
3. Lambda weights too high/low
4. Success threshold too strict (try 0.01m instead of 0.005m)

### High collision rate (>20%)
**Possible causes**:
1. Lambda_collision too low (try 10.0)
2. Collision classifier not working (check training accuracy)
3. Not enough optimization steps (try --max_steps 500)

## What's Still Missing (MoveIt Integration)

The current script uses the collision classifier to check collisions. For complete paper replication, you would need:

1. **MoveIt environment** integration for ground truth
2. **Trajectory execution** in simulation
3. **Physical collision verification**

However, the collision classifier was trained on MoveIt labels, so using it for evaluation is a reasonable approximation.

## Next Steps

1. ✅ Train the collision classifier
2. ✅ Run evaluation with 1000 scenarios
3. ✅ Compare results with paper
4. ✅ Run ablation studies
5. ❓ (Optional) Integrate MoveIt for ground truth verification
