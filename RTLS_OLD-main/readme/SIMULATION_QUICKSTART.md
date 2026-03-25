# Quick Start: MoveIt Simulation

## ⚡ Fastest Path (Ubuntu 22.04)

**Install ROS 2 Humble + MoveIt:**
```bash
sudo apt update
sudo apt install ros-humble-desktop-full ros-humble-moveit ros-humble-moveit-resources-panda-moveit-config
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**Run simulation:**
```bash
# Terminal 1: Launch MoveIt
ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# Terminal 2: Run
cd "/home/chandrakiran/Projects/Reaching Through Latent Space"
source /opt/ros/humble/setup.bash
python3 src/simulate_with_ros2.py \
  --checkpoint model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 10
```

**Complete guide:** [ROS2_SETUP.md](ROS2_SETUP.md)

---

## Alternative: Docker with ROS 1 Noetic

**For Ubuntu 20.04 compatibility:** See [SIMULATION_SETUP.md](SIMULATION_SETUP.md)

```bash
# Terminal 1: Launch MoveIt (if ROS available)
roslaunch panda_moveit_config demo.launch

# Terminal 2: Run simulation
cd "/home/chandrakiran/Projects/Reaching Through Latent Space"
./launch_simulation.sh
```

## What This Does

1. **Generates path plans** using your trained VAE + collision classifier
2. **Visualizes in RViz** - watch robot navigate around obstacles
3. **Validates with MoveIt** - ground truth collision checking
4. **Reports metrics**:
   - Planner success rate (~74% expected)
   - MoveIt validation rate (~100% expected)
   - Planning time per scenario

## Key Features

### Full MoveIt Integration
- Uses same collision checker as training data (consistent labels)
- Visualizes robot, obstacles, and trajectories in RViz
- Validates every waypoint for collisions

### Matches Paper Methodology
- 1000 scenarios for full evaluation
- Cylindrical obstacles placed between start/goal
- 5mm success threshold
- 300 max optimization steps

### Three Modes
1. **With visualization**: See robot animate through paths (slow)
2. **No animation**: Show only start/end states (medium)
3. **No visualization**: Only compute statistics (fast)

## Example Output

```
--- Scenario 42/100 ---
Start: [-0.234, 0.567, 0.412]
Target: [0.456, -0.123, 0.789]
Distance: 0.893m
Obstacle: (0.112, 0.223), h=0.75m, r=0.089m
Planning...
✓ Planner SUCCESS: 98.3ms, 47 steps
Animating path in RViz...
Validating with MoveIt collision checker...
✓ MoveIt VALIDATED: Path is collision-free

==================================================
Progress: 100/100
Planner success rate: 74.2%
MoveIt validation rate: 100.0%
==================================================
```

## Files Created

1. **simulate_in_moveit.py** - Main simulation script (600+ lines)
   - MoveItEnvironment: Scene setup, collision checking
   - LatentSpacePlanner: Path planning algorithm
   - Visualization and validation pipeline

2. **launch_simulation.sh** - Interactive launcher
   - Auto-detects ROS/MoveIt
   - Provides preset configurations
   - Handles all arguments

3. **SIMULATION_SETUP.md** - Complete setup guide
   - ROS/MoveIt installation
   - Troubleshooting
   - Advanced usage

## Without ROS

If ROS not installed, script runs in simulation-only mode:
- ✓ Plans paths with learned model
- ✓ Reports planner success rate
- ✗ No RViz visualization
- ✗ No MoveIt validation

## Next Steps

### 1. Quick Test (10 scenarios)
```bash
./launch_simulation.sh  # Choose option 1
```

### 2. Full Evaluation (1000 scenarios)
```bash
./launch_simulation.sh  # Choose option 4
```

### 3. Custom Configuration
```bash
python src/simulate_in_moveit.py \
  --checkpoint model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 50 \
  --num_obstacles 2 \
  --no_animation
```

## Why This Matters

The paper uses MoveIt for:
1. **Training labels**: Collision classifier trained on MoveIt collision checks
2. **Evaluation**: All success metrics validated with MoveIt
3. **Ground truth**: MoveIt geometric checker is the gold standard

**Using a different simulator would invalidate results** due to label mismatch.

This script ensures your evaluation matches the paper's methodology exactly.
