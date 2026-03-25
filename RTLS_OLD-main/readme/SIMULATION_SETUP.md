# MoveIt Simulation Setup Guide

This guide explains how to set up and run end-to-end simulations with RViz visualization and MoveIt validation.

## Prerequisites

### ⚠️ Ubuntu Version Check

**Check your Ubuntu version first:**
```bash
lsb_release -a
```

**ROS Noetic (required for this script) officially supports Ubuntu 20.04 only.**

If you have Ubuntu 22.04, see "Ubuntu 22.04 Solutions" section below.

---

### For Ubuntu 20.04 (Direct Installation) ✅

#### 1. Install ROS Noetic
```bash
sudo apt update
sudo apt install ros-noetic-desktop-full

# Source ROS
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install MoveIt
sudo apt install ros-noetic-moveit

# Install Franka Panda MoveIt config
sudo apt install ros-noetic-panda-moveit-config

```

#### 2. Install MoveIt
```bash
sudo apt install ros-noetic-moveit
```

#### 3. Install Franka Panda MoveIt Config
```bash
sudo apt install ros-noetic-panda-moveit-config
```

#### 4. Install Python Dependencies
```bash
sudo apt install python3-rospy ros-noetic-moveit-commander
pip install rospkg catkin_pkg
```

#### 5. Verify Installation
```bash
cd "/home/chandrakiran/Projects/Reaching Through Latent Space"
./check_ros_setup.sh
```

---

### For Ubuntu 18.04 (Direct Installation)

#### 1. Install ROS Melodic
```bash
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### 2-4. Follow same steps as Ubuntu 20.04
Replace `ros-noetic-*` with `ros-melodic-*` in all commands.

---

### For Ubuntu 22.04 (Docker Solution) ⚠️

**ROS Noetic is NOT officially supported on Ubuntu 22.04.**

Use Docker to run Ubuntu 20.04 with ROS Noetic:

#### 1. Install Docker
```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Log out and back in for group changes
```

#### 2. Enable X11 for GUI (RViz)
```bash
# Allow Docker to access display
xhost +local:docker
```

#### 3. Create Dockerfile
```bash
cd "/home/chandrakiran/Projects/Reaching Through Latent Space"
cat > Dockerfile << 'EOF'
FROM ros:noetic-robot-focal

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-moveit \
    ros-noetic-panda-moveit-config \
    ros-noetic-moveit-commander \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch numpy rospkg catkin_pkg

# Setup workspace
WORKDIR /workspace
EOF
```

#### 4. Build Docker Image
```bash
docker build -t ros-noetic-simulation .
```

#### 5. Run Container with GUI Support
```bash
docker run -it --rm \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$(pwd):/workspace" \
  --network=host \
  ros-noetic-simulation \
  bash

# Inside container:
source /opt/ros/noetic/setup.bash
roslaunch panda_moveit_config demo.launch
```

#### 6. Run Simulation (in another terminal)
```bash
# Get container ID
docker ps

# Attach to container
docker exec -it <container_id> bash

# Inside container:
source /opt/ros/noetic/setup.bash
cd /workspace
./launch_simulation.sh
```

---

### Python Dependencies
```bash
# Install ROS Python packages
pip install rospkg catkin_pkg

# Verify rospy available
python3 -c "import rospy; print('ROS Python OK')"
```

## Usage

### Step 1: Launch MoveIt (Terminal 1)
```bash
# Start RViz with Panda robot
roslaunch panda_moveit_config demo.launch

# You should see:
# - RViz window with Panda robot
# - Planning scene (empty)
# - Motion planning plugin
```

### Step 2: Run Simulation (Terminal 2)
```bash
cd "/home/chandrakiran/Projects/Reaching Through Latent Space/src"

# Basic simulation (10 scenarios, with visualization)
python simulate_in_moveit.py \
  --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 10 \
  --num_obstacles 1

# Full evaluation (100+ scenarios)
python simulate_in_moveit.py \
  --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 100 \
  --num_obstacles 1

# Without visualization (faster)
python simulate_in_moveit.py \
  --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 100 \
  --no_visualization

# Skip animation (show only final result)
python simulate_in_moveit.py \
  --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 50 \
  --no_animation
```

## What You'll See

### In RViz
1. **Robot**: Franka Panda manipulator
2. **Table**: Gray box at z=0
3. **Obstacles**: Cyan cylinders in workspace
4. **Motion**: Robot animating through planned trajectory
5. **Colors**: 
   - Green = collision-free
   - Red = in collision

### In Terminal
```
--- Scenario 1/10 ---
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
Progress: 10/10
Planner success rate: 74.0%
MoveIt validation rate: 100.0%
==================================================
```

## Validation Metrics

The script reports two key metrics:

### 1. Planner Success Rate
- **What**: Percentage of scenarios where learned planner reaches goal
- **Threshold**: 5mm (0.005m) distance to target
- **Expected**: ~74% (matches evaluate_planning_proper.py)

### 2. MoveIt Validation Rate
- **What**: Percentage of planner successes that MoveIt confirms are collision-free
- **Method**: Uses MoveIt's geometric collision checker (ground truth)
- **Expected**: ~100% (our collision classifier is trained on MoveIt labels)

**Important**: If MoveIt validation rate is low (<95%), it indicates:
- Bug in FK or collision checking
- Mismatch between training and test environments
- Overfitting of collision classifier

## Troubleshooting

### Error: "ROS not available"
```bash
# Check ROS installation
rosversion -d  # Should show 'noetic' or 'melodic'

# Source ROS
source /opt/ros/noetic/setup.bash

# Verify Python can import rospy
python3 -c "import rospy"
```

### Error: "Could not initialize MoveIt"
```bash
# Make sure MoveIt demo is running
roslaunch panda_moveit_config demo.launch

# Check ROS master
rostopic list  # Should show many topics

# Check planning scene
rostopic echo /planning_scene -n 1
```

### Error: "check_state_validity service not available"
```bash
# Restart MoveIt demo with move_group
roslaunch panda_moveit_config demo.launch

# Verify service exists
rosservice list | grep check_state_validity
```

### Slow Performance
```bash
# Use --no_animation to skip trajectory visualization
python simulate_in_moveit.py ... --no_animation

# Or disable visualization entirely
python simulate_in_moveit.py ... --no_visualization
```

### Visualization Issues
```bash
# In RViz, make sure you have:
# 1. RobotModel display enabled
# 2. PlanningScene display enabled
# 3. Fixed Frame = "panda_link0" or "world"

# Reset view: Press 'r' in RViz
```

## Running Without ROS

If ROS is not available, the script runs in **simulation-only mode**:
```bash
python simulate_in_moveit.py \
  --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 100
```

This mode:
- ✓ Runs learned planner
- ✓ Reports planner success rates
- ✗ No RViz visualization
- ✗ No MoveIt validation

## Paper Reproduction

To reproduce paper results exactly:
```bash
# 1000 scenarios, 1 obstacle (Table 1 in paper)
python simulate_in_moveit.py \
  --checkpoint ../model_params/panda_10k/model.ckpt-003000.pt \
  --checkpoint_obs ../model_params/panda_10k/snapshots_obs/model.ckpt-000140-000170.pt \
  --config ../model_params/panda_10k/20260105_210436817-runcmd.json \
  --num_scenarios 1000 \
  --num_obstacles 1 \
  --no_visualization  # Faster for large runs

# Expected results:
# Planner success: ~74% (our implementation)
# MoveIt validation: ~100% (confirms collision-free)
```

## Integration with RISE

To use this for RISE project:
1. Replace VAE with RISE's learned model
2. Replace obstacle generation with RISE's environment
3. Keep MoveIt validation pipeline (ground truth collision checking)
4. Compare RISE success rate vs this paper's 74%

Key insight: **Always validate with MoveIt** - it's the ground truth collision checker used during training.
