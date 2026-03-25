# Reaching Through Latent Space - Setup Complete

## Setup Summary

All configuration files have been updated with the correct paths for your system:
- Data directory: `/home/chandrakiran/Projects/Reaching Through Latent Space/data`
- Model directory: `/home/chandrakiran/Projects/Reaching Through Latent Space/model_params/panda_10k`

## Requirements

All Python dependencies have been installed:
- PyTorch
- NumPy, Pandas
- PyYAML
- scikit-learn
- TensorBoard
- Other supporting libraries

## Bug Fix Applied

Fixed a CUDA device mismatch bug in `src/geco.py` where the min/max tensors weren't being moved to the GPU along with the lambda tensor.

## Training the Models

### Option 1: Automated Training (Recommended)

Run both training steps automatically:

```bash
conda activate base
cd ~/Projects/"Reaching Through Latent Space"
./train_models.sh
```

This script will:
1. Train the VAE model
2. Automatically find the latest checkpoint
3. Update the obstacle classifier config with the checkpoint paths
4. Train the VAE obstacle classifier

### Option 2: Manual Training

#### Step 1: Train VAE Model

```bash
conda activate base
cd ~/Projects/"Reaching Through Latent Space"/src
python train_vae.py --c ../config/vae_config/panda_10k.yaml
```

**Note**: The config is set to train for only 4 epochs as a quick test. For better results, edit `config/vae_config/panda_10k.yaml` and increase `epochs_vae` (original paper used many more epochs).

#### Step 2: Update Obstacle Classifier Config

After VAE training completes, find the latest checkpoint:

```bash
ls -t model_params/panda_10k/snapshots/model.ckpt-*.pt | head -1
ls -t model_params/panda_10k/*-runcmd.json | head -1
```

Then update these paths in `config/vae_obs_config/panda_10k.yaml`:
- `vae_run_cmd_path`
- `pretrained_checkpoint_path`

#### Step 3: Train VAE Obstacle Classifier

```bash
cd ~/Projects/"Reaching Through Latent Space"/src
python train_vae_obs.py --c ../config/vae_obs_config/panda_10k.yaml
```

## Directory Structure

```
Reaching Through Latent Space/
├── config/
│   ├── vae_config/
│   │   └── panda_10k.yaml          # VAE training config (UPDATED)
│   └── vae_obs_config/
│       └── panda_10k.yaml          # Obstacle classifier config (UPDATED)
├── data/
│   ├── free_space_10k_train.dat    # VAE training data
│   ├── free_space_10k_test.dat     # VAE validation data
│   ├── collision_10k_train.dat     # Obstacle classifier training data
│   └── collision_10k_test.dat      # Obstacle classifier validation data
├── model_params/
│   └── panda_10k/                  # Model checkpoints saved here
│       └── snapshots/
├── src/
│   ├── train_vae.py                # VAE training script
│   ├── train_vae_obs.py            # Obstacle classifier training script
│   ├── vae.py                      # VAE model definition
│   ├── vae_obs.py                  # Obstacle classifier model
│   ├── geco.py                     # GECO optimizer (FIXED)
│   └── ...
└── train_models.sh                 # Automated training script (NEW)
```

## Monitoring Training

TensorBoard logs are saved during training. To monitor:

```bash
tensorboard --logdir=model_params/panda_10k
```

Then open your browser to `http://localhost:6006`

## Notes

- The current configuration uses small networks (2 hidden layers, 64 units) and only 4 epochs for quick testing
- For full training as described in the paper, you should:
  - Increase `num_hidden_layers` to 4
  - Increase `units_per_layer` to 2048
  - Increase `epochs_vae` to 16000 (or until convergence)
  - Increase `batch_size` to 256
  - Use the 100k training datasets instead of 10k

## Configuration Parameters

Key parameters in `config/vae_config/panda_10k.yaml`:
- `latent_dim: 7` - Dimensionality of latent space (matches robot DoF)
- `g_goal: 0.01` - GECO reconstruction target (in units of squared error)
- `lr_vae: 0.0001` - Learning rate for VAE
- `epochs_vae: 4` - Number of training epochs (increased from config)

## Troubleshooting

If you encounter CUDA out of memory errors, try:
- Reducing `batch_size`
- Reducing `units_per_layer`
- Using CPU by setting `no_cuda: true` in the config
