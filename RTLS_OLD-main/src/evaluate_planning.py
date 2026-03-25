"""
Evaluation script for path planning metrics:
- Success Rate (%)
- Planning Time (ms)
- Path Length

Usage:
    python evaluate_planning.py --checkpoint ../model_params/panda_10k/snapshots/model.ckpt-001000.pt \
                                 --num_problems 100 \
                                 --max_steps 500 \
                                 --success_threshold 0.05
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch import nn

from robot_state_dataset import RobotStateDataset
from vae import VAE
from vae_obs import VAEObstacleBCE

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

FMAX = 1e5


def evaluate_path_planning(model, model_obs, val_dataset, mean_train, std_train, 
                           device, args):
    """
    Evaluate path planning performance
    
    Returns:
        dict with keys: success_rate, avg_planning_time_ms, avg_path_length
    """
    
    model.eval()
    if model_obs is not None:
        model_obs.eval()
    
    mean_train_tensor = torch.Tensor(mean_train).to(device)
    std_train_tensor = torch.Tensor(std_train).to(device)
    
    successes = 0
    planning_times = []
    path_lengths = []
    min_distances = []
    
    num_problems = min(args.num_problems, len(val_dataset))
    
    logging.info(f"Evaluating {num_problems} planning problems...")
    logging.info(f"Success threshold: {args.success_threshold}m")
    logging.info(f"Max planning steps: {args.max_steps}")
    
    for i in range(num_problems):
        x, goal_xyz = val_dataset[i]
        x = x.to(device)
        goal_xyz = goal_xyz.to(device)
        
        # Denormalize goal
        goal_xyz = goal_xyz * std_train_tensor[:, model.input_dim:model.input_dim + 3]
        goal_xyz = goal_xyz + mean_train_tensor[:, model.input_dim:model.input_dim + 3]
        
        # Get initial latent code
        z = model.get_features(x).detach().requires_grad_(True)
        
        # Initialize optimizer
        optimizer = optim.Adam([z], lr=args.planning_lr)
        
        # Track path through latent space
        latent_path = [z.detach().cpu().numpy().copy()]
        
        # Start planning timer
        start_time = time.time()
        
        min_dist = FMAX
        
        # Gradient descent in latent space
        for step in range(args.max_steps):
            optimizer.zero_grad()
            
            # Decode latent to configuration
            xPrime = model.decoder(z)
            
            # Denormalize output
            xPrime = xPrime * std_train_tensor[:, :model.input_dim]
            xPrime = xPrime + mean_train_tensor[:, :model.input_dim]
            
            # Extract end-effector position
            ee_xyz = xPrime[:, -3:]
            
            # Compute distance to goal
            dist_to_goal = torch.dist(ee_xyz, goal_xyz)
            min_dist = min(min_dist, dist_to_goal.item())
            
            # Check for success
            if dist_to_goal.item() < args.success_threshold:
                end_time = time.time()
                planning_time = (end_time - start_time) * 1000  # ms
                
                successes += 1
                planning_times.append(planning_time)
                
                # Compute path length in latent space
                path_length = compute_path_length(latent_path)
                path_lengths.append(path_length)
                
                min_distances.append(min_dist)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Problem {i+1}/{num_problems}: SUCCESS in {step+1} steps, "
                               f"{planning_time:.2f}ms, path_length={path_length:.4f}")
                break
            
            # Optimize
            dist_to_goal.backward()
            optimizer.step()
            
            # Track path
            if step % 10 == 0:  # Sample every 10 steps to save memory
                latent_path.append(z.detach().cpu().numpy().copy())
        else:
            # Failed to reach goal
            end_time = time.time()
            planning_time = (end_time - start_time) * 1000
            min_distances.append(min_dist)
            
            if (i + 1) % 10 == 0:
                logging.info(f"Problem {i+1}/{num_problems}: FAILED (min_dist={min_dist:.4f}m)")
    
    # Compute statistics
    success_rate = (successes / num_problems) * 100
    avg_planning_time = np.mean(planning_times) if planning_times else 0
    std_planning_time = np.std(planning_times) if planning_times else 0
    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    std_path_length = np.std(path_lengths) if path_lengths else 0
    avg_min_distance = np.mean(min_distances)
    
    results = {
        'num_problems': num_problems,
        'successes': successes,
        'success_rate_percent': success_rate,
        'avg_planning_time_ms': avg_planning_time,
        'std_planning_time_ms': std_planning_time,
        'avg_path_length': avg_path_length,
        'std_path_length': std_path_length,
        'avg_min_distance': avg_min_distance,
        'planning_times_ms': planning_times,
        'path_lengths': path_lengths,
        'min_distances': min_distances
    }
    
    return results


def compute_path_length(latent_path):
    """
    Compute path length in latent space
    
    Args:
        latent_path: List of latent codes (numpy arrays)
    
    Returns:
        Total path length
    """
    if len(latent_path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(latent_path)):
        diff = latent_path[i] - latent_path[i-1]
        total_length += np.linalg.norm(diff)
    
    return total_length


def print_results(results):
    """Print results in paper-ready format"""
    
    logging.info("\n" + "="*60)
    logging.info("PATH PLANNING EVALUATION RESULTS")
    logging.info("="*60)
    logging.info(f"Test Problems:      {results['num_problems']}")
    logging.info(f"Successful:         {results['successes']}")
    logging.info("-"*60)
    logging.info(f"Success Rate:       {results['success_rate_percent']:.2f}%")
    logging.info(f"Planning Time:      {results['avg_planning_time_ms']:.2f} ± {results['std_planning_time_ms']:.2f} ms")
    logging.info(f"Path Length:        {results['avg_path_length']:.4f} ± {results['std_path_length']:.4f}")
    logging.info(f"Avg Min Distance:   {results['avg_min_distance']:.4f} m")
    logging.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate path planning performance')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to VAE checkpoint')
    parser.add_argument('--checkpoint_obs', type=str, default=None,
                       help='Path to obstacle classifier checkpoint (optional)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config JSON file')
    
    # Evaluation arguments
    parser.add_argument('--num_problems', type=int, default=100,
                       help='Number of planning problems to test')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum planning steps per problem')
    parser.add_argument('--planning_lr', type=float, default=0.03,
                       help='Learning rate for latent space optimization')
    parser.add_argument('--success_threshold', type=float, default=0.05,
                       help='Distance threshold for success (meters)')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='/home/chandrakiran/Projects/Reaching Through Latent Space/data',
                       help='Path to dataset')
    parser.add_argument('--test_data_name', type=str, default='free_space_10k_test.dat',
                       help='Test dataset filename')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    # Device
    parser.add_argument('--no_cuda', action='store_true', default=False,
                       help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
        if 'parsed_args' in config:
            config = config['parsed_args']
    
    # Load dataset
    val_dataset = RobotStateDataset(
        args.data_path, 
        train=2,  # TEST set
        test_data_name=args.test_data_name
    )
    
    mean_train = val_dataset.get_mean_train()
    std_train = val_dataset.get_std_train()
    
    # Load VAE model
    logging.info(f"Loading VAE from {args.checkpoint}")
    model = VAE(
        config['input_dim'],
        config['latent_dim'],
        config['units_per_layer'],
        config['num_hidden_layers']
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logging.info(f"Model: {config['units_per_layer']} units, {config['latent_dim']}D latent space")
    
    # Load obstacle classifier (optional)
    model_obs = None
    if args.checkpoint_obs is not None:
        logging.info(f"Loading obstacle classifier from {args.checkpoint_obs}")
        model_obs = VAEObstacleBCE(
            config['input_dim'],
            config['latent_dim'],
            config['units_per_layer'],
            config['num_hidden_layers']
        )
        checkpoint_obs = torch.load(args.checkpoint_obs, map_location=device)
        model_obs.load_state_dict(checkpoint_obs['model_state_dict'])
        model_obs.to(device)
        model_obs.eval()
    
    # Run evaluation
    results = evaluate_path_planning(
        model, model_obs, val_dataset, mean_train, std_train, device, args
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
