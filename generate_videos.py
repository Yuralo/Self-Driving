#!/usr/bin/env python3
"""
Script to generate prediction videos from a trained model.
Usage: python3 generate_videos.py [checkpoint_path]
"""

import argparse
import os
import sys

import torch

# Add current directory to path
sys.path.append(os.getcwd())

from src.data.comma_dataset import CommaDataset
from src.data.udacity_dataset import DrivingDataset, get_transforms
from src.models.policy import DrivingPolicy, ResidualStreamPolicy
from src.visualization.video_predictions import (
    create_prediction_video,
    create_sequence_video,
)


def main():
    parser = argparse.ArgumentParser(description='Generate prediction videos from trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: latest)')
    parser.add_argument('--dataset', type=str, choices=['udacity', 'comma', 'both'], default='udacity',
                       help='Which dataset to use for video generation')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples for prediction video')
    parser.add_argument('--num-sequences', type=int, default=20,
                       help='Number of sequences for sequence video')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for videos')
    parser.add_argument('--model', type=str, choices=['standard', 'residual'], default='residual',
                       help='Model architecture: standard or residual (default: residual)')
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    if args.model == 'residual':
        print("Using ResidualStreamPolicy architecture")
        model = ResidualStreamPolicy(sequence_length=5, d_model=256, nhead=4, dropout=0.1).to(device)
    else:
        print("Using DrivingPolicy architecture")
        model = DrivingPolicy(sequence_length=5, dropout=0.3).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoints = [f for f in os.listdir('models') if f.endswith('.pth')]
        if not checkpoints:
            print("No checkpoint found! Please train a model first or specify --checkpoint")
            return
        checkpoint_path = f"models/{sorted(checkpoints)[-1]}"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both old and new checkpoint formats
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Train loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
            print(f"  Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint (legacy format)")
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying to load with strict=False...")
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Loaded checkpoint with strict=False (some weights may not match)")
    
    model.eval()
    
    # Load dataset
    transform = get_transforms()
    sequence_length = 5
    
    datasets_to_process = []
    if args.dataset in ['udacity', 'both']:
        try:
            udacity_ds = DrivingDataset('data/data', transform=transform, sequence_length=sequence_length)
            datasets_to_process.append(('udacity', udacity_ds))
            print(f"Loaded Udacity dataset: {len(udacity_ds)} samples")
        except Exception as e:
            print(f"Could not load Udacity dataset: {e}")
    
    if args.dataset in ['comma', 'both']:
        try:
            comma_ds = CommaDataset('data/comma2k19', transform=transform, sequence_length=sequence_length)
            if len(comma_ds) > 0:
                datasets_to_process.append(('comma', comma_ds))
                print(f"Loaded Comma2k19 dataset: {len(comma_ds)} samples")
        except Exception as e:
            print(f"Could not load Comma2k19 dataset: {e}")
    
    if not datasets_to_process:
        print("No datasets loaded! Cannot generate videos.")
        return
    
    # Generate videos for each dataset
    for dataset_name, dataset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Generating videos for {dataset_name} dataset")
        print(f"{'='*60}")
        
        # Prediction video
        print(f"\n1. Creating prediction video...")
        pred_video_path = f'visualization/{dataset_name}_predictions_video.mp4'
        create_prediction_video(
            model, dataset,
            output_path=pred_video_path,
            num_samples=args.num_samples,
            fps=args.fps,
            device=device
        )
        
        # Sequence video
        print(f"\n2. Creating sequence video...")
        seq_video_path = f'visualization/{dataset_name}_sequence_predictions.mp4'
        create_sequence_video(
            model, dataset,
            output_path=seq_video_path,
            num_sequences=args.num_sequences,
            fps=args.fps,
            device=device
        )
    
    print(f"\n{'='*60}")
    print("All videos generated successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

