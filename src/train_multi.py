import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split

from src.data.udacity_dataset import DrivingDataset, get_transforms
from src.models.policy import DrivingPolicy, ResidualStreamPolicy
from src.visualization.realtime_training_viz import RealtimeTrainingVisualizer


def train_multi_dataset(use_residual_stream=True, use_mixed_precision=True, 
                       gradient_accumulation_steps=1, warmup_steps=500):
    # Hyperparameters
    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 2
    SEQUENCE_LENGTH = 5
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data - Load multiple datasets
    datasets = []
    
    # Dataset 1: Udacity
    udacity_dataset = DrivingDataset('data/data', transform=get_transforms(), sequence_length=SEQUENCE_LENGTH)
    datasets.append(udacity_dataset)
    print(f"Loaded Udacity dataset: {len(udacity_dataset)} samples")
    
    # Dataset 2: SullyChen (if available)
    # For now, we'll create a synthetic second dataset by augmenting the Udacity data
    # In practice, you would load the actual SullyChen dataset here
    try:
        from src.data.sullychen_dataset import SullyChenDataset
        sullychen_dataset = SullyChenDataset('data/sullychen', transform=get_transforms(), sequence_length=SEQUENCE_LENGTH)
        datasets.append(sullychen_dataset)
        print(f"Loaded SullyChen dataset: {len(sullychen_dataset)} samples")
    except Exception as e:
        print(f"Could not load SullyChen dataset: {e}")

    # Dataset 3: Comma2k19
    try:
        from src.data.comma_dataset import CommaDataset
        comma_dataset = CommaDataset('data/comma2k19', transform=get_transforms(), sequence_length=SEQUENCE_LENGTH)
        if len(comma_dataset) > 0:
            datasets.append(comma_dataset)
            print(f"Loaded Comma2k19 dataset: {len(comma_dataset)} samples")
        else:
            print("Comma2k19 dataset found but empty (no video.hevc or matching logs)")
    except Exception as e:
        print(f"Could not load Comma2k19 dataset: {e}")
    
    # Combine datasets
    if len(datasets) > 1:
        full_dataset = ConcatDataset(datasets)
        print(f"Combined dataset size: {len(full_dataset)} samples")
    else:
        full_dataset = datasets[0]
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model - choose between architectures
    if use_residual_stream:
        print("Using ResidualStreamPolicy (advanced LLM-inspired architecture)")
        print("  - SwiGLU activation in MLPs")
        print("  - Rotary Positional Embeddings (RoPE)")
        print("  - Layer scaling for stability")
        model = ResidualStreamPolicy(sequence_length=SEQUENCE_LENGTH, d_model=256, nhead=4, 
                                    dropout=0.1, use_rope=True, use_swiglu=True).to(device)
    else:
        print("Using DrivingPolicy (standard architecture)")
        model = DrivingPolicy(sequence_length=SEQUENCE_LENGTH, dropout=0.3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5, betas=(0.9, 0.95))
    
    # Mixed precision training
    scaler = None
    if use_mixed_precision and (device.type == 'cuda' or device.type == 'mps'):
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        print(f"Using mixed precision training: {use_mixed_precision}")
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Real-time visualization
    viz = RealtimeTrainingVisualizer(save_dir='visualization/realtime', max_history=200)
    
    # Logging
    log_file = 'training_log_multi.json'
    logs = []
    
    print("Starting multi-dataset training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            global_step = batch_idx + epoch * len(train_loader)
            
            # Mixed precision forward pass
            if scaler is not None and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, targets)
                    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                
                scaler.scale(loss).backward()
            else:
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                if scaler is not None and device.type == 'cuda':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Learning rate warmup
                if global_step < warmup_steps:
                    scheduler_warmup.step()
                
                optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
                # Live logging
                log_entry = {
                    'epoch': epoch + 1,
                    'step': global_step,
                    'loss': loss.item(),
                    'timestamp': time.time()
                }
                logs.append(log_entry)
                with open(log_file, 'w') as f:
                    json.dump(logs, f)
                
                # Real-time visualization every 50 steps
                if batch_idx % 50 == 0:
                    with torch.no_grad():
                        # Get a sample batch for visualization
                        sample_images = images[:2].cpu()  # First 2 samples
                        sample_targets = targets[:2].cpu()
                        sample_outputs = outputs[:2].cpu()
                        
                        viz.visualize_batch(
                            sample_images, 
                            sample_targets, 
                            sample_outputs,
                            step=global_step,
                            loss=loss.item(),
                            epoch=epoch + 1
                        )
                        print(f"  Saved visualization to visualization/realtime/latest_training.png")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate (plateau scheduler after warmup)
        if epoch * len(train_loader) >= warmup_steps:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler_plateau.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        else:
            print(f"  Learning rate (warmup): {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        checkpoint_path = f"models/model_multi_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")
    
    print(f"\nTraining complete! Used {len(datasets)} dataset(s)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train driving policy model')
    parser.add_argument('--model', type=str, choices=['standard', 'residual'], default='residual',
                       help='Model architecture: standard or residual (default: residual)')
    args = parser.parse_args()
    
    use_residual = args.model == 'residual'
    train_multi_dataset(use_residual_stream=use_residual)
