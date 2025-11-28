import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from src.data.udacity_dataset import DrivingDataset, get_transforms
from src.models.policy import DrivingPolicy
import os
import json
import time

def train_multi_dataset():
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
    
    # Model
    model = DrivingPolicy(sequence_length=SEQUENCE_LENGTH).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Logging
    log_file = 'training_log_multi.json'
    logs = []
    
    print("Starting multi-dataset training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
                # Live logging
                log_entry = {
                    'epoch': epoch + 1,
                    'step': batch_idx + epoch * len(train_loader),
                    'loss': loss.item(),
                    'timestamp': time.time()
                }
                logs.append(log_entry)
                with open(log_file, 'w') as f:
                    json.dump(logs, f)
        
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
        
        # Save checkpoint
        torch.save(model.state_dict(), f"models/model_multi_epoch_{epoch+1}.pth")
    
    print(f"\nTraining complete! Used {len(datasets)} dataset(s)")

if __name__ == '__main__':
    train_multi_dataset()
