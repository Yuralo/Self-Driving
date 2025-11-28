import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.udacity_dataset import DrivingDataset, get_transforms
from src.models.policy import DrivingPolicy
from torch.utils.data import DataLoader

def visualize_predictions():
    # Load Model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = DrivingPolicy(sequence_length=5).to(device)
    
    # Load latest checkpoint
    checkpoints = [f for f in os.listdir('models') if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found.")
        return
    latest_checkpoint = sorted(checkpoints)[-1]
    model.load_state_dict(torch.load(f'models/{latest_checkpoint}', map_location=device))
    model.eval()
    
    # Load Data
    dataset = DrivingDataset('data/data', transform=get_transforms(), sequence_length=5)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch
    images, targets = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images).squeeze()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i in range(4):
        # Get the last image in the sequence
        img_tensor = images[i, -1, :, :, :].cpu()
        
        # Unnormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {targets[i]:.2f}, Pred: {predictions[i]:.2f}")
        axes[i].axis('off')
        
        # Draw steering lines (approximate)
        # Center is (100, 66) roughly
        center_x, center_y = 100, 66
        length = 40
        
        # True
        angle_true = targets[i].item() * 25 # Scale up for visibility (approx 25 deg max)
        rad_true = np.radians(angle_true + 90) # +90 because 0 is right in trig, but up in steering? 
        # Actually steering 0 is straight. - is left, + is right usually.
        # Let's assume 0 is up (90 deg).
        # x = cos(theta), y = sin(theta). 
        # If 0 is up, theta = 90 - angle.
        
        end_x_true = center_x + length * np.sin(np.radians(angle_true))
        end_y_true = center_y - length * np.cos(np.radians(angle_true))
        axes[i].plot([center_x, end_x_true], [center_y, end_y_true], 'g-', linewidth=3, label='True')
        
        # Pred
        angle_pred = predictions[i].item() * 25
        end_x_pred = center_x + length * np.sin(np.radians(angle_pred))
        end_y_pred = center_y - length * np.cos(np.radians(angle_pred))
        axes[i].plot([center_x, end_x_pred], [center_y, end_y_pred], 'r--', linewidth=3, label='Pred')
        
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    plt.savefig('visualization/predictions.png')
    print("Saved visualization to visualization/predictions.png")

if __name__ == '__main__':
    visualize_predictions()
