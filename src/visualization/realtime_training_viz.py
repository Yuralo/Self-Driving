import json
import os
import time
from collections import deque

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class RealtimeTrainingVisualizer:
    def __init__(self, save_dir='visualization/realtime', max_history=100):
        """
        Real-time training visualization that shows predictions vs targets on images.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_history = max_history
        
        # History for loss plotting
        self.loss_history = deque(maxlen=max_history)
        self.step_history = deque(maxlen=max_history)
        
        # Setup figure
        self.fig = None
        self.axes = None
        self.setup_figure()
        
    def setup_figure(self):
        """Setup matplotlib figure with subplots."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('Real-time Training Progress', fontsize=16, fontweight='bold')
        
        # Top row: Sample predictions
        self.axes[0, 0].set_title('Sample 1: Prediction vs Target', fontsize=12)
        self.axes[0, 1].set_title('Sample 2: Prediction vs Target', fontsize=12)
        
        # Bottom row: Loss curve and statistics
        self.axes[1, 0].set_title('Training Loss', fontsize=12)
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Prediction Statistics', fontsize=12)
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
        
    def draw_steering_line(self, ax, image, steering_angle, color, label, linewidth=3):
        """
        Draw steering angle as a line on the image.
        steering_angle: normalized (-1 to 1)
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h - 20  # Bottom center of image
        length = min(h, w) * 0.4
        
        # Convert normalized steering to angle in degrees
        # Assuming -1 to 1 maps to approximately -25 to 25 degrees
        angle_deg = steering_angle * 25.0
        
        # Convert to radians (0 is straight up, negative is left, positive is right)
        angle_rad = np.radians(90 - angle_deg)
        
        end_x = center_x + length * np.cos(angle_rad)
        end_y = center_y - length * np.sin(angle_rad)
        
        ax.plot([center_x, end_x], [center_y, end_y], 
                color=color, linewidth=linewidth, label=label, alpha=0.8)
        
        # Draw circle at center
        circle = patches.Circle((center_x, center_y), 3, color=color, zorder=10)
        ax.add_patch(circle)
        
    def visualize_batch(self, images, targets, predictions, step, loss, epoch):
        """
        Visualize a batch of predictions.
        images: (B, Seq, C, H, W) tensor
        targets: (B,) tensor
        predictions: (B,) tensor
        """
        # Update loss history
        self.loss_history.append(loss)
        self.step_history.append(step)
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Get first two samples
        num_samples = min(2, images.shape[0])
        
        for i in range(num_samples):
            ax = self.axes[0, i]
            
            # Get last frame in sequence
            img_tensor = images[i, -1, :, :, :].cpu()
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img_tensor * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            # Display image
            ax.imshow(img)
            
            # Get predictions and targets
            target = targets[i].item()
            pred = predictions[i].item()
            error = abs(pred - target)
            
            # Draw steering lines
            self.draw_steering_line(ax, img, target, 'green', f'Target: {target:.3f}', linewidth=4)
            self.draw_steering_line(ax, img, pred, 'red', f'Pred: {pred:.3f}', linewidth=3)
            
            # Add text info
            info_text = f'Epoch: {epoch} | Step: {step}\n'
            info_text += f'Target: {target:.3f}\n'
            info_text += f'Pred: {pred:.3f}\n'
            info_text += f'Error: {error:.3f}'
            
            ax.text(10, 10, info_text, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, verticalalignment='top',
                   fontweight='bold')
            
            ax.legend(loc='upper right', fontsize=9)
            ax.axis('off')
            ax.set_title(f'Sample {i+1}: Prediction vs Target', fontsize=12, fontweight='bold')
        
        # Plot loss curve
        if len(self.loss_history) > 1:
            self.axes[1, 0].plot(self.step_history, self.loss_history, 'b-', linewidth=2, label='Loss')
            self.axes[1, 0].set_xlabel('Step', fontsize=10)
            self.axes[1, 0].set_ylabel('Loss', fontsize=10)
            self.axes[1, 0].set_title('Training Loss (Last 100 Steps)', fontsize=12, fontweight='bold')
            self.axes[1, 0].grid(True, alpha=0.3)
            self.axes[1, 0].legend()
            
            # Add moving average
            if len(self.loss_history) > 10:
                window = min(10, len(self.loss_history))
                ma = np.convolve(list(self.loss_history), np.ones(window)/window, mode='valid')
                ma_steps = list(self.step_history)[window-1:]
                self.axes[1, 0].plot(ma_steps, ma, 'r--', linewidth=2, alpha=0.7, label='MA(10)')
        
        # Statistics
        stats_ax = self.axes[1, 1]
        stats_ax.axis('off')
        
        if len(predictions) > 0:
            preds_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
            targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else np.array(targets)
            
            errors = np.abs(preds_np - targets_np)
            
            stats_text = f'Batch Statistics:\n'
            stats_text += f'Current Loss: {loss:.6f}\n'
            stats_text += f'Mean Error: {errors.mean():.4f}\n'
            stats_text += f'Max Error: {errors.max():.4f}\n'
            stats_text += f'Min Error: {errors.min():.4f}\n'
            stats_text += f'Std Error: {errors.std():.4f}\n\n'
            
            if len(self.loss_history) > 1:
                stats_text += f'Loss Statistics:\n'
                stats_text += f'Current: {self.loss_history[-1]:.6f}\n'
                stats_text += f'Mean (last {len(self.loss_history)}): {np.mean(list(self.loss_history)):.6f}\n'
                stats_text += f'Min: {min(self.loss_history):.6f}\n'
                stats_text += f'Max: {max(self.loss_history):.6f}\n'
            
            stats_ax.text(0.1, 0.5, stats_text, 
                         fontsize=11, verticalalignment='center',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.save_dir, f'training_step_{step:06d}.png')
        self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
        
        # Also save latest
        latest_path = os.path.join(self.save_dir, 'latest_training.png')
        self.fig.savefig(latest_path, dpi=100, bbox_inches='tight')
        
        return save_path

