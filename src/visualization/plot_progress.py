import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def plot_progress(log_file=None):
    """
    Plot training progress from log file(s).
    If log_file is None, will try to find training_log_multi.json or training_log.json
    """
    if log_file is None:
        # Try to find log files
        if os.path.exists('training_log_multi.json'):
            log_file = 'training_log_multi.json'
        elif os.path.exists('training_log.json'):
            log_file = 'training_log.json'
        else:
            print("No log file found. Looking for training_log_multi.json or training_log.json")
            return
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        print("Log file is empty or corrupted.")
        return
    
    if not logs:
        print("Log file is empty.")
        return

    steps = [entry['step'] for entry in logs]
    losses = [entry['loss'] for entry in logs]
    epochs = [entry.get('epoch', 1) for entry in logs]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss over steps
    axes[0].plot(steps, losses, 'b-', alpha=0.6, linewidth=1, label='Training Loss')
    
    # Add moving average
    if len(losses) > 10:
        window = min(50, len(losses) // 10)
        ma = np.convolve(losses, np.ones(window)/window, mode='valid')
        ma_steps = steps[window-1:]
        axes[0].plot(ma_steps, ma, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss by epoch
    unique_epochs = sorted(set(epochs))
    epoch_losses = []
    for epoch in unique_epochs:
        epoch_loss = [loss for e, loss in zip(epochs, losses) if e == epoch]
        epoch_losses.append(np.mean(epoch_loss))
    
    axes[1].plot(unique_epochs, epoch_losses, 'go-', linewidth=2, markersize=8, label='Average Loss per Epoch')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Average MSE Loss', fontsize=12)
    axes[1].set_title('Average Loss per Epoch', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(unique_epochs)
    
    plt.tight_layout()
    
    os.makedirs('visualization', exist_ok=True)
    output_path = 'visualization/training_loss.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Print statistics
    print(f"\nTraining Statistics:")
    print(f"  Total steps: {len(steps)}")
    print(f"  Total epochs: {max(epochs)}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Min loss: {min(losses):.6f}")
    print(f"  Mean loss: {np.mean(losses):.6f}")

if __name__ == '__main__':
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    plot_progress(log_file)
