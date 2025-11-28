import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_comparison():
    """Plot comparison between single and multi-dataset training"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Self-Driving RL Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss Comparison
    ax1 = axes[0, 0]
    
    # Load single dataset logs
    if os.path.exists('training_log.json'):
        with open('training_log.json', 'r') as f:
            logs_single = json.load(f)
        steps_single = [entry['step'] for entry in logs_single]
        losses_single = [entry['loss'] for entry in logs_single]
        ax1.plot(steps_single, losses_single, label='Single Dataset (Udacity)', alpha=0.7, linewidth=2)
    
    # Load multi dataset logs
    if os.path.exists('training_log_multi.json'):
        with open('training_log_multi.json', 'r') as f:
            logs_multi = json.load(f)
        steps_multi = [entry['step'] for entry in logs_multi]
        losses_multi = [entry['loss'] for entry in logs_multi]
        ax1.plot(steps_multi, losses_multi, label='Multi Dataset', alpha=0.7, linewidth=2)
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Distribution
    ax2 = axes[0, 1]
    if os.path.exists('training_log.json') and os.path.exists('training_log_multi.json'):
        ax2.hist(losses_single, bins=30, alpha=0.6, label='Single Dataset', edgecolor='black')
        ax2.hist(losses_multi, bins=30, alpha=0.6, label='Multi Dataset', edgecolor='black')
        ax2.set_xlabel('Loss Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Loss Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Moving Average
    ax3 = axes[1, 0]
    window = 20
    
    if os.path.exists('training_log.json'):
        ma_single = np.convolve(losses_single, np.ones(window)/window, mode='valid')
        ax3.plot(range(len(ma_single)), ma_single, label=f'Single Dataset (MA-{window})', linewidth=2)
    
    if os.path.exists('training_log_multi.json'):
        ma_multi = np.convolve(losses_multi, np.ones(window)/window, mode='valid')
        ax3.plot(range(len(ma_multi)), ma_multi, label=f'Multi Dataset (MA-{window})', linewidth=2)
    
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Smoothed Loss', fontsize=12)
    ax3.set_title('Smoothed Training Loss', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "Training Statistics\n" + "="*40 + "\n\n"
    
    if os.path.exists('training_log.json'):
        final_loss_single = np.mean(losses_single[-50:]) if len(losses_single) > 50 else np.mean(losses_single)
        stats_text += f"Single Dataset:\n"
        stats_text += f"  • Total steps: {len(losses_single)}\n"
        stats_text += f"  • Final avg loss: {final_loss_single:.4f}\n"
        stats_text += f"  • Min loss: {min(losses_single):.4f}\n"
        stats_text += f"  • Max loss: {max(losses_single):.4f}\n\n"
    
    if os.path.exists('training_log_multi.json'):
        final_loss_multi = np.mean(losses_multi[-50:]) if len(losses_multi) > 50 else np.mean(losses_multi)
        stats_text += f"Multi Dataset:\n"
        stats_text += f"  • Total steps: {len(losses_multi)}\n"
        stats_text += f"  • Final avg loss: {final_loss_multi:.4f}\n"
        stats_text += f"  • Min loss: {min(losses_multi):.4f}\n"
        stats_text += f"  • Max loss: {max(losses_multi):.4f}\n\n"
    
    stats_text += "\nArchitecture:\n"
    stats_text += "  • CNN Spatial Encoder\n"
    stats_text += "  • Transformer Temporal Encoder\n"
    stats_text += "  • Sequence Length: 5 frames\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('visualization/training_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to visualization/training_comparison.png")

if __name__ == '__main__':
    plot_comparison()
