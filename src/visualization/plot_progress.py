import json
import matplotlib.pyplot as plt
import os
import time

def plot_progress():
    log_file = 'training_log.json'
    if not os.path.exists(log_file):
        print("No log file found.")
        return

    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        print("Log file is empty or corrupted.")
        return

    steps = [entry['step'] for entry in logs]
    losses = [entry['loss'] for entry in logs]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualization/training_loss.png')
    print("Saved plot to visualization/training_loss.png")

if __name__ == '__main__':
    plot_progress()
