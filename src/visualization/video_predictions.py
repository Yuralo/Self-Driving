import os
import sys

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.comma_dataset import CommaDataset
from src.data.udacity_dataset import DrivingDataset, get_transforms
from src.models.policy import DrivingPolicy


def draw_steering_overlay(image, steering_angle, color=(0, 255, 0), thickness=3, label=""):
    """
    Draw steering angle as a line on the image.
    image: numpy array (H, W, 3) in RGB
    steering_angle: normalized (-1 to 1)
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h - 20  # Bottom center
    length = min(h, w) * 0.4
    
    # Convert normalized steering to angle in degrees
    angle_deg = steering_angle * 25.0
    angle_rad = np.radians(90 - angle_deg)
    
    end_x = int(center_x + length * np.cos(angle_rad))
    end_y = int(center_y - length * np.sin(angle_rad))
    
    # Draw line
    cv2.line(image, (center_x, center_y), (end_x, end_y), color, thickness)
    
    # Draw circle at center
    cv2.circle(image, (center_x, center_y), 5, color, -1)
    
    # Add text
    if label:
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2, cv2.LINE_AA)
    
    return image

def tensor_to_image(img_tensor):
    """Convert normalized tensor to numpy image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def create_prediction_video(model, dataset, output_path='visualization/predictions_video.mp4', 
                           num_samples=100, fps=10, device='cpu'):
    """
    Create a video showing model predictions vs targets.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create temporary directory for frames
    temp_dir = 'visualization/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    frame_count = 0
    errors = []
    
    print(f"Generating {num_samples} frames...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if batch_idx >= num_samples:
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            # Get prediction
            predictions = model(images).squeeze()
            
            # Get last frame in sequence
            img_tensor = images[0, -1, :, :, :]
            img = tensor_to_image(img_tensor)
            
            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Get values
            target = targets[0].item()
            pred = predictions[0].item()
            error = abs(pred - target)
            errors.append(error)
            
            # Draw steering lines
            # Target in green
            img_bgr = draw_steering_overlay(img_bgr, target, color=(0, 255, 0), 
                                           thickness=4, label=f"Target: {target:.3f}")
            # Prediction in red
            img_bgr = draw_steering_overlay(img_bgr, pred, color=(0, 0, 255), 
                                           thickness=3, label=f"Pred: {pred:.3f}")
            
            # Add text overlay
            text_y = 60
            cv2.putText(img_bgr, f"Frame: {batch_idx}", (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, f"Target: {target:.3f}", (10, text_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, f"Pred: {pred:.3f}", (10, text_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, f"Error: {error:.3f}", (10, text_y + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.png')
            cv2.imwrite(frame_path, img_bgr)
            frame_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{num_samples} frames")
    
    print(f"Mean error: {np.mean(errors):.4f}, Max error: {np.max(errors):.4f}")
    
    # Create video from frames
    print("Creating video...")
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    
    if not frame_files:
        print("No frames to create video from!")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
    h, w, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(temp_dir, frame_file))
        out.write(frame)
    
    out.release()
    
    # Cleanup temp frames
    for frame_file in frame_files:
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)
    
    print(f"Video saved to {output_path}")
    return output_path

def create_sequence_video(model, dataset, output_path='visualization/sequence_predictions.mp4',
                         num_sequences=20, fps=5, device='cpu'):
    """
    Create a video showing predictions on sequential frames from the dataset.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    temp_dir = 'visualization/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    frame_count = 0
    
    print(f"Generating sequence video with {num_sequences} sequences...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if batch_idx >= num_sequences:
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images).squeeze()
            
            # Create a frame showing all frames in the sequence
            seq_length = images.shape[1]
            frames = []
            
            for i in range(seq_length):
                img_tensor = images[0, i, :, :, :]
                img = tensor_to_image(img_tensor)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Only draw steering on last frame
                if i == seq_length - 1:
                    target = targets[0].item()
                    pred = predictions[0].item()
                    img_bgr = draw_steering_overlay(img_bgr, target, color=(0, 255, 0), thickness=3)
                    img_bgr = draw_steering_overlay(img_bgr, pred, color=(0, 0, 255), thickness=2)
                    
                    # Add text
                    cv2.putText(img_bgr, f"T:{target:.2f} P:{pred:.2f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                frames.append(img_bgr)
            
            # Concatenate frames horizontally
            combined = np.hstack(frames)
            
            # Add sequence info
            cv2.putText(combined, f"Sequence {batch_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.png')
            cv2.imwrite(frame_path, combined)
            frame_count += 1
    
    # Create video
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    if frame_files:
        first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
        h, w, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(temp_dir, frame_file))
            out.write(frame)
        
        out.release()
        
        # Cleanup
        for frame_file in frame_files:
            os.remove(os.path.join(temp_dir, frame_file))
        os.rmdir(temp_dir)
        
        print(f"Sequence video saved to {output_path}")

if __name__ == '__main__':
    # Load model
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DrivingPolicy(sequence_length=5).to(device)
    
    # Load latest checkpoint
    checkpoints = [f for f in os.listdir('models') if f.endswith('.pth')]
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        print(f"Loading checkpoint: {latest}")
        model.load_state_dict(torch.load(f'models/{latest}', map_location=device))
    else:
        print("No checkpoint found, using random weights")
    
    # Load dataset
    transform = get_transforms()
    dataset = DrivingDataset('data/data', transform=transform, sequence_length=5)
    
    # Create videos
    print("\n1. Creating prediction video...")
    create_prediction_video(model, dataset, 
                           output_path='visualization/predictions_video.mp4',
                           num_samples=100, fps=10, device=device)
    
    print("\n2. Creating sequence video...")
    create_sequence_video(model, dataset,
                         output_path='visualization/sequence_predictions.mp4',
                         num_sequences=20, fps=5, device=device)
    
    print("\nDone!")

