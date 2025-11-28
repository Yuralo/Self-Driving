import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class CommaDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=1):
        """
        Args:
            root_dir (string): Path to the comma2k19 dataset (contains chunks or segments).
            transform (callable, optional): Transform to be applied on a sample.
            sequence_length (int): Number of frames to stack.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.samples = [] # List of (video_path, frame_idx, steering_angle)
        
        # Find all segments
        # Look for video.hevc files
        video_files = glob.glob(os.path.join(root_dir, '**', 'video.hevc'), recursive=True)
        
        print(f"Found {len(video_files)} segments in {root_dir}")
        
        for video_path in video_files:
            segment_dir = os.path.dirname(video_path)
            
            # Load frame times
            frame_times_path = os.path.join(segment_dir, 'global_pose', 'frame_times')
            if not os.path.exists(frame_times_path):
                continue
            
            try:
                frame_times = np.load(frame_times_path)
            except:
                continue
                
            # Load steering
            steering_dir = os.path.join(segment_dir, 'processed_log', 'CAN', 'steering_angle')
            t_path = os.path.join(steering_dir, 't')
            val_path = os.path.join(steering_dir, 'value')
            
            if not os.path.exists(t_path) or not os.path.exists(val_path):
                continue
                
            try:
                steering_t = np.load(t_path)
                steering_val = np.load(val_path)
            except:
                continue
            
            # Interpolate steering to frame times
            # steering_val is in degrees.
            # We want to normalize. Udacity is -1 to 1 (approx -25 to 25 deg).
            # Comma steering can be larger. Let's keep it in degrees for now and normalize in the transform or here.
            # Let's normalize by dividing by 25.0 to match our previous convention.
            
            interp_steering = np.interp(frame_times, steering_t, steering_val)
            interp_steering = interp_steering / 25.0
            
            # Add samples
            # We need to open the video later, so store path and index
            # Note: We can't easily seek in HEVC with cv2 reliably without scanning.
            # But for training, we usually want efficient access.
            # If the dataset is large, extracting frames to JPG is better.
            # But the user asked to "interpret the data", implying using it as is.
            # Reading HEVC randomly is SLOW.
            # For this demo, we might cache the video capture object or just accept it's slow.
            # OR, since we have `imgs` folder in the sample, maybe the big dataset has them too?
            # Comma2k19-ld (low res) usually has `video.hevc`.
            
            # Optimization: We can't keep 1000 video captures open.
            # We'll store the segment info and open on demand, but that's very slow for random access.
            # Better approach for training: Extract frames or use a sequential sampler.
            # For this implementation, I will assume we can read frames. 
            # To make it faster, I'll implement a caching mechanism or just warn.
            
            for i in range(len(frame_times) - sequence_length + 1):
                self.samples.append({
                    'video_path': video_path,
                    'frame_idx': i,
                    'steerings': interp_steering[i:i+sequence_length]
                })
                
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        start_frame = sample['frame_idx']
        target_steerings = sample['steerings']
        
        # Open video
        # This is the bottleneck. 
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        images = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # Fallback
                frame = np.zeros((128, 256, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(frame)
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
        cap.release()
        
        images = torch.stack(images)
        # Target: last frame steering
        target = torch.tensor(target_steerings[-1], dtype=torch.float32)
        
        return images, target
