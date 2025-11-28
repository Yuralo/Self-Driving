import numpy as np
import cv2
import os

def inspect_comma():
    base_dir = 'data/comma_ai'
    
    # Check video
    video_path = os.path.join(base_dir, 'video.hevc')
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {video_path}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {frame_count/fps:.2f}s" if fps > 0 else "  Duration: Unknown")
    else:
        print("No video.hevc found")

    # Check steering
    steering_dir = os.path.join(base_dir, 'processed_log/CAN/steering_angle')
    t_path = os.path.join(steering_dir, 't')
    val_path = os.path.join(steering_dir, 'value')
    
    if os.path.exists(t_path) and os.path.exists(val_path):
        try:
            # Try loading as numpy
            t = np.load(t_path)
            val = np.load(val_path)
            print(f"Steering Data (npy):")
            print(f"  Count: {len(t)}")
            print(f"  t range: {t[0]} to {t[-1]}")
            print(f"  val range: {val.min()} to {val.max()}")
        except Exception as e:
            print(f"Could not load as npy: {e}")
            # Try raw binary (float64?)
            try:
                t = np.fromfile(t_path, dtype=np.float64)
                val = np.fromfile(val_path, dtype=np.float64)
                print(f"Steering Data (raw float64):")
                print(f"  Count: {len(t)}")
                print(f"  t range: {t[0]} to {t[-1]}")
            except:
                print("Could not load as raw float64")

if __name__ == '__main__':
    inspect_comma()
