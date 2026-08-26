
import cv2
import numpy as np
from typing import List

class VideoOperations:
    @staticmethod
    def encode_frames_to_video(frames: List[np.ndarray], output_path: str, fps: float = 25.0) -> str:
        """Encode frame list to MP4 video"""
        if not frames:
            raise ValueError("No frames to encode")
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for i, frame in enumerate(frames):
            out.write(frame)
            if (i + 1) % 30 == 0:
                print(f"  ✓ Encoded {i + 1}/{len(frames)} frames...")
        
        out.release()
        print(f"✅ Video saved: {output_path}")
        return output_path
    
    @staticmethod
    def read_video_frames(video_path: str) -> tuple:
        """Read all frames from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        print(f"✅ Loaded {len(frames)} frames from {video_path}")
        return frames, fps
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": frame_count,
            "width": w,
            "height": h,
            "duration_seconds": frame_count / fps
        }
