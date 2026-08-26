
import cv2
import numpy as np
from typing import Optional

class FaceEnhancer:
    def __init__(self):
        try:
            from gfpgan import GFPGANer
            self.enhancer = GFPGANer(model_path="detection_Resnet50_Final.pth", upscale=1, arch="clean", channel_multiplier=2)
            self.available = True
        except Exception as e:
            print(f"GFPGAN not available: {e}")
            self.available = False
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance face quality in frame"""
        if not self.available:
            return frame
        
        try:
            _, restored_faces, _ = self.enhancer.enhance(frame, has_aligned=False, only_center_face=False, weight=0.5)
            if restored_faces:
                return restored_faces[0]
        except Exception as e:
            print(f"Enhancement failed: {e}")
        
        return frame
    
    def enhance_video(self, video_path: str, output_path: str) -> str:
        """Enhance all frames in video"""
        if not self.available:
            print("⚠️  GFPGAN not available, skipping enhancement")
            return video_path
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            enhanced = self.enhance_frame(frame)
            out.write(enhanced)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  ✓ Enhanced {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"✅ Enhanced video: {output_path}")
        return output_path
