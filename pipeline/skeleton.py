
import cv2
import json
from typing import Dict
from DWPoses import DWposeDetector
from PIL import Image
import numpy as np

class SkeletonExtractor:
    def __init__(self):
        self.detector = DWposeDetector()
    
    def extract_skeleton(self, video_path: str) -> Dict:
        """Extract 2D skeleton from video using DWPoses"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        poses = []
        frame_id = 0
        
        print(f"📹 Processing with DWPoses: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            pose_result = self.detector(pil_image)
            
            frame_data = {
                "frame_id": frame_id,
                "body_keypoints": {},
                "hand_keypoints": {"left": [], "right": []}
            }
            
            # Body keypoints from candidates
            if "bodies" in pose_result and "candidate" in pose_result["bodies"]:
                candidates = pose_result["bodies"]["candidate"]
                for idx, kp in enumerate(candidates):
                    x = int(kp[0] * w)
                    y = int(kp[1] * h)
                    frame_data["body_keypoints"][f"j{idx}"] = {"x": x, "y": y}
            
            # Hand keypoints
            if "hands" in pose_result:
                hands = pose_result["hands"]
                if hands is not None and len(hands) > 0:
                    for hand_idx, hand in enumerate(hands):
                        hand_side = "left" if hand_idx == 0 else "right"
                        hand_points = [[int(kp[0] * w), int(kp[1] * h)] for kp in hand if kp[0] != -1 and kp[1] != -1]
                        frame_data["hand_keypoints"][hand_side] = hand_points
            
            poses.append(frame_data)
            frame_id += 1
            
            if frame_id % 30 == 0:
                print(f"  ✓ {frame_id} frames...")
        
        cap.release()
        print(f"✅ Done: {frame_id} frames")
        
        return {"fps": fps, "total_frames": frame_id, "width": w, "height": h, "poses": poses}
    
    def save_skeleton(self, skeleton: Dict, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(skeleton, f, indent=2)
        print(f"💾 Saved: {output_path}")
