import cv2
import numpy as np
from typing import Optional

# Confirmed from GFPGAN's own official inference_gfpgan.py: this is the real URL
# for v1.4 weights (yes, hosted under the v1.3.0 release tag — that's genuinely
# how TencentARC tagged it). GFPGANer downloads+caches this itself, no manual
# wget step needed.
GFPGAN_V14_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"


class FaceEnhancer:
    def __init__(self):
        try:
            from gfpgan import GFPGANer
            self.enhancer = GFPGANer(
                model_path=GFPGAN_V14_URL,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
            )
            self.available = True
        except Exception as e:
            print(f"GFPGAN not available: {e}")
            self.available = False

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance face quality in frame"""
        if not self.available:
            return frame

        try:
            # enhance() returns (cropped_faces, restored_faces, restored_img).
            # restored_faces[0] is just the cropped face crop — using it here was
            # a real bug that would replace the whole frame with a tiny face patch.
            # restored_img is the full frame with the restoration pasted back in.
            _, _, restored_img = self.enhancer.enhance(
                frame, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5
            )
            if restored_img is not None:
                return restored_img
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
