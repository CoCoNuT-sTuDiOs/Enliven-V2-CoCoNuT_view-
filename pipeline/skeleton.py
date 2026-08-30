import cv2
import json
import numpy as np
from typing import Dict
from pathlib import Path
from DWPoses import DWposeDetector
from PIL import Image


class SkeletonExtractor:
    def __init__(self):
        self.detector = DWposeDetector()

    def extract_skeleton(self, video_path: str) -> Dict:
        """Extract 2D skeleton from video using DWPoses.
        Keeps RAW NORMALIZED [0,1] coords — draw_handpose() scales by W/H itself,
        so pre-multiplying here would double-scale everything.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        poses = []
        frame_id = 0
        w = h = None

        print(f"📹 Processing with DWPoses: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.detector(Image.fromarray(rgb_frame))

            hands_raw = pose_result.get("hands", [])
            hands = np.full((2, 21, 2), -1.0, dtype=np.float32)
            hands_score = np.zeros((2, 21), dtype=np.float32)

            for hand_idx, hand in enumerate(hands_raw[:2]):
                for pt_idx, kp in enumerate(hand[:21]):
                    x, y = float(kp[0]), float(kp[1])
                    if x != -1 and y != -1:
                        hands[hand_idx, pt_idx] = [x, y]
                        hands_score[hand_idx, pt_idx] = 1.0

            # BUG FIX: bodies_candidate can be a numpy array straight from DWPoses'
            # output — json.dump() can't serialize ndarrays. Force it to a plain
            # list, same as we already do for hands/hands_score.
            bodies_candidate = pose_result.get("bodies", {}).get("candidate", [])
            if isinstance(bodies_candidate, np.ndarray):
                bodies_candidate = bodies_candidate.tolist()

            poses.append({
                "frame_id": frame_id,
                "hands": hands.tolist(),
                "hands_score": hands_score.tolist(),
                "bodies_candidate": bodies_candidate,
            })
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

    def export_echomimic_npy(self, skeleton: Dict, output_dir: str, target_w: int = 768, target_h: int = 768):
        """Convert extracted skeleton into per-frame .npy files matching EchoMimicV2's
        expected dict format. One file per frame: 0.npy, 1.npy, ...
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        draw_pose_params = (target_h, target_w, 0, target_h, 0, target_w)

        for frame in skeleton["poses"]:
            pose_dict = {
                "bodies": {"candidate": np.array(frame.get("bodies_candidate", [])), "subset": np.array([])},
                "hands": np.array(frame["hands"], dtype=np.float32),
                "hands_score": np.array(frame["hands_score"], dtype=np.float32),
                "faces": np.zeros((0, 68, 2), dtype=np.float32),
                "faces_score": np.zeros((0, 68), dtype=np.float32),
                "num": 1,
                "draw_pose_params": draw_pose_params,
            }
            np.save(out_dir / f"{frame['frame_id']}.npy", pose_dict, allow_pickle=True)

        print(f"💾 Exported {len(skeleton['poses'])} pose .npy files to {out_dir}")
