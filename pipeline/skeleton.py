import cv2
import json
import numpy as np
from typing import Dict
from pathlib import Path
from DWPoses import DWposeDetector
from PIL import Image


class SkeletonExtractor:
    MAX_SIZE = 768

    def __init__(self):
        self.detector = DWposeDetector()

    def _detect_single(self, frame_rgb: np.ndarray) -> Dict:
        """Wraps DWPoses' detector call, fabricating the score arrays our package
        doesn't provide (real echomimic_v2's own detector does). Safe because we
        only ever detect one person, so real code's multi-person score-based
        selection logic never actually triggers on this data."""
        pose = self.detector(Image.fromarray(frame_rgb))
        n_people = pose["bodies"]["subset"].shape[0]
        pose["bodies"]["score"] = np.ones((n_people, 18), dtype=np.float32)
        n_faces = pose["faces"].shape[0]
        pose["faces_score"] = np.ones((n_faces, 68), dtype=np.float32)
        pose["hands_score"] = np.ones((2, 21), dtype=np.float32)
        return pose

    @staticmethod
    def _resize_and_pad_param(imh, imw, max_size):
        """Ported verbatim from the real demo.ipynb."""
        half = max_size // 2
        if imh > imw:
            imh_new = max_size
            imw_new = int(round(imw / imh * imh_new))
            half_w = imw_new // 2
            rb, re = 0, max_size
            cb = half - half_w
            ce = cb + imw_new
        else:
            imw_new = max_size
            imh_new = int(round(imh / imw * imw_new))
            imh_new = max_size
            half_h = imh_new // 2
            cb, ce = 0, max_size
            rb = half - half_h
            re = rb + imh_new
        return imh_new, imw_new, rb, re, cb, ce

    @staticmethod
    def _resize_and_pad(img, max_size):
        """Ported verbatim from the real demo.ipynb."""
        img_new = np.zeros((max_size, max_size, 3)).astype("uint8")
        imh, imw = img.shape[0], img.shape[1]
        half = max_size // 2
        if imh > imw:
            imh_new = max_size
            imw_new = int(round(imw / imh * imh_new))
            half_w = imw_new // 2
            rb, re = 0, max_size
            cb = half - half_w
            ce = cb + imw_new
        else:
            imw_new = max_size
            imh_new = int(round(imh / imw * imw_new))
            half_h = imh_new // 2
            cb, ce = 0, max_size
            rb = half - half_h
            re = rb + imh_new
        img_resize = cv2.resize(img, (imw_new, imh_new))
        img_new[rb:re, cb:ce, :] = img_resize
        return img_new

    def _get_pose_params(self, detected_poses, height, width, max_size=MAX_SIZE):
        """Ported from demo.ipynb's get_pose_params — computes ONE crop box
        aggregated across ALL frames (min/max/mean), not per-frame."""
        w_min_all, w_max_all, h_min_all, h_max_all, mid_all = [], [], [], [], []

        for num, dp in enumerate(detected_poses):
            dp["num"] = num
            candidate_body = dp["bodies"]["candidate"]
            candidate_face = dp["faces"][0] if dp["faces"].shape[0] >= 1 else dp["faces"]

            body_pose = candidate_body
            mid_ = body_pose[1, 0]  # neck x-coordinate
            face_pose = candidate_face

            h_min, h_max = np.min(face_pose[:, 1]), np.max(body_pose[:7, 1])
            h_ = h_max - h_min
            mid_w = mid_
            w_min = mid_w - h_ / 2
            w_max = mid_w + h_ / 2

            w_min_all.append(w_min)
            w_max_all.append(w_max)
            h_min_all.append(h_min)
            h_max_all.append(h_max)
            mid_all.append(mid_w)

        w_min = np.min(w_min_all)
        w_max = np.max(w_max_all)
        h_min = np.min(h_min_all)
        h_max = np.max(h_max_all)
        mid = np.mean(mid_all)

        margin_ratio = 0.25
        h_margin = (h_max - h_min) * margin_ratio
        h_min = max(h_min - h_margin * 0.65, 0)
        h_max = min(h_max + h_margin * 0.05, 1)

        h_min_real = int(h_min * height)
        h_max_real = int(h_max * height)
        mid_real = int(mid * width)

        height_new = h_max_real - h_min_real + 1
        width_new = height_new
        w_min_real = mid_real - width_new // 2
        if w_min_real < 0:
            w_min_real = 0
            width_new = mid_real * 2
        w_max_real = w_min_real + width_new
        w_min = w_min_real / width
        w_max = w_max_real / width

        imh_new, imw_new, rb, re, cb, ce = self._resize_and_pad_param(height_new, width_new, max_size)
        return {
            "draw_pose_params": [imh_new, imw_new, rb, re, cb, ce],
            "pose_params": [w_min, w_max, h_min, h_max],
            "video_params": [h_min_real, h_max_real, w_min_real, w_max_real],
        }

    def align_reference_image(self, photo_path: str, output_path: str = "/tmp/aligned_ref.png") -> str:
        """Ported from demo.ipynb's get_img_pose + save_aligned_img. Replaces a
        naive full-image resize with a real face/shoulder-centered square crop."""
        frame = cv2.imread(photo_path)
        height, width, _ = frame.shape
        short_size = min(height, width)
        resize_ratio = max(self.MAX_SIZE / short_size, 1.0)
        frame = cv2.resize(frame, (int(resize_ratio * width), int(resize_ratio * height)))
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected = [self._detect_single(rgb)]

        res_params = self._get_pose_params(detected, height, width, self.MAX_SIZE)
        h_min_real, h_max_real, w_min_real, w_max_real = res_params["video_params"]
        img = frame[h_min_real:h_max_real, w_min_real:w_max_real, :]
        aligned = self._resize_and_pad(img, self.MAX_SIZE)
        cv2.imwrite(output_path, aligned)
        print(f"✅ Aligned reference image saved: {output_path}")
        return output_path

    def extract_and_align_skeleton(self, video_path: str, output_dir: str = "/tmp/pose_npy") -> str:
        """Ported from demo.ipynb's get_video_pose + get_pose_params + save_pose_params_item.
        Replaces the old extract_skeleton()+export_echomimic_npy() pair — crop params
        are now REAL (from actual keypoints, aggregated across the whole clip), not
        a hardcoded full-canvas guess."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        height, width, _ = frames[0].shape
        print(f"📹 Detecting pose on {len(frames)} frames...")
        detected_poses = []
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_poses.append(self._detect_single(rgb))
            if (i + 1) % 30 == 0:
                print(f"  ✓ {i + 1}/{len(frames)} frames...")

        print("Computing crop/alignment params across full clip...")
        res_params = self._get_pose_params(detected_poses, height, width, self.MAX_SIZE)
        w_min, w_max, h_min, h_max = res_params["pose_params"]
        draw_pose_params = res_params["draw_pose_params"]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for dp in detected_poses:
            num = dp["num"]
            candidate_body = dp["bodies"]["candidate"].copy()
            candidate_face = dp["faces"][0].copy() if dp["faces"].shape[0] >= 1 else dp["faces"].copy()
            candidate_hand = dp["hands"].copy()

            candidate_body[:, 0] = (candidate_body[:, 0] - w_min) / (w_max - w_min)
            candidate_body[:, 1] = (candidate_body[:, 1] - h_min) / (h_max - h_min)
            candidate_face[:, 0] = (candidate_face[:, 0] - w_min) / (w_max - w_min)
            candidate_face[:, 1] = (candidate_face[:, 1] - h_min) / (h_max - h_min)
            candidate_hand[:, :, 0] = (candidate_hand[:, :, 0] - w_min) / (w_max - w_min)
            candidate_hand[:, :, 1] = (candidate_hand[:, :, 1] - h_min) / (h_max - h_min)

            dp_out = {
                "bodies": {"candidate": candidate_body, "subset": dp["bodies"]["subset"]},
                "faces": candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1]),
                "hands": candidate_hand,
                "hands_score": dp["hands_score"],
                "faces_score": dp["faces_score"],
                "num": num,
                "draw_pose_params": draw_pose_params,
            }
            np.save(out_dir / f"{num}.npy", dp_out, allow_pickle=True)

        print(f"💾 Exported {len(detected_poses)} aligned pose .npy files to {out_dir}")
        return str(out_dir)
