import json
import numpy as np
import torch
from pathlib import Path
from pipeline.skeleton import SkeletonExtractor
from pipeline.animation import AnimationEngine
from pipeline.enhance import FaceEnhancer
from pipeline.video_ops import VideoOperations

_animation_engine = None

def extract_skeleton_cpu(driving_video_path: str) -> str:
    print("[Stage 1/4] Extracting skeleton from driving video (CPU)...")
    extractor = SkeletonExtractor()
    skeleton = extractor.extract_skeleton(driving_video_path)
    extractor.save_skeleton(skeleton, "/tmp/skeleton.json")

    npy_dir = "/tmp/pose_npy"
    extractor.export_echomimic_npy(skeleton, npy_dir)
    return npy_dir

def _tensor_to_bgr_frames(video_tensor):
    video = video_tensor[0]
    video = video.permute(1, 2, 3, 0)
    video = (video.clamp(0, 1) * 255).byte().cpu().numpy()
    frames = [frame[..., ::-1] for frame in video]
    return frames

def generate_animation_gpu(avatar_photo_path: str, skeleton_path: str, audio_path: str, force_fp32: bool = False) -> str:
    """GPU stage. Only this should be wrapped in @spaces.GPU by the caller.
    force_fp32: DIAGNOSTIC flag to test the fp16-NaN-instability hypothesis.
    """
    global _animation_engine
    print("[Stage 2/4] Generating half-body animation (GPU)...")
    dtype = torch.float32 if force_fp32 else None
    if _animation_engine is None or force_fp32:
        _animation_engine = AnimationEngine(dtype=dtype)
    _animation_engine.load_models()

    fps = 24
    video_tensor = _animation_engine.animate(avatar_photo_path, skeleton_path, audio_path, fps=fps)
    frames = _tensor_to_bgr_frames(video_tensor)

    silent_path = "/tmp/animation_silent.mp4"
    VideoOperations.encode_frames_to_video(frames, silent_path, fps=fps)

    final_path = "/tmp/animation_raw.mp4"
    VideoOperations.mux_audio(silent_path, audio_path, final_path, fps=fps)
    return final_path

def enhance_face_gpu(video_path: str, audio_path: str) -> str:
    print("[Stage 3/4] Enhancing face quality (GPU)...")
    face_enhancer = FaceEnhancer()
    fps = VideoOperations.get_video_info(video_path)["fps"]
    silent_enhanced_path = "/tmp/animation_enhanced_silent.mp4"
    face_enhancer.enhance_video(video_path, silent_enhanced_path)
    final_enhanced_path = "/tmp/animation_enhanced.mp4"
    VideoOperations.mux_audio(silent_enhanced_path, audio_path, final_enhanced_path, fps=fps)
    return final_enhanced_path

def run_animation(avatar_path: str, video_path: str, audio_path: str, enhance_face: bool = False, force_fp32: bool = False) -> str:
    print("=" * 60)
    print("ENLIVEN v2 - HALF-BODY ANIMATION PIPELINE")
    print("=" * 60)
    skeleton_output = extract_skeleton_cpu(video_path)
    animation_output = generate_animation_gpu(avatar_path, skeleton_output, audio_path, force_fp32=force_fp32)
    if enhance_face:
        animation_output = enhance_face_gpu(animation_output, audio_path)
    else:
        print("[Stage 3/4] Face enhancement skipped")
    print("[Stage 4/4] Finalizing output...")
    print(f"Pipeline complete: {animation_output}")
    return animation_output
