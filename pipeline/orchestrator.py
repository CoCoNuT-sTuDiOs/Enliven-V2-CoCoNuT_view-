import json
import numpy as np
from pathlib import Path
from pipeline.skeleton import SkeletonExtractor
from pipeline.animation import AnimationEngine
from pipeline.enhance import FaceEnhancer
from pipeline.video_ops import VideoOperations

_animation_engine = None  # module-level singleton so load_models() only runs once per process

def extract_skeleton_cpu(driving_video_path: str) -> str:
    """CPU-only stage. No GPU needed, no quota consumed. Safe to run anytime.
    Returns a directory of per-frame .npy files (EchoMimicV2's expected pose format).
    """
    print("[Stage 1/4] Extracting skeleton from driving video (CPU)...")
    extractor = SkeletonExtractor()
    skeleton = extractor.extract_skeleton(driving_video_path)
    extractor.save_skeleton(skeleton, "/tmp/skeleton.json")

    npy_dir = "/tmp/pose_npy"
    extractor.export_echomimic_npy(skeleton, npy_dir)
    return npy_dir

def _tensor_to_bgr_frames(video_tensor):
    """animate() returns shape (1, C, T, H, W), float values in [0,1].
    encode_frames_to_video() needs a list of (H, W, 3) uint8 BGR numpy arrays."""
    video = video_tensor[0]  # (C, T, H, W)
    video = video.permute(1, 2, 3, 0)  # (T, H, W, C)
    video = (video.clamp(0, 1) * 255).byte().cpu().numpy()  # uint8, RGB order
    frames = [frame[..., ::-1] for frame in video]  # RGB -> BGR per frame for cv2.VideoWriter
    return frames

def generate_animation_gpu(avatar_photo_path: str, skeleton_path: str, audio_path: str) -> str:
    """GPU stage. Only this should be wrapped in @spaces.GPU by the caller."""
    global _animation_engine
    print("[Stage 2/4] Generating half-body animation (GPU)...")
    if _animation_engine is None:
        _animation_engine = AnimationEngine()
    _animation_engine.load_models()

    fps = 24  # must match animate()'s fps= arg below
    video_tensor = _animation_engine.animate(avatar_photo_path, skeleton_path, audio_path, fps=fps)
    frames = _tensor_to_bgr_frames(video_tensor)

    silent_path = "/tmp/animation_silent.mp4"
    VideoOperations.encode_frames_to_video(frames, silent_path, fps=fps)

    final_path = "/tmp/animation_raw.mp4"
    VideoOperations.mux_audio(silent_path, audio_path, final_path, fps=fps)
    return final_path

def enhance_face_gpu(video_path: str) -> str:
    """GPU stage. Only this should be wrapped in @spaces.GPU by the caller."""
    print("[Stage 3/4] Enhancing face quality (GPU)...")
    face_enhancer = FaceEnhancer()
    enhanced_output = "/tmp/animation_enhanced.mp4"
    face_enhancer.enhance_video(video_path, enhanced_output)
    return enhanced_output

def run_animation(avatar_path: str, video_path: str, audio_path: str, enhance_face: bool = False) -> str:
    """
    Public API for Gradio UI - orchestrates CPU and GPU stages separately
    so GPU quota is only spent on stages that actually need it.
    """
    print("=" * 60)
    print("ENLIVEN v2 - HALF-BODY ANIMATION PIPELINE")
    print("=" * 60)
    skeleton_output = extract_skeleton_cpu(video_path)
    animation_output = generate_animation_gpu(avatar_path, skeleton_output, audio_path)
    if enhance_face:
        animation_output = enhance_face_gpu(animation_output)
    else:
        print("[Stage 3/4] Face enhancement skipped")
    print("[Stage 4/4] Finalizing output...")
    print(f"Pipeline complete: {animation_output}")
    return animation_output
