import numpy as np
import torch
from pathlib import Path
from pipeline.skeleton import SkeletonExtractor
from pipeline.animation import AnimationEngine
from pipeline.enhance import FaceEnhancer
from pipeline.video_ops import VideoOperations

_animation_engine = None

def extract_skeleton_cpu(avatar_photo_path: str, driving_video_path: str):
    """CPU-only stage. No GPU needed, no quota consumed. Now does REAL alignment
    (photo crop + pose coordinate rescaling) matching demo.ipynb's actual logic,
    not a naive resize + hardcoded crop guess."""
    print("[Stage 1/4] Aligning reference photo + extracting skeleton (CPU)...")
    extractor = SkeletonExtractor()
    aligned_photo = extractor.align_reference_image(avatar_photo_path)
    npy_dir = extractor.extract_and_align_skeleton(driving_video_path)
    return aligned_photo, npy_dir

def _tensor_to_bgr_frames(video_tensor):
    video = video_tensor[0]
    video = video.permute(1, 2, 3, 0)
    video = (video.clamp(0, 1) * 255).byte().cpu().numpy()
    frames = [frame[..., ::-1] for frame in video]
    return frames

def generate_animation_gpu(aligned_photo_path: str, skeleton_path: str, audio_path: str, force_fp32: bool = False) -> str:
    global _animation_engine
    print("[Stage 2/4] Generating half-body animation (GPU)...")

    # Free any previously loaded model before loading a new one — prevents the
    # OOM crash we hit from running two generations in one Kaggle session.
    if _animation_engine is not None:
        del _animation_engine
        torch.cuda.empty_cache()
        _animation_engine = None

    dtype = torch.float32 if force_fp32 else None
    _animation_engine = AnimationEngine(dtype=dtype)
    _animation_engine.load_models()

    fps = 24
    video_tensor = _animation_engine.animate(aligned_photo_path, skeleton_path, audio_path, fps=fps)
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
    aligned_photo, skeleton_output = extract_skeleton_cpu(avatar_path, video_path)
    animation_output = generate_animation_gpu(aligned_photo, skeleton_output, audio_path, force_fp32=force_fp32)
    if enhance_face:
        animation_output = enhance_face_gpu(animation_output, audio_path)
    else:
        print("[Stage 3/4] Face enhancement skipped")
    print("[Stage 4/4] Finalizing output...")
    print(f"Pipeline complete: {animation_output}")
    return animation_output
