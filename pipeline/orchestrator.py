import json
from pathlib import Path
from pipeline.skeleton import SkeletonExtractor
from pipeline.animation import AnimationGenerator
from pipeline.enhance import FaceEnhancer
from pipeline.video_ops import VideoOperations

def extract_skeleton_cpu(driving_video_path: str) -> str:
    """CPU-only stage. No GPU needed, no quota consumed. Safe to run anytime."""
    print("[Stage 1/4] Extracting skeleton from driving video (CPU)...")
    extractor = SkeletonExtractor()
    skeleton_output = "/tmp/skeleton.json"
    skeleton = extractor.extract_skeleton(driving_video_path)
    extractor.save_skeleton(skeleton, skeleton_output)
    return skeleton_output

def generate_animation_gpu(avatar_photo_path: str, skeleton_path: str, audio_path: str) -> str:
    """GPU stage. Only this should be wrapped in @spaces.GPU by the caller."""
    print("[Stage 2/4] Generating half-body animation (GPU)...")
    animation_gen = AnimationGenerator()
    return animation_gen.animate(avatar_photo_path, skeleton_path, audio_path)

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
