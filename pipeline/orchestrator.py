
import json
from pathlib import Path
from pipeline.skeleton import SkeletonExtractor
from pipeline.animation import AnimationGenerator
from pipeline.enhance import FaceEnhancer
from pipeline.video_ops import VideoOperations

class EnlivenPipeline:
    def __init__(self):
        self.skeleton_extractor = SkeletonExtractor()
        self.animation_gen = AnimationGenerator()
        self.face_enhancer = FaceEnhancer()
    
    def run(self, avatar_photo_path: str, driving_video_path: str, enhance_face: bool = False) -> str:
        """Run full Enliven v2 pipeline"""
        
        print("\n" + "="*60)
        print("🎬 ENLIVEN v2 - HALF-BODY ANIMATION PIPELINE")
        print("="*60)
        
        # Stage 1: Extract skeleton
        print("\n[Stage 1/4] Extracting skeleton from driving video...")
        skeleton_output = "/kaggle/working/outputs/skeleton.json"
        skeleton = self.skeleton_extractor.extract_skeleton(driving_video_path)
        self.skeleton_extractor.save_skeleton(skeleton, skeleton_output)
        
        # Stage 2: Generate animation
        print("\n[Stage 2/4] Generating half-body animation...")
        animation_output = self.animation_gen.animate(avatar_photo_path, skeleton_output)
        
        # Stage 3: Enhance (optional)
        if enhance_face:
            print("\n[Stage 3/4] Enhancing face quality...")
            enhanced_output = "/kaggle/working/outputs/animation_enhanced.mp4"
            self.face_enhancer.enhance_video(animation_output, enhanced_output)
            animation_output = enhanced_output
        else:
            print("\n[Stage 3/4] Face enhancement skipped")
        
        # Stage 4: Output
        print("\n[Stage 4/4] Finalizing output...")
        print(f"✅ Pipeline complete!")
        print(f"📄 Output video: {animation_output}")
        
        return animation_output

def run_animation(avatar_path: str, video_path: str, enhance_face: bool = False) -> str:
    """Public API for Gradio UI"""
    pipeline = EnlivenPipeline()
    return pipeline.run(avatar_path, video_path, enhance_face)
