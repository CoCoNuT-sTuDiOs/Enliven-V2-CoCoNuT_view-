
import gradio as gr
import os
from pipeline.orchestrator import run_animation

def process_animation(avatar_image, driving_video, enhance_face):
    """Process animation request"""
    try:
        if avatar_image is None or driving_video is None:
            return None, "❌ Please upload both avatar photo and driving video"
        
        status = "🔄 Processing..."
        output_video = run_animation(avatar_image, driving_video, enhance_face=enhance_face)
        
        status = "✅ Animation complete!"
        return output_video, status
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Build Gradio interface
with gr.Blocks(title="Enliven v2 - Half-Body Animation") as demo:
    gr.Markdown("""
    # 🎬 Enliven v2 - Half-Body Avatar Animation
    
    Transform your static photo into an animated character using real motion reference.
    
    **How it works:**
    1. Upload a static photo of yourself (avatar)
    2. Upload a video of yourself performing the motion
    3. Enliven will animate your avatar based on the motion in the video
    
    **Pro Tip:** For best results, match the avatar photo setting to the driving video (same clothing, similar pose).
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Avatar Photo")
            avatar_input = gr.Image(label="Your static photo", type="filepath", scale=1)
        
        with gr.Column():
            gr.Markdown("### 🎥 Driving Video")
            video_input = gr.Video(label="Your motion reference", scale=1)
    
    with gr.Row():
        enhance_toggle = gr.Checkbox(label="✨ Enhance face quality (GFPGAN)", value=False)
    
    generate_btn = gr.Button("🚀 Generate Animation", variant="primary", scale=2)
    
    with gr.Row():
        output_video = gr.Video(label="🎬 Animated Output", scale=1)
        status_text = gr.Textbox(label="Status", interactive=False, scale=1)
    
    gr.Markdown("""
    ### ⚡ Processing Time
    - Skeleton extraction: 2-3 min
    - Animation generation: 8-12 min
    - Face enhancement: 5-10 min (optional)
    - **Total: ~15-20 minutes on RTX 6000**
    
    ### 📝 Tips
    - Avatar and driving video work best with clear lighting
    - Both should have similar background or use green screen
    - Motion should be natural and smooth
    - Avoid extreme angles or rapid movements
    """)
    
    generate_btn.click(
        process_animation,
        inputs=[avatar_input, video_input, enhance_toggle],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
