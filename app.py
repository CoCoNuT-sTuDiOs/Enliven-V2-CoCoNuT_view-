
import os
import gradio as gr
from pathlib import Path
import spaces

def download_weights():
    weights_dir = Path("pretrained_weights")
    if not weights_dir.exists():
        print("⏳ Downloading weights...")
        os.system("mkdir -p pretrained_weights")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/denoising_unet.pth")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/reference_unet.pth")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/motion_module.pth")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/pose_encoder.pth")
        print("✅ Weights ready")

download_weights()

from pipeline.orchestrator import run_animation

@spaces.GPU(duration=300)
def process_animation(avatar_image, driving_video, enhance_face):
    try:
        if avatar_image is None or driving_video is None:
            return None, "❌ Upload both files"
        output_video = run_animation(avatar_image, driving_video, enhance_face=enhance_face)
        return output_video, "✅ Complete!"
    except Exception as e:
        return None, f"❌ {str(e)}"

with gr.Blocks(title="Enliven v2") as demo:
    gr.Markdown("# 🎬 Enliven v2 - Half-Body Avatar Animation")
    
    with gr.Row():
        avatar_input = gr.Image(label="📷 Avatar", type="filepath")
        video_input = gr.Video(label="🎥 Driving Video")
    
    enhance_toggle = gr.Checkbox(label="✨ Enhance face", value=False)
    generate_btn = gr.Button("🚀 Generate", variant="primary")
    
    output_video = gr.Video(label="🎬 Output")
    status_text = gr.Textbox(label="Status", interactive=False)
    
    generate_btn.click(
        process_animation,
        inputs=[avatar_input, video_input, enhance_toggle],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
