import os
from gradio_client import utils as gradio_client_utils
_original_get_type = gradio_client_utils.get_type
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean" if schema else "None"
    return _original_get_type(schema)
gradio_client_utils.get_type = _safe_get_type
import gradio as gr
import spaces
from pathlib import Path
from huggingface_hub import snapshot_download

def download_weights():
    weights_dir = Path("pretrained_weights")
    if not weights_dir.exists():
        print("Downloading weights...")
        os.system("mkdir -p pretrained_weights")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/denoising_unet.pth")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/reference_unet.pth")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/motion_module.pth")
        os.system("cd pretrained_weights && wget -q https://huggingface.co/BadToBest/EchoMimicV2/resolve/main/pose_encoder.pth")

        # These 3 are multi-file diffusers folders (config.json + weights), not single
        # files, so wget-by-guessed-filename isn't safe — pull the exact subfolders
        # from the same BadToBest/EchoMimicV2 repo via snapshot_download instead.
        print("Downloading vae/base-model/audio-processor folders...")
        snapshot_download(
            repo_id="BadToBest/EchoMimicV2",
            allow_patterns=["sd-vae-ft-mse/*", "sd-image-variations-diffusers/*", "audio_processor/tiny.pt"],
            local_dir="pretrained_weights",
        )
        print("Weights ready")
download_weights()
from pipeline.orchestrator import run_animation, extract_skeleton_cpu
def process_skeleton_only(driving_video):
    """CPU-only test button - no GPU quota used."""
    try:
        if driving_video is None:
            return "Upload a driving video first"
        skeleton_path = extract_skeleton_cpu(driving_video)
        return f"Skeleton extracted successfully: {skeleton_path}"
    except Exception as e:
        return f"Error: {str(e)}"
@spaces.GPU(duration=300)
def process_animation(avatar_image, driving_video, audio_file, enhance_face):
    try:
        if avatar_image is None or driving_video is None or audio_file is None:
            return None, "Upload avatar photo, driving video, and audio file"
        output_video = run_animation(avatar_image, driving_video, audio_file, enhance_face=enhance_face)
        return output_video, "Complete"
    except Exception as e:
        return None, f"Error: {str(e)}"
with gr.Blocks(title="Enliven v2") as demo:
    gr.Markdown("# Enliven v2 - Half-Body Avatar Animation")
    gr.Markdown("Test skeleton extraction (CPU, free) separately from full generation (GPU, uses quota).")
    with gr.Tab("Test Skeleton Extraction (CPU only, free)"):
        with gr.Row():
            skel_video_input = gr.Video(label="Driving Video")
        skel_btn = gr.Button("Extract Skeleton (CPU)")
        skel_output = gr.Textbox(label="Result")
        skel_btn.click(process_skeleton_only, inputs=[skel_video_input], outputs=[skel_output])
    with gr.Tab("Full Generation (GPU, uses quota)"):
        with gr.Row():
            avatar_input = gr.Image(label="Avatar Photo", type="filepath")
            video_input = gr.Video(label="Driving Video")
        audio_input = gr.Audio(label="Audio (required by EchoMimicV2)", type="filepath")
        enhance_toggle = gr.Checkbox(label="Enhance face", value=False)
        generate_btn = gr.Button("Generate", variant="primary")
        output_video = gr.Video(label="Output")
        status_text = gr.Textbox(label="Status", interactive=False)
        generate_btn.click(
            process_animation,
            inputs=[avatar_input, video_input, audio_input, enhance_toggle],
            outputs=[output_video, status_text]
        )
if __name__ == "__main__":
    demo.queue()
    demo.launch()
