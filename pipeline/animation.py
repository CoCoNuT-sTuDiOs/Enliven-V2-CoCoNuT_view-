import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler

from vendor.echomimicv2.src.models.unet_2d_condition import UNet2DConditionModel
from vendor.echomimicv2.src.models.unet_3d_emo import EMOUNet3DConditionModel
from vendor.echomimicv2.src.models.whisper.audio2feature import load_audio_model
from vendor.echomimicv2.src.models.pose_encoder import PoseEncoder
from vendor.echomimicv2.src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from vendor.echomimicv2.src.utils.dwpose_util import draw_pose_select_v2


class AnimationEngine:
    """Wraps EchoMimicV2 model loading + inference. Load once, reuse across calls."""

    def __init__(self, config_path="config/infer.yaml", device="cuda", dtype=torch.float16):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.weight_dtype = dtype if self.device == "cuda" else torch.float32
        self.config = OmegaConf.load(config_path)
        self.infer_config = OmegaConf.load(self.config.inference_config)
        self.pipe = None

    def load_models(self):
        """Call ONLY inside the @spaces.GPU-wrapped function — never at import time
        or in the CPU-only skeleton extraction path."""
        if self.pipe is not None:
            return

        cfg, ic = self.config, self.infer_config
        device, dtype = self.device, self.weight_dtype

        vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_path).to(device, dtype=dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, subfolder="unet"
        ).to(device=device, dtype=dtype)
        reference_unet.load_state_dict(torch.load(cfg.reference_unet_path, map_location="cpu"))

        if not os.path.exists(cfg.motion_module_path):
            raise FileNotFoundError(f"motion module not found: {cfg.motion_module_path}")

        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs,
        ).to(device=device, dtype=dtype)
        denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)

        pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
            device=device, dtype=dtype
        )
        pose_net.load_state_dict(torch.load(cfg.pose_encoder_path))

        audio_processor = load_audio_model(model_path=cfg.audio_model_path, device=device)

        scheduler = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))

        self.pipe = EchoMimicV2Pipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_processor,
            pose_encoder=pose_net,
            scheduler=scheduler,
        ).to(device, dtype=dtype)

    def animate(self, photo, skeleton_path, audio_path, width=768, height=768,
                max_frames=240, steps=30, cfg_scale=2.5, fps=24, seed=3407):
        """
        skeleton_path: dir of per-frame .npy pose files in EchoMimicV2's format
                       (produced by the pose-conversion step — item #2, not yet built)
        audio_path: required — EchoMimicV2 has no audio-free mode
        """
        if self.pipe is None:
            raise RuntimeError("Call load_models() first (inside the @spaces.GPU wrapper)")

        from moviepy.editor import AudioFileClip

        generator = torch.manual_seed(seed)
        ref_image_pil = Image.open(photo).resize((width, height))

        audio_clip = AudioFileClip(audio_path)
        n_pose_frames = len(os.listdir(skeleton_path))
        length = min(max_frames, int(audio_clip.duration * fps), n_pose_frames)

        pose_list = []
        for idx in range(length):
            tgt_mask = np.zeros((width, height, 3)).astype("uint8")
            detected_pose = np.load(
                os.path.join(skeleton_path, f"{idx}.npy"), allow_pickle=True
            ).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
            im = np.transpose(np.array(im), (1, 2, 0))
            tgt_mask[rb:re, cb:ce, :] = im
            tgt_mask_pil = Image.fromarray(tgt_mask).convert("RGB")
            pose_list.append(
                torch.Tensor(np.array(tgt_mask_pil))
                .to(dtype=self.weight_dtype, device=self.device)
                .permute(2, 0, 1) / 255.0
            )
        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)

        video = self.pipe(
            ref_image_pil,
            audio_path,
            poses_tensor[:, :, :length, ...],
            width,
            height,
            length,
            steps,
            cfg_scale,
            generator=generator,
            audio_sample_rate=16000,
            context_frames=12,
            fps=fps,
            context_overlap=3,
            start_idx=0,
        ).videos

        final_length = min(video.shape[2], poses_tensor.shape[2], length)
        return video[:, :, :final_length, :, :]
