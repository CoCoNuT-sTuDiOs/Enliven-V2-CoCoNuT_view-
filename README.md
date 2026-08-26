---
title: Enliven V2
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🎬 Enliven v2 - Half-Body Avatar Animation

**Bring your static photos to life with realistic motion transfer.**

Transform a static avatar photo into an animated half-body video using real motion reference from a driving video.

## Features

✅ Half-body animation - Arms, hands, torso with natural motion  
✅ Hand-aware - Explicit hand pose conditioning prevents hand melting  
✅ Motion transfer - Real video → skeleton → animation (not AI-guessed)  
✅ Open-source - DWPose + EchoMimicV2 + GFPGAN  

## How to Use

1. Upload avatar photo - Static image of yourself
2. Upload driving video - Video of yourself doing the motion
3. Click Generate - Wait 5-15 minutes
4. Download result - Animated MP4 video

## Technical Stack

- Skeleton extraction: DWPose (2D keypoints + hand landmarks)
- Animation: EchoMimicV2 (diffusion-based)
- Enhancement: GFPGAN v1.3 (optional)
- UI: Gradio + HF Spaces

## License

Apache 2.0 - Open source

## Author

CoCoNuT sTuDiOs (Darwin)

**Start animating!** Upload your photos above.
