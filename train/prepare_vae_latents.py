import os
import glob
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from decord import VideoReader
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor


def main(args):   
    vae_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda") if args.device == "cuda" else "cpu"
    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=vae_dtype).to(device=device)
    video_processor = VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(f"MODEL HAS BEEN LOADED TO {device}")
    
    os.makedirs(args.out_latents_dir, exist_ok=True)
    input_video_paths = glob.glob(os.path.join(args.input_video_dir, "*.mp4"))

    for input_video_path in tqdm(input_video_paths, total=len(input_video_paths)):
        basename = os.path.basename(input_video_path)

        video_reader = VideoReader(input_video_path)
        video_length = len(video_reader)
        
        clip_length = min(video_length, (args.sample_n_frames - 1) * args.sample_stride + 1)
        batch_index = np.linspace(0, clip_length - 1, args.sample_n_frames, dtype=int)
        np_video = video_reader.get_batch(batch_index).asnumpy()
        del video_reader

        preprocessed_video = video_processor.preprocess(
            [Image.fromarray(x) for x in np_video], 
            height=args.height, 
            width=args.width
        ).permute(1, 0, 2, 3).unsqueeze(0) .to(device, dtype=vae_dtype)
        with torch.no_grad():
            latents = vae.encode(preprocessed_video).latent_dist.sample(generator)
        
        out_latents_path = os.path.join(args.out_latents_dir, basename.replace(".mp4", ".pt"))
        torch.save(latents, out_latents_path)


# CUDA_VISIBLE_DEVICES=0 python prepare_vae_latents.py \
# --input_video_dir "path to input video dir" \
# --out_latents_dir "dir for output latents" \
# --base_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
# --sample_stride 2 \
# --width 832 \
# --height 480 \
# --sample_n_frames 81 \
# --seed 42 \
# --device "cuda" \
# --dtype "fp32"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate input latents for training.")
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The path of the pre-trained model with vae model"
    )
    parser.add_argument("--input_video_dir", type=str, required=True, help="Directory with video for processing")
    parser.add_argument("--out_latents_dir", type=str, required=True, help="Directory for latents")
    parser.add_argument("--sample_stride", type=int, default=2, help="get each N frame")
    parser.add_argument("--width", type=int, default=832, help="width for preprocessor")
    parser.add_argument("--height", type=int, default=480, help="height for preprocessor")
    parser.add_argument("--sample_n_frames", type=int, default=81, help="total frames count")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32 or fp16")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    main(args)