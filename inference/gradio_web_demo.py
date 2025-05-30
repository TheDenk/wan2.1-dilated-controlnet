import argparse
import os
import threading
import time

import gradio as gr
import torch
from diffusers.utils import export_to_video, load_video
from transformers import UMT5EncoderModel, T5Tokenizer
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip
from controlnet_aux import HEDdetector, CannyDetector, MidasDetector

from wan_controlnet import WanControlnet
from wan_transformer import CustomWanTransformer3DModel
from wan_controlnet_pipeline import WanControlnetPipeline


os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
    'depth': MidasDetector,
}

def init_controlnet_processor(controlnet_type):
    if controlnet_type in ['canny', 'lineart']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    clip = clip.set_fps(16)
    clip = clip.resize(height=480)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=16)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

def main(args):
    controlnet_processor = init_controlnet_processor(args.controlnet_type)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(args.base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    transformer = CustomWanTransformer3DModel.from_pretrained(args.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    # flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
    # scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)

    controlnet = WanControlnet.from_pretrained(args.controlnet_model_path, torch_dtype=torch.bfloat16)
    pipe = WanControlnetPipeline(
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae, 
        controlnet=controlnet,
        scheduler=scheduler,
    )
    pipe = pipe.to(device="cuda")
    pipe.enable_model_cpu_offload()
    
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / args.lora_rank)


    def infer(
            prompt: str, negative_prompt: str, controlnet_frames: list, num_inference_steps: int, guidance_scale: float, seed: int, width: int, height: int, num_frames: int, 
            controlnet_guidance_start: float, controlnet_guidance_end: float, controlnet_weight: float, controlnet_stride: int, teacache_treshold: float, progress=gr.Progress(track_tqdm=True)
        ):
        torch.cuda.empty_cache()

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pil",
        
            controlnet_frames=controlnet_frames,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
            controlnet_weight=controlnet_weight,
            controlnet_stride=controlnet_stride,

            teacache_treshold=float(teacache_treshold.value if hasattr(teacache_treshold, 'value') else teacache_treshold),
        ).frames[0]

        return output

    with gr.Blocks() as demo:
        gr.Markdown("""
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                Dilated Controlnet for Wan2.1 Space🤗
                """)

        with gr.Row():
            with gr.Column():
                with gr.Column():
                    video_input = gr.Video(label="Video for controlnet processing", width=720, height=720)
                    with gr.Row():
                        download_video_button = gr.File(label="📥 Download Video", visible=False)
                        download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (Less than 200 Words)", 
                    value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", 
                    placeholder="Enter your prompt here", 
                    lines=3
                )

                with gr.Column():
                    gr.Markdown(
                        "**Optional Parameters** (default values are recommended)<br>"
                        "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                        "50 steps are recommended for most cases.<br>"
                    )
                    with gr.Row():
                        width = gr.Number(label="Output Video Width", value=832, step=16)
                        height = gr.Number(label="Output Video Height", value=480, step=16)
                        num_frames = gr.Number(label="Output Frames Count", value=81, step=1)
                    with gr.Row():
                        num_inference_steps = gr.Number(label="Inference Steps", value=50, step=1)
                        guidance_scale = gr.Number(label="Guidance Scale", value=5.0, step=0.05)
                        seed = gr.Number(label="Seed", value=42, step=1)
                    with gr.Row():
                        controlnet_guidance_start = gr.Number(label="Controlnet Guidance Start", interactive=True, precision=2, value=0.0, minimum=0.0, maximum=1.0, step=0.05)
                        controlnet_guidance_end = gr.Number(label="Controlnet Guidance End", interactive=True, precision=2, value=0.8, minimum=0.0, maximum=1.0, step=0.05)
                        controlnet_weight = gr.Number(label="Controlnet Weight", interactive=True, value=0.8, precision=2, minimum=0.0, maximum=1.0, step=0.05)
                        controlnet_stride = gr.Number(label="Controlnet Stride", interactive=True, value=3, minimum=1, step=1)
                    with gr.Row():
                        teacache_treshold = gr.Number(label="TeaCache Treshold. Less coef -> Better quality, but longer inference.", interactive=True, value=0.3, precision=2, minimum=0.0, maximum=1.5, step=0.05)
                    
                    generate_button = gr.Button("🎬 Generate Video")

            with gr.Column():
                video_output = gr.Video(label="Generate Video", width=720, height=720)
                with gr.Row():
                    download_video_button = gr.File(label="📥 Download Video", visible=False)
                    download_gif_button = gr.File(label="📥 Download GIF", visible=False)

        def generate(prompt, negative_prompt, video_input, num_inference_steps, guidance_scale, seed, width, height, num_frames, 
                controlnet_guidance_start, controlnet_guidance_end, controlnet_weight, controlnet_stride, teacache_treshold, progress=gr.Progress(track_tqdm=True)):
            video = load_video(video_input)[:num_frames]
            controlnet_frames = [controlnet_processor(x) for x in video]
            tensor = infer(
                prompt, negative_prompt, controlnet_frames, num_inference_steps, guidance_scale, seed, width, height, num_frames, 
                controlnet_guidance_start, controlnet_guidance_end, controlnet_weight, controlnet_stride, teacache_treshold, progress=progress
            )
            video_path = save_video(tensor)
            video_update = gr.update(visible=True, value=video_path)
            gif_path = convert_to_gif(video_path)
            gif_update = gr.update(visible=True, value=gif_path)

            return video_path, video_update, gif_update

        generate_button.click(
            generate,
            inputs=[prompt, negative_prompt, video_input, num_inference_steps, guidance_scale, seed, width, height, num_frames, 
                controlnet_guidance_start, controlnet_guidance_end, controlnet_weight, controlnet_stride, teacache_treshold, ],
            outputs=[video_output, download_video_button, download_gif_button],
        )
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan2.1")
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_type", type=str, default='hed', help="Type of controlnet model (e.g. canny, hed)")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    args = parser.parse_args()
    main(args)
