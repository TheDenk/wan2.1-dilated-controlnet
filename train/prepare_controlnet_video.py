import os
import glob
import argparse

import cv2
import numpy as np
from tqdm.notebook import tqdm
from decord import VideoReader
from controlnet_aux import CannyDetector, HEDdetector, MidasDetector


def init_controlnet(controlnet_type):
    if controlnet_type in ['canny']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators')


def save_video(out_path, frames, fps):
    img_h, img_w = np.array(frames[0]).shape[:2]
    output = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_w, img_h))
    for frame in frames: 
        output.write(np.array(frame)) 
    output.release() 
    

controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
    'depth': MidasDetector,
}


def main(args):   
    controlnet_model = init_controlnet(args.controlnet_type)
    os.makedirs(args.out_controlnet_video_dir, exist_ok=True)
    input_video_paths = glob.glob(os.path.join(args.input_video_dir, "*.mp4"))

    for input_video_path in tqdm(input_video_paths, total=len(input_video_paths)):
        basename = os.path.basename(input_video_path)

        video_reader = VideoReader(input_video_path)
        video_length = len(video_reader)
        fps_original = int(video_reader.get_avg_fps())

        clip_length = min(video_length, (args.sample_n_frames - 1) * args.sample_stride + 1)
        batch_index = np.linspace(0, clip_length - 1, args.sample_n_frames, dtype=int)
        np_video = video_reader.get_batch(batch_index).asnumpy()
        del video_reader

        controlnet_frames = [controlnet_model(x) for x in np_video]
        out_controlnet_path = os.path.join(args.out_controlnet_video_dir, basename)
        save_video(out_controlnet_path, controlnet_frames, fps_original)
        break

# python prepare_controlnet_video.py \
# --input_video_dir "path to input video dir" \
# --out_controlnet_video_dir "dir for output controlnet video" \
# --controlnet_type "canny" \
# --sample_stride 2 \
# --width 832 \
# --height 480 \
# --sample_n_frames 81 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate input latents for training.")
    parser.add_argument("--input_video_dir", type=str, required=True, help="Directory with video for processing")
    parser.add_argument("--out_controlnet_video_dir", type=str, required=True, help="Directory for controlnet video")
    parser.add_argument("--controlnet_type", type=str, default="canny", help="canny, hed or depth")
    parser.add_argument("--sample_stride", type=int, default=2, help="get each N frame")
    parser.add_argument("--width", type=int, default=832, help="width for preprocessor")
    parser.add_argument("--height", type=int, default=480, help="height for preprocessor")
    parser.add_argument("--sample_n_frames", type=int, default=81, help="total frames count")
    args = parser.parse_args()
    main(args)