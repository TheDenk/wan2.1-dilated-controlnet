import os
import glob
import random

import torch
import numpy as np
from decord import VideoReader
from torch.utils.data.dataset import Dataset


class ControlnetDataset(Dataset):
    def __init__(
            self, 
            latents_dir,
            text_embeds_dir,
            controlnet_video_dir,
        ):
        self.latents_dir = latents_dir
        self.text_embeds_dir = text_embeds_dir
        self.controlnet_video_dir = controlnet_video_dir
        videos_paths = glob.glob(os.path.join(self.controlnet_video_dir, '*.mp4'))
        self.videos_names = [os.path.basename(x) for x in videos_paths]
        self.length = len(self.videos_names)
        
    def __len__(self):
        return self.length
    
    def get_batch(self, idx):
        video_name = self.videos_names[idx]

        text_embeds_path = os.path.join(self.text_embeds_dir, video_name.replace(".mp4", ".pt"))
        text_embeds = torch.load(text_embeds_path, map_location="cpu", weights_only=True)

        latents_path = os.path.join(self.latents_dir, video_name.replace(".mp4", ".pt"))
        latents = torch.load(latents_path, map_location="cpu", weights_only=True)
        
        video_path = os.path.join(self.controlnet_video_dir, video_name)
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        batch_index = np.arange(video_length)
        controlnet_video = video_reader.get_batch(batch_index).asnumpy()
        controlnet_video = torch.from_numpy(controlnet_video).permute(3, 0, 1, 2).contiguous()
        img_h, img_w = latents.shape[-2:]
        controlnet_video = torch.nn.functional.interpolate(controlnet_video, (img_h * 8, img_w * 8))
        controlnet_video = controlnet_video / 127.5 - 1
        del video_reader
        return latents, text_embeds, controlnet_video
        
    def __getitem__(self, idx):
        while True:
            try:
                latents, text_embeds, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)
        data = {
            'latents': latents[0],  ## [C, F, H, W] torch.Size([16, 21, 60, 104])
            'text_embeds': text_embeds[0], 
            'controlnet_video': controlnet_video, ## [C, F, H, W] torch.Size([3, 81, 480, 832])
        }
        return data
    