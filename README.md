# Dilated Controlnet for Wan2.1

https://github.com/user-attachments/assets/97c0ece2-da42-4425-a0b6-a84929aa4d6e

This repo contains the code for dilated controlnet module for Wan2.1 model.  
Dilated controlnet has less basic blocks and also has `stride` parameter. For Wan1.3B model controlnet blocks count = 8 and stride = 3. 
<p>
    <img src="./resources/scheme.png" width="832" height="420" title="dilated_scheme"/>
</p>

### Models  
Supported models for 1.3B:
- Hed (<a href="https://huggingface.co/TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1">HF Model Link</a>)  

Supported models for 14B:
- Hed (<a href="https://huggingface.co/TheDenk/wan2.1-t2v-14b-controlnet-hed-v1">HF Model Link</a>)  


### How to
Clone repo 
```bash
git clone https://github.com/TheDenk/wan2.1-dilated-controlnet.git
cd wan2.1-dilated-controlnet
```
  
Create venv  
```bash
python -m venv venv
source venv/bin/activate
```
  
Install requirements
```bash
pip install -r requirements.txt
```

### Inference examples
#### Inference with cli
```bash
python -m inference.cli_demo \
    --video_path "resources/physical-4.mp4" \
    --prompt "A balloon filled with water was thrown to the ground, exploding and splashing water in all directions. There were graffiti on the wall, studio lighting, and commercial movie shooting." \
    --controlnet_type "hed" \
    --controlnet_stride 3 \
    --base_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --controlnet_model_path TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1
```

#### Inference with Gradio
```bash
python -m inference.gradio_web_demo \
    --controlnet_type "hed" \
    --base_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --controlnet_model_path TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1
```
#### Detailed Inference
```bash
python -m inference.cli_demo \
    --video_path "resources/physical-4.mp4" \
    --prompt "A balloon filled with water was thrown to the ground, exploding and splashing water in all directions. There were graffiti on the wall, studio lighting, and commercial movie shooting." \
    --controlnet_type "hed" \
    --base_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --controlnet_model_path TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1 \
    --controlnet_weight 0.8 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.8 \
    --controlnet_stride 3 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --video_height 480 \
    --video_width 832 \
    --num_frames 81 \
    --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --out_fps 16 \
    --output_path "result.mp4"
```


## Training
Wan 1.3B model requires `18 GB VRAM` with `batch_size=1`. But it also depends on the number of transformer blocks which default is 8 (`controlnet_transformer_num_layers` parameter in the config).  

#### Dataset
<a href="https://huggingface.co/datasets/nkp37/OpenVid-1M">OpenVid-1M</a> dataset was taken as the base variant. CSV files for the dataset you can find <a href="https://huggingface.co/datasets/nkp37/OpenVid-1M/tree/main/data/train">here</a>.

#### Prepare dataset
Download dataset and prepare data. We do not use raw data to save memory.   
Extract text embeddings. Initially all text are located in .csv file.    
```bash
CUDA_VISIBLE_DEVICES=0 python prepare_text_embeddings.py \
--csv_path "path to csv" \
--out_embeds_dir "path to output dir" \
--base_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
--device "cuda" \
--dtype "bf16"
```
Encode video into vae latents.  
```bash
CUDA_VISIBLE_DEVICES=0 python prepare_vae_latents.py \
--input_video_dir "path to input video dir" \
--out_latents_dir "dir for output latents" \
--base_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
--sample_stride 2 \
--width 832 \
--height 480 \
--sample_n_frames 81 \
--seed 42 \
--device "cuda" \
--dtype "fp32"
```
Preprocess original video with controlnet processor.  
```bash
python prepare_controlnet_video.py \
--input_video_dir "path to input video dir" \
--out_controlnet_video_dir "dir for output controlnet video" \
--controlnet_type "canny" \
--sample_stride 2 \
--width 832 \
--height 480 \
--sample_n_frames 81 
```

#### Train script
For start training you need fill the config files `accelerate_config_machine_single.yaml` and `train_controlnet.sh`.  
In `accelerate_config_machine_single.yaml` set parameter`num_processes: 1` to your GPU count.  
In `train_controlnet.sh`:  
1. Set `MODEL_PATH for` base CogVideoX model. Default is THUDM/CogVideoX-2b.  
2. Set `CUDA_VISIBLE_DEVICES` (Default is 0).  
3. Set `output_dir`, `latents_dir`, `text_embeds_dir` and `controlnet_video_dir` parameters.  

Run taining
```
cd train
bash train_controlnet.sh
```

## Acknowledgements
Original code and models [Wan2.1](https://github.com/Wan-Video/Wan2.1).  


## Citations
```
@misc{TheDenk,
    title={Dilated Controlnet},
    author={Karachev Denis},
    url={https://github.com/TheDenk/wan2.1-dilated-controlnet},
    publisher={Github},
    year={2025}
}
```

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
