#!/bin/bash

export MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# bash train_controlnet.sh
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_controlnet.py \
  --tracker_name "wan2.1-controlnet" \
  --pretrained_model_name_or_path $MODEL_PATH \
  --weighting_scheme 'none' \
  --seed 17 \
  --mixed_precision bf16 \
  --output_dir "path for output checkpoints directory" \
  --latents_dir "path to directory with latents tensors" \
  --text_embeds_dir "path to directory with text emneddings tensors" \
  --controlnet_video_dir "path to directory with preprocessed controlnet video" \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 1.0 \
  --controlnet_stride 3 \
  --save_checkpoint_postfix "_3stride_8blocks" \
  --init_from_transformer \
  --train_batch_size 4 \
  --dataloader_num_workers 0 \
  --num_train_epochs 1 \
  --checkpointing_steps 64 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 64 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --gradient_checkpointing \
  --report_to wandb 
  # --pretrained_controlnet_path "TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1" 