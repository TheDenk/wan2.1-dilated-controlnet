import os
import re
import argparse

import html
import ftfy
import torch
import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text
    
@torch.no_grad
def extract_text_embeddings(tokenizer, text_encoder, prompt, max_sequence_length=512, device=torch.device("cuda"), dtype=torch.bfloat16):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )
    return prompt_embeds


def main(args):
    torch_dtype = torch.bfloat16
    if args.dtype == "fp32":
        torch_dtype = torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.bfloat16
    
    device = torch.device("cuda") if args.device == "cuda" else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder", torch_dtype=torch_dtype
    ).to(device=device)
    print(f"MODEL HAS BEEN LOADED TO {device} WITH DTYPE: {torch_dtype}")

    os.makedirs(args.out_embeds_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt_embeds = extract_text_embeddings(tokenizer, text_encoder, row["caption"]).cpu()
        out_embeds_path = os.path.join(args.out_embeds_dir, row["video"].replace(".mp4", ".pt"))
        torch.save(prompt_embeds, out_embeds_path)

# CUDA_VISIBLE_DEVICES=0 python prepare_text_embeddints.py \
# --csv_path "path to csv" \
# --out_embeds_dir "path to output dir" \
# --base_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
# --device "cuda" \
# --dtype "bf16"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a text embeddings for training.")
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The path of the pre-trained model with text encoder"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to csv file from OpenVid dataset or other with text in column 'caption'.")
    parser.add_argument("--out_embeds_dir", type=str, required=True, help="Directory for embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="bf16", help="fp32, fp16 or bf16")
    args = parser.parse_args()
    main(args)