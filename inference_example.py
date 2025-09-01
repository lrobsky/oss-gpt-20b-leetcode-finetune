"""
Inference example for oss-gpt-20b ,fine-tuned on LeetCode-style questions.
"""

import torch

from unsloth import FastLanguageModel
from transformers import TextStreamer,AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel



#  Detect device and VRAM 
has_cuda = torch.cuda.is_available()
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0
print(f"CUDA available: {has_cuda}, GPU VRAM: {total_vram:.1f} GB")

# Decide 4-bit qantization 
if has_cuda:
    device = "cuda"
    if total_vram >= 24:
        load_in_4bit = False  # enough vram for full model
    else:
        load_in_4bit = True   # use quantization
else:
    load_in_4bit = False
    device = "cpu"

print(f"Loading model with 4-bit: {load_in_4bit}, device: {device}")



base_model = "unsloth/gpt-oss-20b"
adapter_model = "lrobsky/gpt-oss-20b-finetuned-leetcode"
prompt = "Explain the Two Sum problem in Python."


# First option - unsloth FastLanguageModel  

# Load base model + adapter weights
model, tokenizer = FastLanguageModel.from_pretrained(
    base_model,
    max_seq_length=512,
    load_in_4bit=load_in_4bit,
    device_map="auto")

model = FastLanguageModel.get_peft_model(model, adapter_model)
# Enable inference
FastLanguageModel.for_inference(model) 



# Example prompt
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

# Stream generation
streamer = TextStreamer(tokenizer, skip_prompt=True)
outputs = model.generate(input_ids, streamer=streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
print("Model output : ")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



# Second option -  Hugging Face + PEFT 

# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     device_map="auto" if has_cuda else None,
#     torch_dtype=torch.float16 if has_cuda else torch.float32
# )
# model = PeftModel.from_pretrained(model, adapter_model)
# tokenizer = AutoTokenizer.from_pretrained(base_model)

# inputs = tokenizer(prompt, return_tensors="pt").to(device) #move tensor to proper same device
# outputs = model.generate(**inputs, max_new_tokens=128)
# print("Model output : ")
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

