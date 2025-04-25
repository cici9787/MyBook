from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir = '/root/autodl-tmp')
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "auto", torch_dtype = torch.bfloat16)
print(model.eval())