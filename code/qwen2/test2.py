from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download("/root/autodl-tmp/Qwen2-1.5B-Instruct", cache_dir = './')