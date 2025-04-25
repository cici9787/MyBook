from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir = '/root/autodl-tmp')