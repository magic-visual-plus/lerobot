from modelscope import snapshot_download
import os 
# get current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
# model_name = 'llava-hf/llava-1.5-7b-hf'
model_name = 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
model_dir = snapshot_download(model_name, cache_dir=f'{cur_dir}/ckpts', revision='master')