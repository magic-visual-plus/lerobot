from modelscope import snapshot_download
import os 
# get current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
# model_name = 'llava-hf/llava-1.5-7b-hf'
model_name = 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
model_name = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
ckpt_path = f'{cur_dir}/ckpts'
ckpt_path = '/root/autodl-fs/weights'
model_dir = snapshot_download(model_name, cache_dir=ckpt_path, revision='master')