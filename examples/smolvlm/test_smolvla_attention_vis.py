import dataclasses
from lerobot.policies.pretrained import PreTrainedPolicy
from typing import Any
from pprint import pprint
import lerobot
from loguru import logger
import numpy as np
import cv2
import draccus
import imageio
import sys
import torch
import pathlib
import os
import ipdb
import math
import time
from PIL import Image
import matplotlib.pyplot as plt

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot.policies.smolvla3.modeling_smolvla3 import SmolVLA3Policy
from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
from lerobot.policies.smolvla4.modeling_smolvla4 import SmolVLA4Policy
from lerobot.policies.pretrained import PreTrainedPolicy

np.set_printoptions(precision=3, suppress=True)

NEED_FIRST_FRAME = True
LIBERO_ENV_RESOLUTION = 256


policy_version_map: dict[str, Any] = {
    "v2" : SmolVLA2Policy,
    "v3" : SmolVLA3Policy,
    "v4" : SmolVLA4Policy,
}

@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    # --- Hugging Face arguments ---
    # policy_path: str = "/autodl-fs/data/ckpts/1020/libero_smolvla4_1020_new_goal_autodl_only_bbox/checkpoints/030000/pretrained_model"
    policy_path: str = "/root/autodl-fs/ckpts/1030/libero_smolvla4_1030_goal_autodl_bbox_pretrain/checkpoints/020000/pretrained_model"
    policy_path: str = "/root/autodl-fs/ckpts/1030/libero_smolvla4_1030_goal_single_task_5_ds_mix_action_no_pre/checkpoints/030000/pretrained_model"

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_goal"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "/opt/product/lerobot/datas"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""
    # 预测的版本
    version: str = "v4"
    
def init_policy(args: Args ):
    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    print(f'args version {args.version}')
    policy_clazz = policy_version_map[args.version]
    policy = policy_clazz.from_pretrained(args.policy_path)
     # Handle accelerate-wrapped models by unwrapping them
    if hasattr(policy, 'module') and isinstance(policy.module, PreTrainedPolicy):
        print("got accelerate model")
        # This is likely an accelerate-wrapped model (DistributedDataParallel)
        policy: PreTrainedPolicy = policy.module
     
    print(f'n_action_steps:{policy.config.n_action_steps}')
    policy.config.n_action_steps = 8
    print(f'after reset n_action_steps:{policy.config.n_action_steps}')
    # change attn implementation to eager
    policy.model.vlm.vlm.text_model.config._attn_implementation = 'eager'
    policy.to(args.device)
    policy.eval()
    return policy


def prepare_obs_from_dataset(dataset_path = '/autodl-fs/data/datasets/libero_goal_no_lerobot_0', episode_idx = 0) -> None:
    dataset = LeRobotDataset(dataset_path)
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")
    
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    print(f"episode idx {episode_idx}, task {tasks}, from idx {from_idx} to idx {to_idx}")
    to_pil = transforms.ToPILImage()

    for frame_index in np.arange(from_idx, to_idx).tolist():
        # get first eposido
        frame = dataset[frame_index]
        state = frame['observation.state']
        task_description = tasks
        
        agentview_image = to_pil(frame['observation.images.image'])
        # compose obs and send to policy
        observation = {
          "observation.images.image": frame['observation.images.image']
          .to(args.device).unsqueeze(0),
          "observation.images.wrist_image": frame['observation.images.wrist_image']
          .to(args.device).unsqueeze(0),
          "observation.state": state.to(args.device).unsqueeze(0),
          "task": task_description,
        }
        return observation, agentview_image


def prepare_fix_obs() -> None:
    task_description = "push the plate to the front of the stove"
    # task_description = "push the bowl to the front of the stove"
    # task_description = "push the stove to the front of the stove"
    # task_description = "describe the image"
    task_description = "bottle the"
    cur_dir = pathlib.Path(__file__).parent.resolve()
    print(f'cur_dir is {cur_dir}')
    agentview_image = cv2.imread(f"{cur_dir}/910_main.jpg")
    # agentview_image = cv2.imread(f"{cur_dir}/4_start.png")
    state = np.array([0.04544, 0.15893,  0.92672 , 3.12924 ,-0.03993 , 0.04949 , 0.00098, -0.00002])
    wrist_img = cv2.imread(f"{cur_dir}/910_wrist.jpg")
    observation = {
        "observation.images.image": torch.from_numpy(agentview_image / 255.0)
        .permute(2, 0, 1).to(torch.float32).to('cuda').unsqueeze(0),
        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0)
        .permute(2, 0, 1).to(torch.float32).to('cuda').unsqueeze(0),
        "observation.state": torch.from_numpy(state).to(torch.float32).to('cuda').unsqueeze(0),
        "task": task_description,
    }
    return observation, agentview_image


def infer_vlm_model(policy, obs):
    """"
    run single vlm forward for vlm inference, need to check attention
    """
    batch = policy._prepare_batch(obs)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    
    print(f'len lang {lang_tokens.shape}')
    
    prefix_embs, prefix_pad_masks, prefix_position_ids, ranges = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    img_range, lang_range, state_range, box_range, depth_range, point_range = ranges
    bsize = state.shape[0]
    
    _, suffix_pad_masks, _ = policy.model.embed_suffix_autoregressive(bsize)
    img_len = len(images)
    attention_matrix_prefix, attention_matrix_cross, attention_matrix_suffix = policy.model.generate_attention_matrix(
        prefix_pad_masks, suffix_pad_masks, img_len, img_range, lang_range, state_range, box_range, depth_range, point_range
    )
    past_key_values, prefix_embs_after_attn, _, result = policy.model.vlm.prepare_for_generation(
        attention_mask=attention_matrix_prefix,
        position_ids=prefix_position_ids,
        inputs_embeds=prefix_embs,
        output_hidden_states=True,
        output_attentions=True,
    )
    return prefix_embs, prefix_embs_after_attn, ranges, result

def calculate_plt_size(attention_layer_num):
    """
    calculate_plt_size 的作用是计算绘图网格的行列数
    Args:
        attention_layer_num (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_layers = attention_layer_num
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    return rows, cols

@draccus.wrap()
def replay_and_eval_bbox(args: Args):
    task_id = 5
    cur_dir = pathlib.Path(__file__).parent.resolve()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # --- Load Policy ---
    policy: PreTrainedPolicy = init_policy(args)
    
    observation, agentview_image = prepare_fix_obs()

    prefix_embs, prefix_embs_after_attn, ranges, infer_result = infer_vlm_model(policy, observation)
    
    img_range, lang_range, state_range, box_range, depth_range, point_range = ranges
    
    # extract token input embeding
    text_token_start_index = lang_range[0]
    input_embedings = prefix_embs[0][text_token_start_index]
    first_token_embeding = prefix_embs[0][text_token_start_index]
    first_token_embeding = first_token_embeding.to(dtype=torch.float32)
    np_first_token = first_token_embeding.detach().cpu().numpy()
    lang_emb_dim = 576
    np_first_token = np_first_token / math.sqrt(lang_emb_dim)
    np.save(f"{cur_dir}/smovla_first_token_embeding.npy", np_first_token)
    
    # 先看下 lang 和 img 的 attention score
    for i, hidden_state in enumerate(infer_result.hidden_states):
        print(f"Layer {i} seq_len: {hidden_state.shape[1]}")
    atten_layer_size = len(infer_result.attentions)
    print(f"Attention layer size: {atten_layer_size}")
    rows, cols = calculate_plt_size(atten_layer_size)
    output_shape = [8, 8]
    print(f"attention shape {infer_result.attentions[0].shape}")
    
    # plate word
    query_index = lang_range[0] + 2
    # the word
    query_index = lang_range[0] + 1
   
    # stove word
    # query_index = lang_range[1] - 1
    # query_index = -1
    
    # query_index = lang_range[0]
    img_start_index = img_range[0] + 2
    img_end_index = img_range[1] // 2  - 1 
    
    # box query 
    # query_index = box_range[0]
    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
    # print(f"axes: {axes.flatten()}")
    for i, ax in enumerate(axes.flatten()):
        if i < atten_layer_size:
         #  output.attentions[i][0, :, -1, pos:pos_end] shape: (num_heads, seq_len)
            att = infer_result.attentions[i][0, :, query_index, img_start_index:img_end_index].mean(dim=0)
            # att shape: (seq_len)
            att = att.to(torch.float32).detach().cpu().numpy()
            ax.imshow(
                att.reshape(output_shape), cmap="plasma", interpolation="nearest"
            )
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    timestamp = int(time.time() * 1000)
    file_name = f"smovla4_attention_map_{timestamp}.png"
    print(f'save file name {file_name}')
    plt.savefig(file_name)
    

if __name__ == "__main__":
    replay_and_eval_bbox()