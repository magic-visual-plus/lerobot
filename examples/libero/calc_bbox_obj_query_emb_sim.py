import dataclasses
from gc import disable
from torch._tensor import Tensor
from torch.nn.modules.module import Module
from torch._tensor import Tensor
from torch.nn.modules.module import Module
from typing import Any
from pprint import pprint
import lerobot
from loguru import logger
import numpy as np
import ipdb
import torch
import cv2

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot.policies.smolvla3.modeling_smolvla3 import SmolVLA3Policy
from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
from lerobot.policies.smolvla4.modeling_smolvla4 import SmolVLA4Policy, OutputProjectionMLP
from lerobot.policies.pretrained import PreTrainedPolicy

np.set_printoptions(precision=3, suppress=True)

policy_version_map: dict[str, Any] = {
    "v2" : SmolVLA2Policy,
    "v3" : SmolVLA3Policy,
    "v4" : SmolVLA4Policy,
}

LIBERO_ENV_RESOLUTION = 256

NEED_FIRST_EPISODE = True
DISABLE_STATE = False
DISABLE_IMAGE = True
DISABLE_WRIST_IMAGE = False
DISABLE_LANG = True

@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    # --- Hugging Face arguments ---
    policy_path: str = "/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/only_bbox/pretrained_model_2w"
    # policy_path: str = "/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/disable_action/pretrained_model_2w"

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_goal"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "/opt/projects/xbkaishui/lerobot/data/libero/1020/disable_action2_old_remove_image"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""
    # 预测的版本
    version: str = "v4"
    
def init_policy(args: Args ):
    print(f'args version {args.version}')
    policy_clazz = policy_version_map[args.version]
    policy = policy_clazz.from_pretrained(args.policy_path)
     # Handle accelerate-wrapped models by unwrapping them
    if hasattr(policy, 'module') and isinstance(policy.module, PreTrainedPolicy):
        print("got accelerate model")
        # This is likely an accelerate-wrapped model (DistributedDataParallel)
        policy: PreTrainedPolicy = policy.module
     
    print(f'n_action_steps:{policy.config.n_action_steps}')
    # policy.config.n_action_steps = 8
    print(f'after reset n_action_steps:{policy.config.n_action_steps}')
    policy.to(args.device, dtype=torch.float32)
    policy.eval()
    # ipdb.set_trace()
    return policy

def calc_linear_sim(pretrain_output_linear,  disable_action_output_linear):
    # Get the weight and bias of output_linear layers
    pretrain_weight = pretrain_output_linear.weight.data.cpu().numpy()
    
    if pretrain_output_linear.bias is not None:
        pretrain_bias = pretrain_output_linear.bias.data.cpu().numpy()
    
    disable_action_weight = disable_action_output_linear.weight.data.cpu().numpy()
    if disable_action_output_linear.bias is not None:
        disable_action_bias = disable_action_output_linear.bias.data.cpu().numpy()
    
    # print(f"Pretrain box_mlp output_linear weight shape: {pretrain_weight.shape}")
    # print(f"Disable action box_mlp output_linear weight shape: {disable_action_weight.shape}")
    
    # Flatten weights and biases for similarity calculation
    pretrain_weight_flat = pretrain_weight.flatten()
    disable_action_weight_flat = disable_action_weight.flatten()
    
    # Calculate similarities for weights
    weight_cos_sim = np.dot(pretrain_weight_flat, disable_action_weight_flat) / (
        np.linalg.norm(pretrain_weight_flat) * np.linalg.norm(disable_action_weight_flat)
    )
    weight_euclidean_dist = np.linalg.norm(pretrain_weight_flat - disable_action_weight_flat)
    
    if pretrain_output_linear.bias is not None:
    # Calculate similarities for biases
        bias_cos_sim = np.dot(pretrain_bias, disable_action_bias) / (
            np.linalg.norm(pretrain_bias) * np.linalg.norm(disable_action_bias)
        )
        bias_euclidean_dist = np.linalg.norm(pretrain_bias - disable_action_bias)
        
        # print("\nBias Similarities:")
        print(f"Bias Cosine Similarity: {bias_cos_sim}")
        # print(f"  Euclidean Distance: {bias_euclidean_dist}")
    
    # Print results
    # print("Weight Similarities:")
    print(f"Weight Cosine Similarity: {weight_cos_sim}")
    # print(f"  Euclidean Distance: {weight_euclidean_dist}")                

def prepare_obs() -> None:
    task_description = "push the plate to the front of the stove"
    agentview_image = cv2.imread("/opt/projects/xbkaishui/lerobot/910_main.jpg")
    state = np.array([0.04544, 0.15893,  0.92672 , 3.12924 ,-0.03993 , 0.04949 , 0.00098, -0.00002])
    wrist_img = cv2.imread("/opt/projects/xbkaishui/lerobot/910_wrist.jpg")
    observation = {
    "observation.images.image": torch.from_numpy(agentview_image / 255.0)
    .permute(2, 0, 1).to(torch.float32).to('cuda').unsqueeze(0),
    "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0)
    .permute(2, 0, 1).to(torch.float32).to('cuda').unsqueeze(0),
    "observation.state": torch.from_numpy(state).to(torch.float32).to('cuda').unsqueeze(0),
    "task": task_description,
    }
    return observation, agentview_image
    ...

def infer_image(policy, obs):
    batch = policy._prepare_batch(obs)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    
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
    past_key_values, prefix_embs_after_attn, hidden_states = policy.model.vlm.prepare_for_generation(
        attention_mask=attention_matrix_prefix,
        position_ids=prefix_position_ids,
        inputs_embeds=prefix_embs,
    )
    return prefix_embs, prefix_embs_after_attn, ranges

def calc_embeding_sim(emb1, emb2):
    emb1_np = emb1.detach().cpu().numpy()
    emb2_np = emb2.detach().cpu().numpy()
     # Cosine similarity (normalized dot product)
    cos_sim = np.dot(emb1_np, emb2_np) / (
        np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)
    )
    # print(f"emb cosine similarity: {cos_sim}")
    euclidean_dist = np.linalg.norm(emb1_np - emb2_np)
    # print(f"emb euclidean distance: {euclidean_dist}")
    return cos_sim, euclidean_dist


def calc_obj_query_emb_sim(): 
    model_ckpt_pre_train = '/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/only_bbox/pretrained_model_2w'
    
    # model_disable_action = "/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/disable_action/pretrained_model_2w"
    
    # model_disable_action: str = '/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1023/libero_smolvla4_1023_goal_autodl_disable_bbox_emb_vit_encoder_action/pretrained_model_5k'

    model_disable_action: str = '/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1023/libero_smolvla4_1023_goal_autodl_disable_bbox_emb_vit_encoder_action/pretrained_model_2w'

    # model_disable_action = model_ckpt_pre_train
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre_train_args = Args(policy_path=model_ckpt_pre_train)
    pre_train_policy = init_policy(pre_train_args)
    
    disable_action_args = Args(policy_path=model_disable_action)
    disable_action_policy = init_policy(disable_action_args)
    # disable_action_policy = pre_train_policy
    obj_query_pretrain: Any | Tensor | Module = pre_train_policy.model.box_in_emb
    obj_query_disable_action: Any | Tensor | Module = disable_action_policy.model.box_in_emb
    
    # obj query is shape 1, 10, 969
    print(f'obj_query_pretrain shape {obj_query_pretrain.shape}')
    print(f'obj_query_disable_action shape {obj_query_disable_action.shape}')
    
    first_query_pretrain = obj_query_pretrain[0, 0, :].detach().cpu().numpy()
    first_query_disable_action = obj_query_disable_action[0, 0, :].detach().cpu().numpy()
    # print(f'first_query_pretrain {first_query_pretrain}')
    # print(f'first_query_disable_action {first_query_disable_action}')
    
    # Cosine similarity (normalized dot product)
    cos_sim = np.dot(first_query_pretrain, first_query_disable_action) / (
        np.linalg.norm(first_query_pretrain) * np.linalg.norm(first_query_disable_action)
    )
    print(f"object query cosine similarity: {cos_sim}")
    
    pretrain_box_mlp:OutputProjectionMLP = pre_train_policy.model.box_out_proj
    disable_action_box_mlp:OutputProjectionMLP = disable_action_policy.model.box_out_proj
    
    # print(f'pretrain_box_mlp {pretrain_box_mlp}') # type: ignore
    
    # calc mlp last layer similar
    calc_linear_sim(pretrain_box_mlp.output_linear, disable_action_box_mlp.output_linear)
    # calc input layer similar
    calc_linear_sim(pretrain_box_mlp.input_linear, disable_action_box_mlp.input_linear)
    
    # ipdb.set_trace()
    
    vlm_pretrain = pre_train_policy.model.vlm
    vlm_disable_action = disable_action_policy.model.vlm
    
    print("text encoder last layer mlp gate")
    pretrain_text_model_ffn_first_layer = vlm_pretrain.vlm.text_model.layers[-1].mlp.gate_proj
    disable_action_text_model_ffn_first_layer = vlm_disable_action.vlm.text_model.layers[-1].mlp.gate_proj
    calc_linear_sim(pretrain_text_model_ffn_first_layer, disable_action_text_model_ffn_first_layer)
    
    print("text encoder last layer mlp down_proj")
    pretrain_text_model_ffn_last_layer = vlm_pretrain.vlm.text_model.layers[-1].mlp.down_proj
    disable_action_text_model_ffn_last_layer = vlm_disable_action.vlm.text_model.layers[-1].mlp.down_proj
    calc_linear_sim(pretrain_text_model_ffn_last_layer, disable_action_text_model_ffn_last_layer)
    # down_proj
    print("text encoder last layer attn out_proj")
    pretrain_text_model_attn_o_last_layer = vlm_pretrain.vlm.text_model.layers[-1].self_attn.o_proj
    disable_action_text_model_attn_o_last_layer = vlm_disable_action.vlm.text_model.layers[-1].self_attn.o_proj
    calc_linear_sim(pretrain_text_model_attn_o_last_layer, disable_action_text_model_attn_o_last_layer)
    
    print("text encoder first layer attn out_proj")
    pretrain_text_model_attn_o_first_layer = vlm_pretrain.vlm.text_model.layers[0].self_attn.o_proj
    disable_action_text_model_attn_o_first_layer = vlm_disable_action.vlm.text_model.layers[0].self_attn.o_proj
    calc_linear_sim(pretrain_text_model_attn_o_first_layer, disable_action_text_model_attn_o_first_layer)
    
    print("text encoder last layer attn q_proj")
    pretrain_text_model_attn_q_last_layer = vlm_pretrain.vlm.text_model.layers[-1].self_attn.q_proj
    disable_action_text_model_attn_q_last_layer = vlm_disable_action.vlm.text_model.layers[-1].self_attn.q_proj
    calc_linear_sim(pretrain_text_model_attn_q_last_layer, disable_action_text_model_attn_q_last_layer)
    
    print("text encoder last layer attn k_proj")
    pretrain_text_model_attn_k_last_layer = vlm_pretrain.vlm.text_model.layers[-1].self_attn.k_proj
    disable_action_text_model_attn_k_last_layer = vlm_disable_action.vlm.text_model.layers[-1].self_attn.k_proj
    calc_linear_sim(pretrain_text_model_attn_k_last_layer, disable_action_text_model_attn_k_last_layer)

    print("text encoder last layer attn v_proj")
    pretrain_text_model_attn_v_last_layer = vlm_pretrain.vlm.text_model.layers[-1].self_attn.v_proj
    disable_action_text_model_attn_v_last_layer = vlm_disable_action.vlm.text_model.layers[-1].self_attn.v_proj
    calc_linear_sim(pretrain_text_model_attn_v_last_layer, disable_action_text_model_attn_v_last_layer)

    # ipdb.set_trace()
    
    print("image encoder last layer mlp fc1")
    prtrain_image_encoder_ffn_first_layer = vlm_pretrain.vlm.vision_model.encoder.layers[-1].mlp.fc1
    disable_action_image_encoder_ffn_first_layer = vlm_disable_action.vlm.vision_model.encoder.layers[-1].mlp.fc1
    calc_linear_sim(prtrain_image_encoder_ffn_first_layer, disable_action_image_encoder_ffn_first_layer)
    
    print("image encoder last layer mlp fc2")
    prtrain_image_encoder_ffn_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[-1].mlp.fc2
    disable_action_image_encoder_ffn_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[-1].mlp.fc2
    calc_linear_sim(prtrain_image_encoder_ffn_last_layer, 
disable_action_image_encoder_ffn_last_layer)
    
    print("image encoder attention last layer ffn")
    prtrain_image_encoder_attn_ffn_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[-1].self_attn.out_proj
    disable_action_image_encoder_attn_ffn_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[-1].self_attn.out_proj
    calc_linear_sim(prtrain_image_encoder_attn_ffn_last_layer, disable_action_image_encoder_attn_ffn_last_layer)
    
    print("image encoder attention last layer k_proj")
    prtrain_image_encoder_attn_k_proj_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[-1].self_attn.k_proj
    disable_action_image_encoder_attn_k_proj_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[-1].self_attn.k_proj
    calc_linear_sim(prtrain_image_encoder_attn_k_proj_last_layer, 
disable_action_image_encoder_attn_k_proj_last_layer)
    
    print("image encoder attention last layer q_proj")
    prtrain_image_encoder_attn_q_proj_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[-1].self_attn.q_proj
    disable_action_image_encoder_attn_q_proj_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[-1].self_attn.q_proj
    calc_linear_sim(prtrain_image_encoder_attn_q_proj_last_layer, 
disable_action_image_encoder_attn_q_proj_last_layer)
    
    print("image encoder attention last layer v_proj")
    prtrain_image_encoder_attn_v_proj_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[-1].self_attn.v_proj
    disable_action_image_encoder_attn_v_proj_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[-1].self_attn.v_proj
    calc_linear_sim(prtrain_image_encoder_attn_v_proj_last_layer, 
disable_action_image_encoder_attn_v_proj_last_layer)
    

    print("image encoder attention first layer ffn")
    prtrain_image_encoder_attn_ffn_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[0].self_attn.out_proj
    disable_action_image_encoder_attn_ffn_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[0].self_attn.out_proj
    calc_linear_sim(prtrain_image_encoder_attn_ffn_last_layer, 
disable_action_image_encoder_attn_ffn_last_layer)
    
    print("image encoder attention first layer k_proj")
    prtrain_image_encoder_attn_k_proj_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[0].self_attn.k_proj
    disable_action_image_encoder_attn_k_proj_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[0].self_attn.k_proj
    calc_linear_sim(prtrain_image_encoder_attn_k_proj_last_layer, 
disable_action_image_encoder_attn_k_proj_last_layer)
    
    print("image encoder attention first layer q_proj")
    prtrain_image_encoder_attn_q_proj_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[0].self_attn.q_proj
    disable_action_image_encoder_attn_q_proj_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[0].self_attn.q_proj
    calc_linear_sim(prtrain_image_encoder_attn_q_proj_last_layer, 
disable_action_image_encoder_attn_q_proj_last_layer)
    
    print("image encoder attention first layer v_proj")
    prtrain_image_encoder_attn_v_proj_last_layer = vlm_pretrain.vlm.vision_model.encoder.layers[0].self_attn.v_proj
    disable_action_image_encoder_attn_v_proj_last_layer = vlm_disable_action.vlm.vision_model.encoder.layers[0].self_attn.v_proj
    calc_linear_sim(prtrain_image_encoder_attn_v_proj_last_layer, 
disable_action_image_encoder_attn_v_proj_last_layer)
    
    print("image encoder attention first layer ffn")
    prtrain_image_encoder_attn_ffn_first_layer = vlm_pretrain.vlm.vision_model.encoder.layers[0].self_attn.out_proj
    disable_action_image_encoder_attn_ffn_first_layer = vlm_disable_action.vlm.vision_model.encoder.layers[0].self_attn.out_proj
    calc_linear_sim(prtrain_image_encoder_attn_ffn_first_layer, disable_action_image_encoder_attn_ffn_first_layer)
    
    # calc vision encoder connector mlp 
    print(f'vision encoder connector mlp sim')
    pt_vlm_connector_mlp = vlm_pretrain.vlm.connector.modality_projection.proj
    da_vlm_connector_mlp = vlm_disable_action.vlm.connector.modality_projection.proj
    calc_linear_sim(pt_vlm_connector_mlp, da_vlm_connector_mlp)
    # calc text embeding sim
    # infer images
    
    observation, agentview_image = prepare_obs()
    
    agentview_image_bbox =  agentview_image.copy()         
    pre_train_policy.eval()
    disable_action_policy.eval()
    action_result_tensor = disable_action_policy.select_action(observation, need_bbox = True)
    # action_result_tensor = disable_action_policy.select_action(observation, need_bbox = True)
    # plot box
    bbox = action_result_tensor['box']
    
    box0 = bbox[0, 0, :].cpu().numpy()
    box0 = (box0 * LIBERO_ENV_RESOLUTION).astype(int)
    box0 = box0.tolist()
    cv2.rectangle(
     agentview_image_bbox,
     (box0[0], box0[1]),
     (box0[0]+box0[2], box0[1]+box0[3]),
     (0, 255, 0),
     2,
    )
    cv2.imwrite("/opt/projects/xbkaishui/lerobot/910_main_bbox.jpg", agentview_image_bbox)
    
    # calc image embeding sim
    # ipdb.set_trace()
    # test lang embdeing 
    pre_train_prefix_embs, pre_train_prefix_embs_after_attn, pre_train_ranges = infer_image(pre_train_policy, observation)
    
    disable_action_prefix_embs, disable_action_prefix_embs_after_attn, disable_action_ranges = infer_image(disable_action_policy, observation)
    
    pt_img_range, pt_lang_range, pt_state_range, pt_box_range, pt_depth_range, pt_point_range = pre_train_ranges
    da_img_range, da_lang_range, da_state_range, da_box_range, da_depth_range, da_point_range = disable_action_ranges
    
    # ipdb.set_trace()
    pt_img_emb = pre_train_prefix_embs[:, pt_img_range[0]:pt_img_range[1] // 2, :]
    pt_lang_emb = pre_train_prefix_embs[:, pt_lang_range[0]:pt_lang_range[1], :]
    pt_pre_box_emb = pre_train_prefix_embs[:, pt_box_range[0]:pt_box_range[1], :]  # (B, num_box, D)
    
    pt_after_attn_lang_emb = pre_train_prefix_embs_after_attn[:, pt_lang_range[0]:pt_lang_range[1], :]
    pt_after_attn_img_emb = pre_train_prefix_embs_after_attn[:, pt_img_range[0]:pt_img_range[1] // 2, :]
    pt_after_attn_box_emb = pre_train_prefix_embs_after_attn[:, pt_box_range[0]:pt_box_range[1], :]
    
    da_img_emb = disable_action_prefix_embs[:, da_img_range[0]:da_img_range[1] // 2, :]
    da_lang_emb = disable_action_prefix_embs[:, da_lang_range[0]:da_lang_range[1], :]
    da_pre_box_emb = disable_action_prefix_embs[:, da_box_range[0]:da_box_range[1], :]
    
    da_after_attn_lang_emb = disable_action_prefix_embs_after_attn[:, da_lang_range[0]:da_lang_range[1], :]
    da_after_attn_img_emb = disable_action_prefix_embs_after_attn[:, da_img_range[0]:da_img_range[1] // 2, :]
    da_after_attn_box_emb = disable_action_prefix_embs_after_attn[:, da_box_range[0]:da_box_range[1], :]
    
    # calc emb sim
    print("===================before attn =============================")
    print("calc lang embeding sim")
    plt_lang_sims = []
    for idx in range(pt_lang_emb.shape[1]):
        cos_sim, _ = calc_embeding_sim(pt_lang_emb[0, idx, :], da_lang_emb[0, idx, :])
        plt_lang_sims.append(cos_sim)
    # print max min avg 
    print(f"lang embeding sim max {max(plt_lang_sims)}, min {min(plt_lang_sims)}, avg {sum(plt_lang_sims) / len(plt_lang_sims)}")
    
    plt_img_sims = []
    print("calc image embeding sim")
    for idx in range(pt_img_emb.shape[1]):
        cos_sim, _ = calc_embeding_sim(pt_img_emb[0, idx, :], da_img_emb[0, idx, :])
        plt_img_sims.append(cos_sim)
    print(f"image embeding sim max {max(plt_img_sims)}, min {min(plt_img_sims)}, avg {sum(plt_img_sims) / len(plt_img_sims)}")
    
    plt_box_sims = []
    print("calc box embeding sim")
    for idx in range(pt_pre_box_emb.shape[1]):
        cos_sim, _ = calc_embeding_sim(pt_pre_box_emb[0, idx, :], da_pre_box_emb[0, idx, :])
        plt_box_sims.append(cos_sim)
    print(f"box embeding sim max {max(plt_box_sims)}, min {min(plt_box_sims)}, avg {sum(plt_box_sims) / len(plt_box_sims)}")
        
    print("===================after attn =============================")
    print("calc lang after embeding sim")
    plt_lang_sims = []
    for idx in range(pt_after_attn_lang_emb.shape[1]):
        cos_sim, _ = calc_embeding_sim(pt_after_attn_lang_emb[0, idx, :], da_after_attn_lang_emb[0, idx, :])
        plt_lang_sims.append(cos_sim)
    print(f"after attn lang embeding sim max {max(plt_lang_sims)}, min {min(plt_lang_sims)}, avg {sum(plt_lang_sims) / len(plt_lang_sims)}")

    print("calc img after embeding sim")
    plt_img_sims = []
    for idx in range(pt_after_attn_img_emb.shape[1]):
        cos_sim, _ = calc_embeding_sim(pt_after_attn_img_emb[0, idx, :], da_after_attn_img_emb[0, idx, :])
        plt_img_sims.append(cos_sim)
    print(f"after attn img embeding sim max {max(plt_img_sims)}, min {min(plt_img_sims)}, avg {sum(plt_img_sims) / len(plt_img_sims)}")  
    
    plt_box_sims = []
    print("calc box after embeding sim")
    for idx in range(pt_after_attn_box_emb.shape[1]):
        cos_sim, _ = calc_embeding_sim(pt_after_attn_box_emb[0, idx, :], da_after_attn_box_emb[0, idx, :])
        plt_box_sims.append(cos_sim)
    print(f"after attn box embeding sim max {max(plt_box_sims)}, min {min(plt_box_sims)}, avg {sum(plt_box_sims) / len(plt_box_sims)}")

    
    pt_box_pred = pre_train_policy.model.box_out_proj(pt_after_attn_box_emb)
    da_box_pred = disable_action_policy.model.box_out_proj(da_after_attn_box_emb)
    print(f"pt box pre {pt_box_pred[0][0] * LIBERO_ENV_RESOLUTION}")
    print(f"da box pre {da_box_pred[0][0] * LIBERO_ENV_RESOLUTION}")
    
    

if __name__ == "__main__":
    calc_obj_query_emb_sim()