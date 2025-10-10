#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
import os
import re
from collections import deque

import safetensors
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from transformers import AutoProcessor

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import (
    Normalize,
    Unnormalize,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla3.configuration_smolvla3 import SmolVLA3Config
from lerobot.policies.smolvla3.smolvlm import SmolVLM
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.utils import get_safe_dtype

import loguru

logger = loguru.logger

# Matches ".soNNN", optionally followed by "-something", up to the "_buffer_" marker
_VARIANT_RE = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")


def canonicalise(k: str) -> str:
    """
    Remove dataset-variant markers like '.so100-blue_' or '.so100_' from a
    normalisation-buffer key.
    """
    return _VARIANT_RE.sub(".buffer_", k)


def standardise_state_dict(
    checkpoint: dict[str, torch.Tensor], ref_keys: set[str], *, verbose: bool = True
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    • Re-keys `checkpoint ` so that every entry matches the *reference* key set.
    • If several variant keys collapse to the same canonical name we keep the
      first one and log the collision.
    • Returns the new dict + a list of entries that could not be matched.
    """
    out, collisions, unmatched = {}, {}, []

    for k, v in checkpoint.items():
        canon = canonicalise(k)
        if canon in ref_keys:
            if canon in out:  # duplicate after collapsing
                collisions.setdefault(canon, []).append(k)
            else:
                out[canon] = v
        else:
            unmatched.append(k)

    if verbose:
        for canon, variants in collisions.items():
            print(f"[standardise_state_dict] '{canon}'  ←  {variants}")
        if unmatched:
            print(f"[standardise_state_dict] kept {len(unmatched)} unmatched keys")

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched


def rename_checkpoint_keys(checkpoint: dict, rename_str: str):
    """
    Renames keys in a checkpoint dictionary based on the given rename string.

    Args:
        checkpoint (dict): The checkpoint dictionary.
        rename_str (str): A string specifying key mappings in the format "old1//new1,old2//new2".

    Returns:
        dict: The modified checkpoint with renamed keys.
    """

    rename_dict = dict(pair.split("//") for pair in rename_str.split(","))

    new_checkpoint = {}
    for k, v in checkpoint.items():
        for old_key, new_key in rename_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
        new_checkpoint[k] = v
    return new_checkpoint


def load_smolvla(
    model: torch.nn.Module,
    filename: str | os.PathLike,
    *,
    device: str = "cpu",
    checkpoint_keys_mapping: str = "",
) -> torch.nn.Module:
    state_dict = safetensors.torch.load_file(filename, device=device)

    # Optional user-supplied renames (e.g. "model._orig_mod.//model.")
    if checkpoint_keys_mapping and "//" in checkpoint_keys_mapping:
        state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)

    state_dict, _ = standardise_state_dict(state_dict, set(model.state_dict().keys()))

    # HACK(aliberts): to not overwrite normalization parameters as they should come from the dataset
    norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if not all(key.startswith(norm_keys) for key in missing) or unexpected:
        raise RuntimeError(
            "SmolVLA %d missing / %d unexpected keys",
            len(missing),
            len(unexpected),
        )

    return model


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLA3Policy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLA3Config
    name = "smolvla3"

    def __init__(
        self,
        config: SmolVLA3Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    # HACK(aliberts, danaaubakirova): we overwrite this classmethod here to fix smolVLA-specific issues
    @classmethod
    def _load_as_safetensor(
        cls,
        model: "SmolVLA3Policy",
        model_file: str,
        map_location: str,
        strict: bool,
    ):
        safetensors.torch.load_model(model, model_file, strict=False, device=map_location)
        return load_smolvla(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        actions, boxes, depth_image = actions

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions, boxes, depth_image

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            actions, boxes, depth_image = actions

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
            pass
        else:
            actions = self._get_action_chunk(batch, noise)
            actions, boxes, depth_image = actions
            pass

        return {
            "action": self._queues[ACTION].popleft(),
            "box": boxes,
            "depth_image": depth_image,
        }

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")

        boxes, box_masks = self.prepare_box(batch)
        depth_image = self.prepare_depth(batch)

        loss_dict = {}
        loss_action, loss_box, loss_depth = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, boxes, box_masks, depth_image)
        loss_dict["losses_after_forward"] = loss_action.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            loss_action = loss_action * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = loss_action.clone()

        # Remove padding
        loss_action = loss_action[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = loss_action.clone()
        loss_action = loss_action.mean()

        loss_box = loss_box * box_masks.unsqueeze(-1)
        loss_box = loss_box.reshape(loss_box.shape[0], -1).sum(dim=-1) / box_masks.sum(dim=-1)
        loss_box = loss_box.mean()

        loss_depth = loss_depth.mean()

        # For backward pass
        loss = loss_action + loss_box + loss_depth
        # For backward pass
        loss_dict["loss"] = loss.item()
        loss_dict["loss_action"] = loss_action.item()
        loss_dict["loss_box"] = loss_box.item()
        loss_dict["loss_depth"] = loss_depth.item()

        # logger.info(f"loss_action: {loss_action.item()}, loss_box: {loss_box.item()}, loss_depth: {loss_depth.item()}")
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    def prepare_box(self, batch):
        boxes = batch['bboxes']
        max_num_boxes = self.config.max_num_embeddings_box
        boxes_tensor = torch.zeros(
            (boxes.shape[0], max_num_boxes, 4), device=boxes.device
        )
        boxes_tensor[:, : boxes.shape[1], :] = boxes[:, : max_num_boxes, -4:]
        box_masks = torch.zeros(
            (boxes.shape[0], max_num_boxes), device=boxes.device, dtype=torch.float32)
        box_masks = boxes_tensor.sum(dim=-1) > 1e-8
        return boxes_tensor, box_masks

    def prepare_depth(self, batch):
        depth_image = batch["observation.images.wrist_depth"]
        depth_image = TF.resize(depth_image, self.config.depth_image_size)
        depth_image = depth_image[:, :1, :, :]  # Keep only one channel if there are more
        return depth_image


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class OutputProjectionMLP(nn.Module):
    """MLP for output projection."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU() # swish == silu
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_ = self.input_linear(x)
        x_ = self.activation(x_)
        x_ = self.norm(x_.transpose(1, 2)).transpose(1, 2)  # BatchNorm1d expects (B, C, L)
        x = self.output_linear(x + x_)
        return x


class DepthImageEncoder(nn.Module):
    def __init__(self, depth_image_size, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=4, padding=3)  # (B, 64, H/4, W/4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=7, stride=4, padding=3)  # (B, 128, H/16, W/16)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # (B, 256, H/32, W/32)
        self.bn3 = nn.BatchNorm2d(512)

        self.output_linear = nn.Linear(512, hidden_size)

    def forward(self, depth_image):
        x = self.conv1(depth_image)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # x: (B, 512, H/32, W/32)
        # make x flat: [B, 512, (H/32)*(W/32)]
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)  # (B, (H/32)*(W/32), 512)
        x = self.output_linear(x)  # (B, (H/32)*(W/32), hidden_size)
        return x
    

class DepthImageDecoder(nn.Module):
    def __init__(self, depth_image_size, hidden_size):
        super().__init__()
        self.input_linear = nn.Linear(hidden_size, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # (B, 256, H/16, W/16)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=7, stride=4, padding=3, output_padding=3)  # (B, 64, H/4, W/4)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=4, padding=3, output_padding=3)  # (B, 1, H, W)

    def forward(self, x):
        # x: (B, L, hidden_size)
        b, l, d = x.shape
        h_w = int(math.sqrt(l))
        if h_w * h_w != l:
            raise ValueError(f"Input length {l} is not a perfect square.")
        
        x = self.input_linear(x)  # (B, L, 512)
        x = x.permute(0, 2, 1)  # (B, 512, L)
        x = x.view(b, 512, h_w, h_w)  # (B, 512, H/32, W/32)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv3(x)  # (B, 1, H, W)
        return x

class TimeEmbeddingMerger(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, time_emb):
        # x: (B, L, D)
        # time_emb: (B, D)
        time_emb = time_emb.unsqueeze(1).expand_as(x)  # (B, L, D)
        x = torch.cat([x, time_emb], dim=-1)  # (B, L, 2*D)
        x = self.mlp(x)  # (B, L, D)
        return x
    
class VLAFlowMatching(nn.Module):
    """
    """

    def __init__(self, config: SmolVLA3Config):
        super().__init__()
        self.config = config

        self.vlm = SmolVLM(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            load_vlm_weights=self.config.load_vlm_weights,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm.config.text_config.hidden_size)
        self.box_in_proj = nn.Linear(4, self.vlm.config.text_config.hidden_size)
        self.depth_image_in_proj = DepthImageEncoder(
            depth_image_size=self.config.depth_image_size,
            hidden_size=self.vlm.config.text_config.hidden_size,
        )
        self.action_out_proj = OutputProjectionMLP(
            input_dim=self.vlm.config.text_config.hidden_size,
            output_dim=self.config.max_action_dim,
            hidden_dim=self.vlm.config.text_config.hidden_size,
        )
        self.box_out_proj = OutputProjectionMLP(
            input_dim=self.vlm.config.text_config.hidden_size,
            output_dim=4,
            hidden_dim=self.vlm.config.text_config.hidden_size,
        )
        self.depth_image_out_proj = DepthImageDecoder(
            depth_image_size=self.config.depth_image_size,
            hidden_size=self.vlm.config.text_config.hidden_size,
        )

        self.action_time_merger = TimeEmbeddingMerger(self.vlm.config.text_config.hidden_size)
        self.box_time_merger = TimeEmbeddingMerger(self.vlm.config.text_config.hidden_size)
        self.depth_image_time_merger = TimeEmbeddingMerger(self.vlm.config.text_config.hidden_size)

        self.set_requires_grad()
        self.fake_image_token = self.vlm.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        position_ids = []
        current_position = 0
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                position_ids.append(
                    torch.cumsum(image_start_mask.type(torch.long), dim=1) - 1 + current_position
                )
                current_position += image_start_mask.shape[1]
                pass

            img_emb = self.vlm.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            position_ids.append(
                torch.cumsum(img_mask.type(torch.long), dim=1) - 1 + current_position
            )
            current_position += img_mask.shape[1]

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                position_ids.append(
                    torch.cumsum(image_end_mask.type(torch.long), dim=1) - 1 + current_position
                )
                current_position += image_end_mask.shape[1]
                pass
        lang_emb = self.vlm.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        position_ids.append(
            torch.cumsum(lang_masks.type(torch.long), dim=1) - 1 + current_position
        )

        # restart position for states
        current_position = self.config.max_image_text_length
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        
        position_ids.append(
            torch.cumsum(state_mask.type(torch.long), dim=1) - 1 + current_position
        )
        current_position += state_mask.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        position_ids = torch.cat(position_ids, dim=1)

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            position_ids = pad_tensor(position_ids, self.prefix_length, pad_value=0)
            pass

        return embs, pad_masks, position_ids

    def embed_suffix(self, noisy_actions, noisy_boxes, noisy_depth, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        position_ids = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        box_emb = self.box_in_proj(noisy_boxes)
        depth_emb = self.depth_image_in_proj(noisy_depth)

        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm.config.text_config.hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        action_time_emb = self.action_time_merger(action_emb, time_emb)
        box_time_emb = self.box_time_merger(box_emb, time_emb)
        depth_time_emb = self.depth_image_time_merger(depth_emb, time_emb)

        # Add to input tokens
        embs.append(action_time_emb)
        embs.append(box_time_emb)
        embs.append(depth_time_emb)

        bsize = action_time_emb.shape[0]
        action_mask = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(action_mask)
        start_position_id = self.config.max_image_text_length + self.config.max_state_dim
        position_ids.append(
            torch.cumsum(action_mask.type(torch.long), dim=1) - 1 + start_position_id
        )
        start_position_id += action_mask.shape[1]
        box_mask = torch.ones(bsize, box_time_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(box_mask)
        position_ids.append(
            torch.cumsum(box_mask.type(torch.long), dim=1) - 1 + start_position_id
        )
        start_position_id += box_mask.shape[1]
        depth_mask = torch.ones(bsize, depth_time_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(depth_mask)
        position_ids.append(
            torch.cumsum(depth_mask.type(torch.long), dim=1) - 1 + start_position_id
        )
        start_position_id += depth_mask.shape[1]


        return embs, pad_masks, position_ids

    def forward_suffix(
            self, past_key_values, prefix_pad_masks,
            suffix_embs, suffix_pad_masks, suffix_position_ids,
    ):
        num_action = suffix_embs[0].shape[1]
        num_box = suffix_embs[1].shape[1]

        suffix_embs = torch.cat(suffix_embs, dim=1)
        attention_mask_self = torch.zeros(
            (suffix_embs.shape[0], suffix_embs.shape[1], suffix_embs.shape[1]),
            dtype=suffix_pad_masks[0].dtype,
            device=suffix_embs.device,
        )
        pad_mask_start_idx = 0
        for suffix_pad_mask in suffix_pad_masks:
            smask = suffix_pad_mask[:, None, :] * suffix_pad_mask[:, :, None]
            snum = suffix_pad_mask.shape[1]

            attention_mask_self[
                :, pad_mask_start_idx:pad_mask_start_idx+snum, pad_mask_start_idx:pad_mask_start_idx+snum] = \
                    smask
            pad_mask_start_idx += snum
            pass
        # make diagonal blocks
        suffix_position_ids = torch.cat(suffix_position_ids, dim=1)
        all_suffix_pad_mask = torch.cat(suffix_pad_masks, dim=1)
        attention_mask_cross = prefix_pad_masks[:, None, :] * all_suffix_pad_mask[:, :, None]

        suffix_outs = self.vlm.forward(
            attention_mask_cross=attention_mask_cross,
            attention_mask_self=attention_mask_self,
            position_ids=suffix_position_ids,
            inputs_embeds=suffix_embs,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        action_out = suffix_outs[:, :num_action, :]
        box_out = suffix_outs[:, num_action:num_action+num_box, :]
        depth_out = suffix_outs[:, num_action+num_box:, :]

        return action_out, box_out, depth_out
       
        
    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, boxes, box_masks, depth_image,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        noise_action = self.sample_noise(actions.shape, actions.device)
        noise_box = self.sample_noise((actions.shape[0], self.config.max_num_embeddings_box, 4), actions.device)
        noise_depth = self.sample_noise(
            (actions.shape[0], 1, self.config.depth_image_size, self.config.depth_image_size), actions.device)

        time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t_action = time_expanded * noise_action + (1 - time_expanded) * actions
        u_t_action = noise_action - actions
        x_t_box = time_expanded * noise_box + (1 - time_expanded) * boxes
        u_t_box = noise_box - boxes
        x_t_depth = time_expanded.unsqueeze(-1) * noise_depth + (1 - time_expanded.unsqueeze(-1)) * depth_image
        u_t_depth = noise_depth - depth_image

        prefix_embs, prefix_pad_masks, prefix_position_ids = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_position_ids = self.embed_suffix(
            x_t_action,
            x_t_box,
            x_t_depth,
            time)
        
        with torch.no_grad():
            past_key_values = self.vlm.prepare_for_generation(
                attention_mask=prefix_pad_masks,
                position_ids=prefix_position_ids,
                inputs_embeds=prefix_embs,
            )

            action_out, box_out, depth_out = self.forward_suffix(
                past_key_values, prefix_pad_masks,
                suffix_embs, suffix_pad_masks, suffix_position_ids,
            )
            pass

        v_t_action = self.action_out_proj(action_out)
        v_t_box = self.box_out_proj(box_out)
        v_t_depth = self.depth_image_out_proj(depth_out)

        loss_action = F.mse_loss(u_t_action, v_t_action, reduction="none") * self.config.loss_weights["action"]
        loss_box = F.mse_loss(u_t_box, v_t_box, reduction="none") * self.config.loss_weights["box"]
        loss_depth = F.mse_loss(u_t_depth, v_t_depth, reduction="none") * self.config.loss_weights["depth"]
        

        return loss_action, loss_box, loss_depth

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise_action = self.sample_noise(actions_shape, device)
        else:
            noise_action = noise
            pass
        boxes_shape = (bsize, self.config.max_num_embeddings_box, 4)
        noise_box = self.sample_noise(boxes_shape, device)
        depth_shape = (bsize, 1, self.config.depth_image_size, self.config.depth_image_size)
        noise_depth = self.sample_noise(depth_shape, device)

        prefix_embs, prefix_pad_masks, prefix_position_ids = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        past_key_values = self.vlm.prepare_for_generation(
            attention_mask=prefix_pad_masks,
            position_ids=prefix_position_ids,
            inputs_embeds=prefix_embs,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t_action = noise_action
        x_t_box = noise_box
        x_t_depth = noise_depth

        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t_action, v_t_box, v_t_depth = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t_action,
                x_t_box,
                x_t_depth,
                expanded_time,
            )
            # Euler step
            x_t_action = x_t_action + v_t_action * dt
            x_t_box = x_t_box + v_t_box * dt
            x_t_depth = x_t_depth + v_t_depth * dt
            time += dt
        return x_t_action, x_t_box, x_t_depth

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t_action,
        x_t_box,
        x_t_depth,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_position_ids = self.embed_suffix(
            x_t_action,
            x_t_box,
            x_t_depth,
            timestep)

        action_out, box_out, depth_out = self.forward_suffix(
            past_key_values, prefix_pad_masks,
            suffix_embs, suffix_pad_masks, suffix_position_ids,
        )

        v_t_action = self.action_out_proj(action_out)
        v_t_box = self.box_out_proj(box_out)
        v_t_depth = self.depth_image_out_proj(depth_out)
        return v_t_action, v_t_box, v_t_depth
