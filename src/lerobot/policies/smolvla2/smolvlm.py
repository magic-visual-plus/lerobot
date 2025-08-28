# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import copy

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMModel,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLM(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        load_vlm_weights: bool = True,
        freeze_vision_encoder: bool = False,
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
            self.vlm = self.vlm.model
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMModel(config=config)
            pass
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config

        self.hidden_size = config.text_config.hidden_size

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False


    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()


    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)


    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
    ):
        num_action_emb = inputs_embeds[1].shape[1]
        embs = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)
        position_ids = torch.cat(position_ids, dim=1)

        # attention_mask: [B, L]
        attention_mask = attention_mask[:, None, :] * attention_mask[:, :, None]
        # attention_mask: [B, L, L]
        # action should not attend to observations
        # attention_mask[:, :-num_action_emb, -num_action_emb:] = 0
        attention_mask = attention_mask.unsqueeze(1)  # [B, 1, L, L]
        
        dtype_vlm = self.get_vlm_model().text_model.get_input_embeddings().weight.dtype
        dtype = embs.dtype
        embs = embs.to(dtype=dtype_vlm)
        # attention_mask = attention_mask.to(dtype=dtype_vlm)
        if past_key_values is not None:
            past_key_values = [
                (past_kv[0].to(dtype=dtype_vlm), past_kv[1].to(dtype=dtype_vlm))
                for past_kv in past_key_values
            ]
            pass
        vlm_outputs = self.vlm.forward(
            attention_mask=attention_mask,
            inputs_embeds=embs,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        embs = vlm_outputs.last_hidden_state
        past_key_values = vlm_outputs.past_key_values
        embs = embs.to(dtype=dtype)
        if past_key_values is not None:
            past_key_values = [
                (past_kv[0].to(dtype=dtype), past_kv[1].to(dtype=dtype)) for past_kv in past_key_values
            ]
            pass

        outputs_embeds = []
        start_idx = 0
        for i, emb in enumerate(inputs_embeds):
            outputs_embeds.append(
                embs[:, start_idx : start_idx + emb.shape[1], :].to(dtype=emb.dtype)
            )
            start_idx += emb.shape[1]
            pass

        return outputs_embeds, past_key_values
