

import torch.nn as nn
from transformers.models.llama import LlamaConfig, LlamaModel
from transformers.cache_utils import DynamicCache


class ExpertModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size):
        super().__init__()
        text_config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
        )
        self.expert = LlamaModel(text_config)
        self.dtype = text_config.torch_dtype
        pass


    def prepare(self, query_embed, attention_mask):
        
        result = self.expert.forward(
            inputs_embeds=query_embed,
            attention_mask=attention_mask.unsqueeze(1),
            use_cache=True,
        )

        past_key_values = result.past_key_values
        out_embeds = result.last_hidden_state
        return past_key_values.to_legacy_cache(), out_embeds

    def decode(
        self,
        attention_mask,
        inputs_embeds,
        past_key_values,
        use_cache=True,
    ):
        result = self.expert.forward(
            attention_mask=attention_mask.unsqueeze(1),
            inputs_embeds=inputs_embeds,
            past_key_values=DynamicCache.from_legacy_cache(past_key_values),
            use_cache=use_cache,
        )
        return result.last_hidden_state
    
    def forward(
            self, **kwargs):
        return self.expert.forward(**kwargs)