from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
import math
import time
import pathlib


def calculate_plt_size(attention_layer_num):
    num_layers = attention_layer_num
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    return rows, cols


device = "cuda"
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
model_path = r"E:\dev\wsl\mllms_know\ckpts\Qwen\Qwen2___5-VL-3B-Instruct"
model_path = "/root/autodl-fs/weights/Qwen/Qwen2___5-VL-3B-Instruct"

model = (
    Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    .eval()
    .to(device)
)
processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=True, padding_side="left", use_fast=True
)

image_path = "./images/demo1.png"
image_path = "./images/4_start.png"
image_path = "/opt/product/lerobot/examples/smolvlm/4_start.png"
cur_dir = pathlib.Path(__file__).parent.resolve()
print(f'cur_dir is {cur_dir}')
image_path = f"{cur_dir}/910_main.jpg"

question = "what is the date of the photo?"

question = "what is the color of the cabinet?"

# question = "what is the color of the bottle?"

question = "push the plate to the front of the stove"

question = "push the bowl to the front of the stove"


messages_query = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "max_pixels": 512 * 28 * 28},
            {
                "type": "text",
                "text": f"{question} Answer the question using a single word or phrase.",
            },
        ],
    }
]

image_inputs, _ = process_vision_info(messages_query)

text_query = processor.apply_chat_template(
    messages_query, tokenize=False, add_generation_prompt=True
)

print(f'text_query {text_query}')

inputs = processor(
    text=[text_query],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

messages_general = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "max_pixels": 512 * 28 * 28},
            {
                "type": "text",
                "text": "Write a general description of the image. Answer the question using a single word or phrase.",
            },
        ],
    }
]

text_general = processor.apply_chat_template(
    messages_general, tokenize=False, add_generation_prompt=True
)

general_inputs = processor(
    text=[text_general],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

image_inputs_aux = processor.image_processor(images=image_inputs)
output_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:] / 2
output_shape = output_shape.astype(int)
print(f"output_shape: {output_shape}")

with torch.no_grad():
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    pos = inputs["input_ids"].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs["input_ids"].tolist()[0].index(vision_end_token_id)
    print(f'pos start {pos} end {pos_end}')
    print(f"Input seq_len: {inputs['input_ids'].shape[1]}")

    output = model(**inputs, output_attentions=True, output_hidden_states=True)
    general_output = model(
        **general_inputs, output_attentions=True, output_hidden_states=True
    )

    for i, hidden_state in enumerate(output.hidden_states):
        print(f"Layer {i} seq_len: {hidden_state.shape[1]}")

    atten_layer_size = len(output.attentions)
    rows, cols = calculate_plt_size(atten_layer_size)
    print(f'attention shape {output.attentions[0].shape}')
    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
    # print(f"axes: {axes.flatten()}")
    text_token_start_index = pos_end + 1
    select_text_token_idx = text_token_start_index + 2
    select_text_token_idx = text_token_start_index + 1
    
    # stove token is the last one
    # text_tokens = question.split(' ')
    # select_text_token_idx = text_token_start_index + len(text_tokens) - 1 
    
    # text_id_index = -1
    for i, ax in enumerate(axes.flatten()):
        if i < atten_layer_size:
            #  output.attentions[i][0, :, -1, pos:pos_end] shape: (num_heads, seq_len)
            att = output.attentions[i][0, :, select_text_token_idx, pos:pos_end].mean(dim=0)
            # att shape: (seq_len)
            att = att.to(torch.float32).detach().cpu().numpy()

            general_att = general_output.attentions[i][0, :, select_text_token_idx, pos:pos_end].mean(
                dim=0
            )
            general_att = general_att.to(torch.float32).detach().cpu().numpy()

            # TODO enable later
            # att = att / general_att

            # seq len -> image w, h
            # ax.imshow(
                # att.reshape(output_shape), cmap="viridis", interpolation="nearest"
            # )
            ax.imshow(
                att.reshape(output_shape), cmap="plasma", interpolation="nearest"
            )
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        else:
            ax.axis("off")

    
    plt.tight_layout()
    timestamp = int(time.time() * 1000)
    file_name = f"{cur_dir}/qwen2_attention_map_{timestamp}.png"
    plt.savefig(file_name)