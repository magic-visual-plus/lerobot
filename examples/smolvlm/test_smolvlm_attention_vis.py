from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import (
    GenerationConfig,
    SmolVLMConfig,
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
)
from PIL import Image
import matplotlib.pyplot as plt
import math


def calculate_plt_size(attention_layer_num):
    """
    calculate_plt_size 的作用是计算绘图网格的行列数：

    输入: attention_layer_num - 注意力层的总数量
    功能: 根据层数计算出最佳的网格布局（行数和列数）
    算法逻辑:
    先计算总层数的平方根并向上取整作为列数
    再根据列数计算需要多少行来容纳所有层
    目标是创建一个尽可能接近正方形的网格布局
    举例说明:

    如果有28层注意力，√28 ≈ 5.3，向上取整得到6列
    28层需要 ⌈28/6⌉ = 5行
    最终得到5×6的网格来显示28个注意力热力图

    Args:
        attention_layer_num (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_layers = attention_layer_num
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    return rows, cols


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    model_name = "/home/superfun77/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct/snapshots/067788b187b95ebe7b2e040b3e4299e342e5b8fd"

    # 加载处理器和模型（指定本地目录）
    processor = AutoProcessor.from_pretrained(model_name)
    model: SmolVLMForConditionalGeneration = (
        SmolVLMForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,  # pyright: ignore[reportArgumentType]
            torch_dtype=torch.bfloat16,
            # attn_implementation="flex_attention",
            # 显存开销太大，不能用
            # attn_implementation="eager",
            attn_implementation="flash_attention_2",  # 显存开销小，但速度慢，建议用这个
            # attn_implementation="eager_with_mem_eff_2
        ).to(device)
    )
    model.eval()
    print("Model Class:", model.__class__)  # 应该是 SmolVLMForVision2Seq
    print("Backbone Class:", model.model.__class__)  # 应该是 SmolVLMModel
    return model, processor


def test_image_infer_with_attention_vis():
    image_path = r"/opt/projects/voyager/data/test_pic/demo1.png"
    image_path = "/opt/projects/voyager/data/test_pic/4_start.png"
    model, processor = load_model()
    question = "what is the date of the photo?"

    general_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {
                    "type": "text",
                    "text": "Write a general description of the image. Answer the question using a single word or phrase.",
                },
            ],
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {
                    "type": "text",
                    "text": f"{question} Answer the question using a single word or phrase",
                },
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    general_inputs = processor.apply_chat_template(
        general_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    image_token_text = processor.image_token
    print(f"image token text: {image_token_text}")
    image_token_id = processor.image_token_id
    input_tokens = processor.tokenizer.batch_decode(input_ids)
    global_img_token_id = processor.tokenizer.convert_tokens_to_ids("<global-img>")
    print(f"input_tokens {input_tokens}")
    input_id_list = input_ids.tolist()[0]
    pos = input_id_list.index(image_token_id) + 1
    global_img_pos = input_id_list.index(global_img_token_id) + 1
    pos_end = len(input_id_list) - input_id_list[::-1].index(image_token_id)
    print(f"image_id token start {pos} end {pos_end}， global_img_pos {global_img_pos}")

    output = model.forward(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
        output_attentions=True,
        output_hidden_states=True,
    )

    general_output = model.forward(
        **general_inputs,
        do_sample=False,
        max_new_tokens=64,
        output_attentions=True,
        output_hidden_states=True,
    )

    for i, hidden_state in enumerate(output.hidden_states):
        print(f"Layer {i} seq_len: {hidden_state.shape[1]}")

    atten_layer_size = len(output.attentions)
    print(f"Attention layer size: {atten_layer_size}")
    rows, cols = calculate_plt_size(atten_layer_size)
    output_shape = [8, 8]
    print(f'attention shape {output.attentions[0].shape}')

    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
    for i, ax in enumerate(axes.flatten()):
        if i < atten_layer_size:
            #  output.attentions[i][0, :, -1, pos:pos_end] shape: (num_heads, seq_len)
            att = output.attentions[i][0, :, -1, global_img_pos:pos_end].mean(dim=0)
            # att shape: (seq_len)
            att = att.to(torch.float32).detach().cpu().numpy()
            general_att = general_output.attentions[i][0, :, -1, global_img_pos:pos_end].mean(
                dim=0
            )
            general_att = general_att.to(torch.float32).detach().cpu().numpy()
            att = att / general_att
            ax.imshow(att.reshape(output_shape), cmap="plasma", interpolation="nearest")
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig("attention_map.png")
    plt.show()


def test_image_infer():

    image_path = r"/opt/projects/voyager/data/test_pic/demo1.png"

    model, processor = load_model()

    question = "what is the date of the photo?"

    general_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {
                    "type": "text",
                    "text": "Write a general description of the image. Answer the question using a single word or phrase.",
                },
            ],
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {
                    "type": "text",
                    "text": f"{question} Answer the question using a single word or phrase",
                },
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        general_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
        output_attentions=True,
        output_hidden_states=True,
    )
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])


if __name__ == "__main__":
    # test_image_infer()
    test_image_infer_with_attention_vis()
