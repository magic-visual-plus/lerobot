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
import numpy as np
import pathlib
import time

def calculate_plt_size(attention_layer_num):
    num_layers = attention_layer_num
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    return rows, cols


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    model_name = "/home/superfun77/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct/snapshots/067788b187b95ebe7b2e040b3e4299e342e5b8fd"
    model_name = "/root/autodl-fs/weights/HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    # model_name = "/root/autodl-fs/weights/HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    # 加载处理器和模型（指定本地目录）
    processor = AutoProcessor.from_pretrained(model_name)
    model: SmolVLMForConditionalGeneration = (
        SmolVLMForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,  # pyright: ignore[reportArgumentType]
            torch_dtype=torch.bfloat16,
            # attn_implementation="flex_attention",
            # 显存开销太大，不能用
            attn_implementation="eager",
            # attn_implementation="flash_attention_2",  # 显存开销小，但速度慢，建议用这个
        ).to(device)
    )
    model.eval()
    print("Model Class:", model.__class__)  # 应该是 SmolVLMForVision2Seq
    print("Backbone Class:", model.model.__class__)  # 应该是 SmolVLMModel
    return model, processor


def get_split_image_attention(
    attention_weights, image_split_idxs, row, col, vis_att=False
):
    """
    Reconstruct attention weights from image patches split into a grid.

    Args:
        attention_weights: Tensor of shape (num_heads, seq_len) containing attention weights
        image_split_idxs: List of indices corresponding to image tokens in the sequence
        row: Number of rows in the image grid
        col: Number of columns in the image grid

    Returns:
        Reconstructed attention map as a 2D tensor of shape (row*8, col*8)
    """
    # Each patch has 64 tokens (8x8)
    patch_size = 8
    tokens_per_patch = 64

    # Extract attention weights for image tokens only
    image_attention = attention_weights[:, image_split_idxs]

    # Average across attention heads
    if len(image_attention.shape) > 1:
        image_attention = image_attention.mean(dim=0)

    # Calculate total expected tokens
    expected_tokens = row * col * tokens_per_patch
    actual_tokens = len(image_split_idxs)

    print(f"Expected tokens: {expected_tokens}, Actual tokens: {actual_tokens}")

    # If we have fewer tokens than expected, pad with zeros
    if actual_tokens < expected_tokens:
        padding = torch.zeros(
            expected_tokens - actual_tokens,
            dtype=image_attention.dtype,
            device=image_attention.device,
        )
        image_attention = torch.cat([image_attention, padding], dim=0)
    elif actual_tokens > expected_tokens:
        # If we have more tokens, truncate
        image_attention = image_attention[:expected_tokens]

    # Reshape to patches: (row, col, patch_size, patch_size)
    reshaped_attention = image_attention.view(row, col, patch_size, patch_size)

    # Rearrange to create the full image: (row*patch_size, col*patch_size)
    # This involves moving from (row, col, 8, 8) to (row*8, col*8)
    full_attention = torch.zeros(
        row * patch_size,
        col * patch_size,
        dtype=image_attention.dtype,
        device=image_attention.device,
    )

    for r in range(row):
        for c in range(col):
            start_r = r * patch_size
            end_r = start_r + patch_size
            start_c = c * patch_size
            end_c = start_c + patch_size
            full_attention[start_r:end_r, start_c:end_c] = reshaped_attention[r, c]

    print(f"Reconstructed attention map shape: {full_attention.shape}")

    if vis_att:
        # Visualize the result
        plt.figure(figsize=(8, 6))
        plt.imshow(
            full_attention.detach().cpu().numpy(),
            cmap="plasma",
            interpolation="nearest",
        )
        plt.colorbar()
        plt.title(f"Reconstructed Attention Map ({row}x{col} patches)")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig("reconstructed_attention.png", dpi=150, bbox_inches="tight")
        plt.show()

    print(f"full attention shape {full_attention.shape}")
    return full_attention


def test_mock_get_split_image_attention():
    attention_weights = torch.from_numpy(
        np.random.randint(size=(9, 884), low=0, high=256, dtype=np.uint8) / 255.0
    ).to(torch.float32)
    print(f"attention_weights shape {attention_weights.shape}")
    image_split_idxs = [
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        270,
        271,
        272,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
        298,
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
        327,
        328,
        329,
        330,
        331,
        332,
        333,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        352,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        360,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        382,
        383,
        384,
        385,
        386,
        387,
        388,
        389,
        390,
        391,
        392,
        393,
        394,
        395,
        396,
        397,
        398,
        399,
        402,
        403,
        404,
        405,
        406,
        407,
        408,
        409,
        410,
        411,
        412,
        413,
        414,
        415,
        416,
        417,
        418,
        419,
        420,
        421,
        422,
        423,
        424,
        425,
        426,
        427,
        428,
        429,
        430,
        431,
        432,
        433,
        434,
        435,
        436,
        437,
        438,
        439,
        440,
        441,
        442,
        443,
        444,
        445,
        446,
        447,
        448,
        449,
        450,
        451,
        452,
        453,
        454,
        455,
        456,
        457,
        458,
        459,
        460,
        461,
        462,
        463,
        464,
        465,
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        475,
        476,
        477,
        478,
        479,
        480,
        481,
        482,
        483,
        484,
        485,
        486,
        487,
        488,
        489,
        490,
        491,
        492,
        493,
        494,
        495,
        496,
        497,
        498,
        499,
        500,
        501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
        516,
        517,
        518,
        519,
        520,
        521,
        522,
        523,
        524,
        525,
        526,
        527,
        528,
        529,
        530,
        531,
        535,
        536,
        537,
        538,
        539,
        540,
        541,
        542,
        543,
        544,
        545,
        546,
        547,
        548,
        549,
        550,
        551,
        552,
        553,
        554,
        555,
        556,
        557,
        558,
        559,
        560,
        561,
        562,
        563,
        564,
        565,
        566,
        567,
        568,
        569,
        570,
        571,
        572,
        573,
        574,
        575,
        576,
        577,
        578,
        579,
        580,
        581,
        582,
        583,
        584,
        585,
        586,
        587,
        588,
        589,
        590,
        591,
        592,
        593,
        594,
        595,
        596,
        597,
        598,
        601,
        602,
        603,
        604,
        605,
        606,
        607,
        608,
        609,
        610,
        611,
        612,
        613,
        614,
        615,
        616,
        617,
        618,
        619,
        620,
        621,
        622,
        623,
        624,
        625,
        626,
        627,
        628,
        629,
        630,
        631,
        632,
        633,
        634,
        635,
        636,
        637,
        638,
        639,
        640,
        641,
        642,
        643,
        644,
        645,
        646,
        647,
        648,
        649,
        650,
        651,
        652,
        653,
        654,
        655,
        656,
        657,
        658,
        659,
        660,
        661,
        662,
        663,
        664,
        667,
        668,
        669,
        670,
        671,
        672,
        673,
        674,
        675,
        676,
        677,
        678,
        679,
        680,
        681,
        682,
        683,
        684,
        685,
        686,
        687,
        688,
        689,
        690,
        691,
        692,
        693,
        694,
        695,
        696,
        697,
        698,
        699,
        700,
        701,
        702,
        703,
        704,
        705,
        706,
        707,
        708,
        709,
        710,
        711,
        712,
        713,
        714,
        715,
        716,
        717,
        718,
        719,
        720,
        721,
        722,
        723,
        724,
        725,
        726,
        727,
        728,
        729,
        730,
        733,
        734,
        735,
        736,
        737,
        738,
        739,
        740,
        741,
        742,
        743,
        744,
        745,
        746,
        747,
        748,
        749,
        750,
        751,
        752,
        753,
        754,
        755,
        756,
        757,
        758,
        759,
        760,
        761,
        762,
        763,
        764,
        765,
        766,
        767,
        768,
        769,
        770,
        771,
        772,
        773,
        774,
        775,
        776,
        777,
        778,
        779,
        780,
        781,
        782,
        783,
        784,
        785,
        786,
        787,
        788,
        789,
        790,
        791,
        792,
        793,
        794,
        795,
        796,
        800,
        801,
        802,
        803,
        804,
        805,
        806,
        807,
        808,
        809,
        810,
        811,
        812,
        813,
        814,
        815,
        816,
        817,
        818,
        819,
        820,
        821,
        822,
        823,
        824,
        825,
        826,
        827,
        828,
        829,
        830,
        831,
        832,
        833,
        834,
        835,
        836,
        837,
        838,
        839,
        840,
        841,
        842,
        843,
        844,
        845,
        846,
        847,
        848,
        849,
        850,
        851,
        852,
        853,
        854,
        855,
        856,
        857,
        858,
        859,
        860,
        861,
        862,
        863,
    ]
    row = 3
    col = 4
    get_split_image_attention(attention_weights, image_split_idxs, row, col)


def test_image_infer_with_attention_vis():
    image_path = r"/opt/projects/voyager/data/test_pic/demo1.png"
    image_path = "/opt/projects/voyager/data/test_pic/4_start.png"
    image_path = "/opt/product/lerobot/examples/smolvlm/4_start.png"
    # image_path = "/opt/product/lerobot/examples/smolvlm/demo1.png"
    
    cur_dir = pathlib.Path(__file__).parent.resolve()
    print(f'cur_dir is {cur_dir}')
    image_path = f"{cur_dir}/910_main.jpg"
 
    model, processor = load_model()
    question = "what is the date of the photo?"
    question = "what is the color of the bottle?"
    question = "what is the color of the cabinet?"
    
    question = "push the plate to the front of the stove"
    
    question = "push the bowl to the front of the stove"

    question = "push the bottle to the front of the stove"
    
    question = "bottle the"

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
                    "text": f"{question}",
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
    #
    image_split_idxs = []
    for idx, input_id in enumerate(input_id_list):
        if input_id == image_token_id:
            image_split_idxs.append(idx)
    print(f"total image tokens {len(image_split_idxs)}")
    print(f"image_split_idxs {image_split_idxs}")
    # TODO get from image size
    split_row = 4
    split_col = 4
    pos = input_id_list.index(image_token_id) + 1
    global_img_pos = input_id_list.index(global_img_token_id) + 1
    pos_end = image_split_idxs[-1] + 1
    print(f"image_id token start {pos} end {pos_end}， global_img_pos {global_img_pos}")
    pixel_values = inputs['pixel_values']
    pixel_attention_mask = inputs['pixel_attention_mask']
    # save text embeding index
    text_token_start_index = pos_end + 1
    input_embedings = model.model.text_model.get_input_embeddings()(input_ids).to(input_ids.device)
    embeding_weights = model.model.text_model.get_input_embeddings().weight.to(dtype=torch.float32).detach().cpu().numpy()
    np.save(f"{cur_dir}/smovlm_embeding_weights.npy", embeding_weights)
    
    first_token_embeding = input_embedings[0][text_token_start_index]
    first_token_embeding = first_token_embeding.to(dtype=torch.float32)
    np_first_token = first_token_embeding.detach().cpu().numpy()
    np.save(f"{cur_dir}/smovlm_first_token_embeding.npy", np_first_token)

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
    print(f"attention shape {output.attentions[0].shape}")

    vis_gobal_image_att = True
    if not vis_gobal_image_att:
        output_shape = [8 * split_row, 8 * split_col]

    print(f"output shape {output_shape}")
    
    # plate token
    select_text_token_idx = text_token_start_index + 2
    # put
    # select_text_token_idx = text_token_start_index
    
    # stove token is the last one
    # text_tokens = question.split(' ')
    # select_text_token_idx = text_token_start_index + len(text_tokens) - 1 

    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
    text_id_index = pos_end + 1
    for i, ax in enumerate(axes.flatten()):
        if i < atten_layer_size:
            """
            [i]: 第i层的注意力
            [0, :, -1, ...]: 取第一个样本，所有注意力头，最后一个token的注意力
            pos:pos_end: 只关注对图像token的注意力
            .mean(dim=0): 对所有注意力头求平均
            """
            print(f"head attention weight shape {output.attentions[i][0, :, -1].shape}")
            #  output.attentions[i][0, :, -1, pos:pos_end] shape: (num_heads, seq_len)
            if vis_gobal_image_att:
                att = output.attentions[i][0, :, select_text_token_idx, global_img_pos:pos_end].mean(dim=0)
                general_att = general_output.attentions[i][
                    0, :, select_text_token_idx, global_img_pos:pos_end
                ].mean(dim=0)
            else:
                att = get_split_image_attention(
                    output.attentions[i][0, :, -1],
                    image_split_idxs,
                    split_row,
                    split_col,
                )
                general_att = get_split_image_attention(
                    general_output.attentions[i][0, :, -1],
                    image_split_idxs,
                    split_row,
                    split_col,
                )
            # att shape: (seq_len)
            att = att.to(torch.float32).detach().cpu().numpy()
            general_att = general_att.to(torch.float32).detach().cpu().numpy()
            # att = att / general_att
            ax.imshow(att.reshape(output_shape), cmap="plasma", interpolation="nearest")
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    timestamp = int(time.time() * 1000)
    file_name = f"{cur_dir}/smovlm_attention_map_{timestamp}.png"
    plt.savefig(file_name)
    plt.show()


def test_image_infer():

    image_path = "/opt/projects/voyager/data/test_pic/demo1.png"
    image_path = "/opt/product/lerobot/examples/smolvlm/demo1.png"
    image_path = "/opt/product/lerobot/examples/smolvlm/4_start.png"

    model, processor = load_model()

    question = "what is the date of the photo?"
    question = (
        "what is the license plate number of the gray card in the center of the photo?"
    )
    question = "what is the color of the cabinet?"
    
    # question = "what is the color of the bottle?"
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

    print(f'query {question}')
    inputs = processor.apply_chat_template(
        messages,
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
    # test_mock_get_split_image_attention()
