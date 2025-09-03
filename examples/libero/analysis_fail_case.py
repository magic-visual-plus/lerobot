from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Messages containing a video url and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/opt/projects/news/lerobot/examples/libero/rollout_task_3_episode_34_open_the_top_drawer_and_put_the_bowl_inside_failure.mp4",
            },
            {"type": "text", "text": "这是一个机器人验证视频，任务描述是， 打开最上面的抽屉，把碗放进去， 这个视频显示的任务失败了， 请分析失败的任务的原因， 用简短的语句描述下"},
        ],
    }
]

model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name_or_path)

fps = 10

#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
