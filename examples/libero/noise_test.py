import torch

# 固定随机数种子
torch.manual_seed(42)  # 42可以换成任意整数
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator(device="cuda")
g.manual_seed(42)


for i in range(5):
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(3, 3),
        generator=g,
        dtype=torch.float32,
        device="cuda",
    )
    print(noise)
