from loguru import logger
import pathlib
import os
import numpy as np

cur_dir = pathlib.Path(__file__).parent.resolve()

emb1_np = np.load('/opt/product/lerobot/examples/smolvlm/smovla_first_token_embeding.npy')
emb2_np = np.load('/opt/product/lerobot/examples/smolvlm/smovlm_first_token_embeding.npy')

cos_sim = np.dot(emb1_np, emb2_np) / (
    np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)
)
# print(f"emb cosine similarity: {cos_sim}")
euclidean_dist = np.linalg.norm(emb1_np - emb2_np)
print(emb1_np[:100])
print(emb2_np[:100])
print(f"first token bottle cos_sim {cos_sim}")
print(f"first token bottle euclidean_dist {euclidean_dist}")