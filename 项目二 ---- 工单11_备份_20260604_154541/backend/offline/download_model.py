"""下载 BGE-base-en-v1.5 模型"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

print("正在下载 BAAI/bge-base-en-v1.5 ...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
print(f"下载完成！向量维度: {model.get_sentence_embedding_dimension()}")

# 测试编码
test = model.encode(["测试句子"])
print(f"编码测试通过，shape: {test.shape}")
