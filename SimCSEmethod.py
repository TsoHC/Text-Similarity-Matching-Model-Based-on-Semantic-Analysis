from sentence_transformers import SentenceTransformer, util
import torch
path = "princeton-nlp/sup-simcse-bert-base-uncased"

model = SentenceTransformer(path)

# 定义两个输入文本
text_a = "The new movie is awesome"
text_b = "The new movie is so bad"
# 使用SimCSE模型对文本进行编码
def simcse_encode(text):
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding

# 对文本A和文本B进行编码
embedding_a = simcse_encode(text_a)
embedding_b = simcse_encode(text_b)

# 计算文本A和文本B的相似度分数
similarity = util.cos_sim(embedding_a, embedding_b)

# 输出相似度分数
print(f"文本A: {text_a}")
print(f"文本B: {text_b}")
print(f"相似度: {similarity.item()}")