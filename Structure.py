from sentence_transformers import SentenceTransformer, util
import torch
path = 'all-MiniLM-L6-v2'
from torch import Tensor

def cosSimilarity(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# 加载预训练的BERT模型

model = SentenceTransformer(path)

# 定义两个输入文本
text_a = "I love natural language processing."
text_b = "I am interesting in natural language processings."

# 将文本编码为嵌入向量
embedding_a = model.encode(text_a)
embedding_b = model.encode(text_b)

# 将嵌入向量转换为PyTorch张量
embedding_a = torch.tensor(embedding_a).unsqueeze(0)
embedding_b = torch.tensor(embedding_b).unsqueeze(0)

# 计算交互注意力矩阵
attention_matrix = torch.matmul(embedding_a, embedding_b.transpose(0, 1))

# 对注意力矩阵进行softmax归一化
attention_matrix = torch.softmax(attention_matrix, dim=-1)

# 计算文本A和文本B的交互表示
interactive_a = torch.matmul(attention_matrix, embedding_b)
interactive_b = torch.matmul(attention_matrix.transpose(0, 1), embedding_a)

# 将原始嵌入和交互表示拼接起来
enhanced_a = torch.cat([embedding_a, interactive_a], dim=-1)
enhanced_b = torch.cat([embedding_b, interactive_b], dim=-1)

# 计算最终的相似度分数
similarity = cosSimilarity(enhanced_a, enhanced_b)



# 输出相似度分数
print(f"文本A: {text_a}")
print(f"文本B: {text_b}")
print(f"相似度: {similarity.item()}")

