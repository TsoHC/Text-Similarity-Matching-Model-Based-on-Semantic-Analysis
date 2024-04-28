from sentence_transformers import SentenceTransformer, util
import torch
path = "bert-base-uncased"

# 加载预训练的all-MiniLM-L6-v2模型
model = SentenceTransformer(path)

# 定义MLM的掩码概率
mlm_probability = 0.001

# 定义两个输入文本
text_a =  "The cat sits outside"
text_b =  "The cat plays outside"


# 对输入文本应用MLM
def apply_mlm(text):
    # 将文本编码为嵌入向量
    embedding = model.encode(text, convert_to_tensor=True)
    # 随机选择要掩码的标记
    masked_indices = torch.bernoulli(torch.full(embedding.shape, mlm_probability)).bool()
    masked_embedding = embedding.clone()
    masked_embedding[masked_indices] = 0.0
    return masked_embedding



# 对文本A和文本B应用MLM

embedding_a = apply_mlm(text_a).unsqueeze(0)  # 在第0维添加一个维度
embedding_b = apply_mlm(text_b).unsqueeze(0)  # 在第0维添加一个维度

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
similarity = util.cos_sim(enhanced_a.squeeze(0), enhanced_b.squeeze(0))  # 移除额外的维度

# 输出相似度分数
print(f"文本A: {text_a}")
print(f"文本B: {text_b}")
print(f"相似度: {similarity.item()}")