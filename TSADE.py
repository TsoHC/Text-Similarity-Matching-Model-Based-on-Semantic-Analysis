from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
# 加载预训练的 TSDAE 模型和分词器
model_name = "TSDAE_pack"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义两个输入文本
text_a = "The new movie is awesome"
text_b = "The new software is so great"

# 使用 TSDAE 模型对文本进行编码
def tsdae_encode(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入移动到GPU(如果可用)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state
    embeddings = embeddings.mean(dim=1)  # 对词嵌入进行平均池化获得句子嵌入
    embeddings = F.normalize(embeddings, p=2, dim=1)  # 对嵌入进行L2归一化
    return embeddings

# 对文本A和文本B进行编码
embedding_a = tsdae_encode(text_a)
embedding_b = tsdae_encode(text_b)

# 计算文本A和文本B的相似度分数
similarity = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

# 输出相似度分数
print(f"文本A: {text_a}")
print(f"文本B: {text_b}")
print(f"相似度: {similarity.item():.4f}")