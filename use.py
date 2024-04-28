import torch
from transformers import AutoTokenizer, AutoModel

# 定义SimCSE模型类
class SimCSE(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc1 = torch.nn.Linear(self.encoder.config.hidden_size, 512)
        self.activation = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.fc1(pooled_output)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# 加载训练好的模型
model = SimCSE('bert-base-uncased')
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 准备要比较的文本对
text1 = "two dogs like a cat."
text2 = "A woman is playing the violin."

# 对文本进行编码
encoded_text1 = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
encoded_text2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')

# 使用模型生成嵌入向量
with torch.no_grad():
    embedding1 = model(encoded_text1['input_ids'], encoded_text1['attention_mask'])
    embedding2 = model(encoded_text2['input_ids'], encoded_text2['attention_mask'])

# 压缩嵌入向量
embedding1 = embedding1.squeeze()
embedding2 = embedding2.squeeze()

# 计算嵌入向量之间的相似度
similarity = torch.cosine_similarity(embedding1, embedding2, dim=0)
print(f"Similarity: {similarity.item():.4f}")