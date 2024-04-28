import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def cosine_similarity(embedding1, embedding2):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    similarity = torch.matmul(embedding1, embedding2.T)
    return similarity

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)

    # 加载训练好的模型参数
    checkpoint_path = 'simcse_model/checkpoint_3.pt'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 输入两个句子
    sentence1 = "The quick brown fox jumps over the lazy dog."
    sentence2 = "A slow yellow dog leaps over the idle canine."

    encoding1 = tokenizer(sentence1, return_tensors='pt').to(device)
    encoding2 = tokenizer(sentence2, return_tensors='pt').to(device)

    with torch.no_grad():
        output1 = model(**encoding1)
        output2 = model(**encoding2)

        embedding1 = output1.last_hidden_state[:, 0, :]
        embedding2 = output2.last_hidden_state[:, 0, :]

        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Similarity: {similarity.item():.4f}")

if __name__ == '__main__':
    main()