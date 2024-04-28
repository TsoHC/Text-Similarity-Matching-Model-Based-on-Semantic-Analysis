import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class SentencePairDataset(Dataset):
    def __init__(self, file_path):
        self.sentence_pairs = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.sentence_pairs.append((data['sentence1'], data['sentence2']))

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        return self.sentence_pairs[idx]


def simcse_loss(embedding1, embedding2, temperature=0.05):
    batch_size = embedding1.shape[0]
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    features = torch.cat([embedding1, embedding2], dim=0)
    labels = torch.arange(batch_size, device=embedding1.device)
    labels = torch.cat([labels, labels], dim=0)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity_matrix.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    similarity_matrix = similarity_matrix / temperature

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)

    dataset = SentencePairDataset('train.jsonl')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    output_dir = 'simcse_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_epochs = 20
    epoch_losses = []  # 用于存储每个 epoch 的损失值
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        k = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            sentence_pairs = batch
            sentence1 = [pair[0] for pair in sentence_pairs]
            sentence2 = [pair[1] for pair in sentence_pairs]
            k = k + 1
            encoding1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt').to(device)
            encoding2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt').to(device)

            output1 = model(**encoding1)
            output2 = model(**encoding2)

            embedding1 = output1.last_hidden_state[:, 0, :]
            embedding2 = output2.last_hidden_state[:, 0, :]

            loss = simcse_loss(embedding1, embedding2)
            epoch_loss += loss.item()
            if k == 60:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_{epoch + 1}.pt')
        torch.save(model.state_dict(), checkpoint_path)

    # 绘制损失函数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()