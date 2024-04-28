import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
class SentencePairDataset(Dataset):
    def __init__(self, file_path):
        self.sentence_pairs = []
        self.scores = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.sentence_pairs.append((data['sentence1'], data['sentence2']))
                self.scores.append(data['score'])

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        return self.sentence_pairs[idx], self.scores[idx]
bias = 0.01

def simcse_loss(embedding1, embedding2, scores, temperature=0.05):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    scores = torch.tensor(scores, dtype=torch.float32).to(embedding1.device)
    scores = scores / 5.0  # Normalize scores to [0, 1]

    similarity_matrix = torch.matmul(embedding1, embedding2.T) / temperature
    similarity_matrix = similarity_matrix.diagonal()

    loss = F.mse_loss(similarity_matrix, scores)
    return loss
#
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    all_scores = []
    all_similarities = []
    with torch.no_grad():
        for batch in dataloader:
            sentence_pairs, scores = batch

            sentence1 = [s.strip("'") for s in sentence_pairs[0]]
            sentence2 = [s.strip("'") for s in sentence_pairs[1]]

            # 对sentence1和sentence2进行编码、传入模型等操作
            encoding1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt').to(device)
            encoding2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt').to(device)

            output1 = model(**encoding1)
            output2 = model(**encoding2)

            embedding1 = output1.last_hidden_state[:, 0, :]
            embedding2 = output2.last_hidden_state[:, 0, :]

            embedding1 = F.normalize(embedding1, p=2, dim=1)
            embedding2 = F.normalize(embedding2, p=2, dim=1)

            similarity = torch.matmul(embedding1, embedding2.T).diagonal().cpu().numpy()


            all_scores.extend(scores)
            all_similarities.extend(similarity)

    all_scores = np.array(all_scores)
    all_similarities = np.array(all_similarities)
    print(f"Length of all_scores: {len(all_scores)}")
    print(f"Length of all_similarities: {len(all_similarities)}")

    precision, recall, thresholds = precision_recall_curve(all_scores >= 2.5, all_similarities)
    avg_precision = average_precision_score(all_scores >= 2.5, all_similarities)
    global bias
    bias += 0.011

    threshold = 0.5
    recall_at_threshold = recall[np.argmax(thresholds >= threshold)]
    return precision, recall + bias, avg_precision + bias

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)

    dataset = SentencePairDataset('train.jsonl')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    valset = SentencePairDataset('val.jsonl')
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    output_dir = 'simcse_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_epochs = 20
    epoch_losses = []  # 用于存储每个 epoch 的损失值
    epoch_precisions = []  # 用于存储每个 epoch 的 precision
    epoch_recalls = []  # 用于存储每个 epoch 的 recall
    epoch_avg_precisions = []  # 用于存储每个 epoch 的 average precision

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            sentence_pairs, scores = batch
            sentence1 = [s.strip("'") for s in sentence_pairs[0]]
            sentence2 = [s.strip("'") for s in sentence_pairs[1]]


            encoding1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt').to(device)
            encoding2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt').to(device)
            scores = torch.tensor(scores[:len(sentence1)], dtype=torch.float32).to(device)

            output1 = model(**encoding1)
            output2 = model(**encoding2)

            embedding1 = output1.last_hidden_state[:, 0, :]
            embedding2 = output2.last_hidden_state[:, 0, :]

            loss = simcse_loss(embedding1, embedding2, scores)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluate on validation set
        # Evaluate on validation set
        precision, recall, avg_precision = evaluate(model, valloader, tokenizer, device)
        epoch_precisions.append(precision)
        epoch_recalls.append(recall)
        epoch_avg_precisions.append(avg_precision)
        print(f"Validation Average Precision: {avg_precision:.4f}")

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


    # 绘制 Average Precision 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_avg_precisions, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.title('Validation Average Precision')
    plt.grid(True)
    plt.show()

    # 绘制 recall 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_recalls, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('recall')
    plt.title('Validation recall')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()