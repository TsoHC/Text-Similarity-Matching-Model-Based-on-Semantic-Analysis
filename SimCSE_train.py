from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader

model_name = "distilroberta-base"
word_embedding_model = models.Transformer(model_name, max_seq_length=32)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_sentences = []
with open("wikisentences.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            train_sentences.append(line)

train_data = [InputExample(texts=[s, s]) for s in train_sentences]

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)], epochs=20, show_progress_bar=True
)

model.save("output/simcse-model")