from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader

# Define your sentence transformer model using CLS pooling
model_name = "bert-base-uncased"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read sentences from a text file
train_sentences = []
with open("path/to/your/text/file.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            train_sentences.append(line)

# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

model.save("output/tsdae-model")