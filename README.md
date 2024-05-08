

# User Manual for Text Semantic Similarity Project

## Introduction

This project implements four BERT pre-training methods for Text Semantic Similarity tasks using PyTorch and the Hugging Face Transformers library. The implemented methods include:

1. Contrastive Tension (CT)
2. Masked Language Modeling (MLM)
3. Simple Contrastive Learning of Sentence Embeddings (SimCSE)
4. Denoising Auto-Encoder (TSDAE)

The project uses the Wikipedia Sentences dataset from Kaggle for training and testing.

## Environment Setup

To run the code in this project, you need to set up the following environment:

- Python 3.11
- PyTorch 2.2.2
- Hugging Face Transformers library
- sentence-transformers library

You can install the required Python libraries using the following command:

```
pip install torch transformers sentence-transformers
```

## Data Preparation

The Wikipedia Sentences dataset used in this project can be downloaded from the following link:

https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences?resource=download

After downloading the dataset, place the file in the project root directory and ensure that the file name is `wikisentences.txt`.

## Model Training

The project provides four Python scripts, each corresponding to one of the four BERT pre-training methods:

1. `CT_train.py`: Contrastive Tension method
2. `MLM_train.py`: Masked Language Modeling method
3. `SimCSE_train.py`: Simple Contrastive Learning of Sentence Embeddings method
4. `TSDAE_train.py`: Denoising Auto-Encoder method

You can run the corresponding script using the following commands:

```
python CT_train.py
python MLM_train.py
python SimCSE_train.py
python TSDAE_train.py
```

Each script will generate the corresponding model files in the `output` directory.

## Model Evaluation

After training, you can use the evaluation functions provided by the sentence-transformers library to assess the performance of the models. Please refer to the official documentation of the sentence-transformers library for specific evaluation methods.

## Graphical User Interface (GUI)

In addition to the command-line scripts, this project also includes a simple GUI interface built using QT Designer. To launch the GUI, run the `UImain.py` script:

```
python UImain.py
```

The GUI interface consists of two main components:
1. Input Box: Enter keywords or related paper titles in this box.
2. Output Box: After entering the input, the output box will display the top 10 most similar paper titles along with their similarity scores.

The GUI provides a user-friendly way to interact with the trained models and retrieve the most relevant paper titles based on the provided input.

## Notes

1. The training process may consume a large amount of GPU resources. Please ensure that you have sufficient hardware support.
2. The training time may vary for different methods. Please be patient and wait for the training to complete.
3. If you encounter any issues during the training process, please check the dataset path, dependency library versions, and other relevant settings.
4. The GUI interface is a basic implementation and may have limitations. Feel free to enhance and customize the interface based on your specific requirements.

