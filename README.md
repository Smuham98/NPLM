# NPLM (Neural Probabilistic Language Model)

This repository contains an implementation of the **Neural Probabilistic Language Model (NPLM)**. The goal of this project is to train a neural network-based language model to predict the next word in a sequence based on a given context.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Configuration](#model-configuration)
- [License](#license)

## Overview

The **NPLM** is a neural network-based language model that uses an embedding layer to map words to dense vectors, followed by an LSTM or GRU to capture context. The model is trained on text data and learns to predict the next word in a sequence, making it suitable for various NLP tasks such as text generation, language modeling, and sequence prediction.

This repository includes scripts for:
- Preprocessing the dataset
- Building the vocabulary
- Training the model
- Evaluating model performance

## Features

- **Tokenizer**: Built to tokenize text data into words and handle punctuation.
- **Vocabulary Building**: Automatically constructs a vocabulary from a dataset based on frequency.
- **Model Architecture**: Implements a simple neural network with word embeddings, LSTM layers, and a softmax output for next-token prediction.
- **Configurable**: Easily configurable via YAML files for model dimensions, learning rate, and more.
- **Data Handling**: Supports JSONL formatted datasets for tokenized text.
- **Pretrained Models**: Optionally load pretrained word embeddings and models.

## Installation

### Prerequisites

- **Python 3.x**
- **PyTorch**: The framework for model training.
- **HuggingFace**: For tokenization and dataset management.
- **Other dependencies**: Listed in `requirements.txt`.

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Smuham98/NPLM.git
   cd NPLM
````

2. Set up a Python virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses **Wikipedia text data** (e.g., Wikitext) for training the language model. The dataset can be downloaded from Kaggle or other sources and should be preprocessed into JSONL format (one document per line).

### Dataset Preprocessing

1. Place the raw dataset in `data/raw/` directory.
2. Run the preprocessing script to convert the dataset into JSONL format:

   ```bash
   python -m nplm.preprocess --input_dir data/raw/wikitext2 --output_dir data/wikitext2_jsonl --shard_size 10000 --lowercase
   ```

## Training

### Configuration

The model training can be customized using YAML configuration files. The configuration files can be found in the `configs/` directory. Key parameters include:

* **context_size**: Size of the context window (how many previous tokens the model will consider).
* **embedding_dim**: Dimensionality of the token embeddings.
* **hidden_dim**: Size of the hidden layers in the model.
* **batch_size**: Number of samples per batch during training.
* **max_steps**: Total number of training steps.

### Train the Model

To start training, simply run the following command:

```bash
python -m nplm.train --config configs/tiny.yaml --data_dir data/wikitext2_encoded --save_dir runs/exp1
```

This will begin the training process and save the model checkpoints in the `runs/exp1` directory.

## Evaluation

After training the model, you can evaluate it on a held-out test set:

```bash
python -m nplm.eval --checkpoint runs/exp1/best.pt --data_dir data/wikitext2_encoded --out_json results/metrics.json
```

This will compute metrics such as **perplexity** and save them in the `results/metrics.json` file.

## Model Configuration

There are two main configuration files provided in the `configs/` directory:

1. **`tiny.yaml`**: A smaller model for quick experimentation and testing.
2. **`medium.yaml`**: A larger model with more capacity for better performance.

You can configure the following parameters in these files:

* **`embedding_dim`**: Size of the word embeddings.
* **`hidden_dim`**: Size of the hidden layers in the LSTM.
* **`lr`**: Learning rate for the optimizer.
* **`batch_size`**: The batch size used during training.
* **`max_steps`**: Total number of steps to train for.

You can specify which configuration to use by passing it to the training script:

```bash
python -m nplm.train --config configs/tiny.yaml --data_dir data/wikitext2_encoded --save_dir runs/exp1
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

````

### **Instructions**:
1. **Copy** the content above.
2. **Paste** it into your `README.md` file.
3. **Commit and push** to GitHub:

```bash
git add README.md
git commit -m "Added high-level README for NPLM project"
git push origin main
````
