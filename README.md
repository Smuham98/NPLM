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
