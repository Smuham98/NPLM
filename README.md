# Neural Probabilistic Language Model (NPLM) Assignment

## Overview
This project implements a **Neural Probabilistic Language Model (NPLM)**, inspired by **Bengio et al. (2003)**, aiming to predict the next word in a sequence using a feed-forward neural network. The task is to develop an end-to-end machine learning pipeline, covering data preprocessing, model building, training, and evaluation. The key evaluation metric is **perplexity**, which is used to gauge the model's prediction accuracy.

## Project Structure

- **`data/`**: Contains the raw and preprocessed data
  - `text8/`: Original downloaded corpus
  - `text8_sentences.jsonl`: Preprocessed sentences

- **`results/`**: Stores experiment results and metrics
  - `metrics.json`: Key evaluation metrics
  - `EXPERIMENTS.md`: Detailed experiment logs
  - `LLM_LOG.md`: Log of LLM interactions
  - `word_to_idx.json`: Word to integer ID mapping
  - `idx_to_word.json`: Integer ID to word mapping

- **`nplm-main/`**: Main source code
  - `README.md`: This comprehensive README file
  - `ASSIGNMENT.md`: Original assignment description
  - `Main.ipynb`: Main project file

## Setup and Installation

Follow these steps to set up the project:

### 1. Clone the Repository
Start by cloning the project repository:
```bash
git clone <URL-of-instructor-repo>
cd nplm-main
````

### 2. Configure Remote Repositories

Remove the instructor's remote and set your own:

```bash
git remote remove origin
git remote add origin git@github.com:<your-username>/<your-repo>.git
git push -u origin main
```

### 3. Install Dependencies

Install the required Python libraries using:

```bash
pip install nltk torch scikit-learn
```

Download necessary NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 4. Data Download

The `text8` corpus is automatically downloaded and preprocessed in the notebook.

## Data Preprocessing

The dataset used is the `text8` corpus, which consists of 100 million characters from Wikipedia text. The preprocessing pipeline involves:

* **Text Cleaning:** Lowercasing and whitespace normalization.
* **Chunking:** The text is split into approximately 50-word chunks, which serve as sentences.
* **Data Splits:** The data is split into 80% training, 10% validation, and 10% testing.
* **Tokenization:** The text is tokenized at the word level with a vocabulary size of 20,000.

## Tokenizer and Vocabulary

* **Tokenization:** We use simple space-based splitting to tokenize the text into words.
* **Vocabulary:** The vocabulary contains the 20,000 most frequent words from the corpus, plus special tokens:

  * `<pad>`: Padding token
  * `<unk>`: Unknown token for out-of-vocabulary words
  * `<s>`: Start-of-sentence token
  * `</s>`: End-of-sentence token
* **Vocabulary Artifacts:** Mappings between words and their respective indices are stored in `word_to_idx.json` and `idx_to_word.json`.

## NPLM Model Architecture

The model architecture is a feed-forward neural network designed to predict the next word in a sequence. It operates on a fixed context window of size 5.

* **Embedding Layer:** Uses `nn.Embedding` to map each word to a dense vector of dimension 50.
* **Hidden Layer:** A linear transformation with `nn.Linear`, followed by a `Tanh` activation function. The hidden layer has 100 dimensions.
* **Output Layer:** A final `nn.Linear` layer projects the hidden representation to a size equal to the vocabulary size, producing logits for each word.

## Training

* **Loss Function:** `nn.CrossEntropyLoss`, which is commonly used for classification tasks.
* **Optimizer:** `torch.optim.Adam` with a learning rate of 0.001.
* **Batch Size:** 128.
* **Epochs:** The model is trained for 2 epochs.

## Results and Evaluation

Key metrics are recorded in `results/metrics.json`, including training and evaluation perplexity:

```json
{
  "train_ppl": 224.791,
  "val_ppl": 201.3897,
  "test_ppl": 203.6972,
  "tokenizer": "word",
  "vocab_size_or_merges": 20000,
  "context_size": 5,
  "embedding_dim": 50,
  "hidden_dim": 100,
  "num_epochs": 2,
  "batch_size": 128,
  "learning_rate": 0.001,
  "train_time_sec": "Not captured in this run"
}
```

### Experiment Log

`results/EXPERIMENTS.md` contains detailed notes on hyperparameter tuning, training configurations, and observations during development.

### LLM Collaboration Log

`results/LLM_LOG.md` documents all interactions with the LLM during the project, including code generation, debugging, and guidance, ensuring transparency in the development process.

## Current Status & Future Work

### Completed

* **Data Preprocessing:** The dataset is successfully processed and split into training, validation, and test sets.
* **Model Training:** The model has been trained and evaluated on the data, with perplexity as the key evaluation metric.
* **Results Logging:** Metrics and experiment logs are captured and saved.

### Future Work

* **Model Enhancements:** Investigate advanced techniques like sampled softmax and tied input-output embeddings.
* **Plotting:** Visualize training/validation perplexity curves over training steps.
* **Language Models:** Explore other architectures like Recurrent Neural Networks (RNNs) or Transformer-based models for better performance.

## Conclusion

This project provides a practical implementation of a **Neural Probabilistic Language Model (NPLM)**, with a focus on **perplexity** as the evaluation metric. Future enhancements could include more sophisticated models and optimization techniques to improve the quality of predictions and reduce training time. The code is designed to be extensible for future experiments and improvements.


This `README.md` file provides a thorough explanation of the project, setup instructions, and an outline for future work. It helps others understand the full pipeline from data preprocessing to model evaluation, while also providing an easy way to run and test the project locally or in a cloud environment.
```
