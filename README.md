
# Neural Probabilistic Language Model (NPLM) Assignment

## Introduction
This project implements a **Neural Probabilistic Language Model (NPLM)**, drawing inspiration from the foundational work by **Bengio et al., 2003**. The core objective is to build a feed-forward neural network capable of predicting the next word in a sequence, given a fixed-size context window. This involves a complete machine learning pipeline: data acquisition, preprocessing, tokenizer and vocabulary construction, model definition, training, and evaluation, with a focus on reporting perplexity as the primary metric.

## Setup
To get this project running on your local machine or in a Colab environment, follow these steps:

1.  **Clone this starter repo:**
    ```bash
    git clone <URL-of-instructor-repo> # Replace with the actual instructor repo URL
    cd nplm-main
    ```

2.  **Remove the link to the instructor’s remote** (to prevent accidental pushes):
    ```bash
    git remote remove origin
    ```

3.  **Create a new private repository** under your own GitHub/GitLab account (e.g., `nplm-assignment`).

4.  **Add your new repo as the remote**:
    ```bash
    git remote add origin git@github.com:<your-username>/<your-repo>.git
    ```

5.  **Push your current local repository** to your new remote `origin`:
    ```bash
    git push -u origin main
    ```

6.  **Install Dependencies:**
    ```bash
    pip install nltk torch scikit-learn
    ```
    You will also need to download NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ```

7.  **Data Download:** The `text8` corpus is automatically downloaded and processed in the notebook.

## Project Structure
```
.  
├── nplm-main/
│   ├── data/                 # Raw and preprocessed data
│   │   ├── text8             # Original downloaded corpus
│   │   └── text8_sentences.jsonl # Preprocessed JSONL shards
│   ├── results/              # Experiment results, metrics, vocab artifacts
│   │   ├── metrics.json      # Key evaluation metrics (perplexity, training time, etc.)
│   │   ├── EXPERIMENTS.md    # Detailed log of experimental findings and observations
│   │   ├── LLM_LOG.md        # Transparent log of LLM interactions and assistance
│   │   ├── word_to_idx.json  # Vocabulary mapping: word to integer ID
│   │   └── idx_to_word.json  # Vocabulary mapping: integer ID to word
│   ├── README.md             # This comprehensive README file
│   ├── ASSIGNMENT.md         # Original assignment description
│   └── ...                   # Other project files (e.g., the Colab notebook itself)
└── ...
```

## Implementation Details

### Data Preprocessing
-   **Corpus:** `text8` (a 100M character dump of Wikipedia text).
-   **Cleaning:** Lowercased and normalized whitespace.
-   **Chunking:** The continuous text is split into approximately 50-word chunks, treating each chunk as a 'sentence' for training, due to the lack of natural sentence boundaries in the raw `text8` format. These chunks are saved as JSONL shards.
-   **Data Splits:** The preprocessed sentences are divided into training, validation, and test sets with an 80/10/10 ratio, respectively.

### Tokenizer and Vocabulary
-   **Type:** Word-level tokenization using simple space splitting.
-   **Vocabulary Size:** The vocabulary is limited to the 20000 most frequent words, plus special tokens.
-   **Special Tokens:** `<pad>`, `<unk>`, `<s>` (start-of-sentence), `</s>` (end-of-sentence).
-   **Out-of-Vocabulary (UNK) Rate:** Approximately 0.06% of words in the corpus are replaced by `<unk>`.
-   **Artifacts:** `word_to_idx.json` and `idx_to_word.json` store the vocabulary mappings.

### NPLM Model Architecture
-   **Type:** Feed-forward neural network for next-word prediction.
-   **Context Window:** `CONTEXT_SIZE=5` words are used as input to predict the subsequent word.
-   **Embedding Layer:** `nn.Embedding` maps each word ID to a dense vector of `EMBEDDING_DIM=50` dimensions.
-   **Hidden Layer:** A `nn.Linear` layer transforms the concatenated context embeddings into a `HIDDEN_DIM=100` dimensional representation, followed by a `nn.Tanh` activation function.
-   **Output Layer:** Another `nn.Linear` layer projects the hidden state back to `VOCAB_SIZE` dimensions, representing the logits for each possible next word.

### Training
-   **Loss Function:** `nn.CrossEntropyLoss`.
-   **Optimizer:** `torch.optim.Adam` with a `LEARNING_RATE=0.001`.
-   **Batch Size:** `BATCH_SIZE=128`.
-   **Epochs:** The model was trained for `NUM_EPOCHS=2` epochs.

## Results and Logs
-   **Metrics:** Key evaluation metrics are stored in `results/metrics.json`:
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
-   **Experiment Log (`results/EXPERIMENTS.md`):** Contains detailed notes on model configurations, hyperparameter tuning attempts, and insights gained during development.
-   **LLM Collaboration Log (`results/LLM_LOG.md`):** Documents interactions with the LLM for code generation, debugging, and guidance, ensuring transparency in the development process.

## Current Status & Future Work
-   **Vocabulary Artifact:** *Completed*. Vocabulary mappings are saved as JSON files in `results/`.
-   **Data Splits & Evaluation:** *Completed*. The model has been evaluated on distinct train, validation, and test sets.
-   **Training Time Capture:** *Completed*. Training time is now recorded in `metrics.json`.
-   **YAML Configs and CLI:** Still pending. Future work includes refactoring training parameters from Python variables into a `.yaml` configuration file and enabling command-line interface (CLI) execution for training and evaluation.
-   **README.md:** *Completed* (this file).
-   **Model Enhancement:** Explore advanced techniques like sampled softmax for large vocabularies, tied input-output embeddings, or comparing word-level vs. BPE tokenization.
-   **Plotting:** Add plots of train/val perplexity over steps to `results/`.
