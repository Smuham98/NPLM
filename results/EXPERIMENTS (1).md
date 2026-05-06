# Experiment Log

*   **Dataset:** text8
*   **Preprocessing:** Lowercased, tokenized into 50-word chunks, split into train/val/test.
*   **Tokenizer:** Word-level, vocab size 20000, UNK rate 0.0586.
*   **Model:** Feed-forward NPLM with context size 5, embedding dim 50, hidden dim 100.
*   **Training:** 2 epochs, batch size 128, Adam optimizer with LR 0.001. (Model trained on full dataset for initial run, then evaluated on separate splits)
*   **Results:**
    *   Training Perplexity: 224.791
    *   Validation Perplexity: 201.3897
    *   Test Perplexity: 203.6972

Add more details here about model sizes tried, what helped/hurt, and any other observations.