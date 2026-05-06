# Experiment Log

*   **Dataset:** text8
*   **Preprocessing:** Lowercased, tokenized into 50-word chunks.
*   **Tokenizer:** Word-level, vocab size 20000, UNK rate 0.0586.
*   **Model:** Feed-forward NPLM with context size 5, embedding dim 50, hidden dim 100.
*   **Training:** 2 epochs, batch size 128, Adam optimizer with LR 0.001.
*   **Results:**
    *   Training Perplexity: 224.791
    *   Evaluation Perplexity: 202.3083

Add more details here about model sizes tried, what helped/hurt, and any other observations.