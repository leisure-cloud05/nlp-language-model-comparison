# Language Model Benchmarking and Representation Learning

An NLP project comparing language modeling architectures, input embedding strategies, and frozen representations for sentiment classification. This README records the actual configurations and results from the completed experiments; model checkpoints and large feature files are generated locally.

**Topic:** Natural Language Processing for Data Science  

**Framework:** PyTorch  

**Dataset:** WikiText-2 (raw)  

**Status:** Parts I–III experiments completed. Final benchmark metrics below were checked against the saved `results/*_final.json`, training histories, and IMDb test results. The written report is maintained separately.

## 1. Project Overview

This project investigates three main questions:

1. How do n-gram, RNN, LSTM, and causal Transformer language models compare in predictive performance, computational efficiency, and text generation quality?
2. How do different input embedding strategies affect language model performance?
3. Do representations learned from language modeling transfer effectively to downstream classification tasks?

The project emphasizes controlled experimental comparisons, reproducibility, and evidence-based interpretation rather than model size alone.

## 2. Experiments and Results

### 2.1 Language Model Comparison

The following language models were trained and evaluated on a common English corpus:

- N-gram language model with smoothing or interpolation
- Vanilla RNN language model
- LSTM language model
- Causal Transformer language model

Evaluation covers next-token cross-entropy and perplexity, training and inference costs, learning curves, and generated text.

#### Trigram Baseline Results

The interpolated Trigram language model has been implemented and evaluated on WikiText-2.

| Metric                                     | Result           |
| ------------------------------------------ | ---------------- |
| Validation Perplexity                      | 264.90           |
| Test Perplexity                            | 258.38           |
| Training Time                              | 1.56 s           |
| Serialized Model Size                      | 24.01 MiB        |
| Evaluation Throughput                      | 844,996 tokens/s |
| Generation Throughput (original Top-k)     | 49.3 tokens/s    |
| Generation Throughput (UNK-filtered Top-k) | ~46.6 tokens/s   |

The model uses recursive interpolated smoothing with document-level BOS padding. Smoothing parameters were selected using the validation split.

Training time measures n-gram counting on preprocessed data. Evaluation throughput measures CPU-based token scoring, while generation throughput includes full-vocabulary probability computation and top-k sampling.

Text generation was evaluated using both standard Top-k sampling and UNK-filtered Top-k sampling. Filtering `<unk>` improved readability but did not eliminate grammatical errors or topic drift. Both decoding strategies use the same trained model, and the decoding changes do not affect the model's test perplexity.

#### Vanilla RNN Baseline Results

A single-layer Vanilla RNN with trainable embeddings was trained from scratch on WikiText-2 using PyTorch and the Apple Silicon MPS backend.

| Metric                                     | Result                       |
| ------------------------------------------ | ---------------------------- |
| Trainable Parameters                       | 7,798,816                    |
| Best Epoch                                 | 5 (early stopped at epoch 8) |
| Validation Perplexity                      | 175.77                       |
| Test Cross-Entropy                         | 5.1323                       |
| Test Perplexity                            | 169.41                       |
| Training and Validation Time               | 633.1 s                      |
| Checkpoint Size                            | 29.75 MiB                    |
| Evaluation Throughput                      | 99,931 tokens/s              |
| Generation Throughput (original Top-k)     | 347.3 tokens/s               |
| Generation Throughput (UNK-filtered Top-k) | 365.8 tokens/s               |

The model uses 128-dimensional trainable embeddings, a 256-dimensional hidden state, and independent sequence windows of up to 64 tokens. The best checkpoint was selected using validation cross-entropy. The model was trained with a maximum of 15 epochs and early stopping (patience 3, minimum improvement 0.001 in validation cross-entropy).

Validation performance peaked at epoch 5 and deteriorated thereafter even as training loss continued to decrease, consistent with overfitting. UNK-filtered sampling improved the readability of generated text, although grammatical inconsistencies and topic drift remained. Decoding changes do not affect the model's test perplexity.

Training time includes eight measured training epochs and validation passes, excluding preprocessing and initialization. Generation throughput was measured after an initial MPS warmup. Throughput comparisons with the CPU-based Trigram reflect different implementations and hardware backends.

#### LSTM Baseline Results

A single-layer LSTM with trainable embeddings was trained on WikiText-2 using PyTorch and the Apple Silicon MPS backend, following the same preprocessing and core optimization settings as the Vanilla RNN.

| Metric                                     | Result                       |
| ------------------------------------------ | ---------------------------- |
| Trainable Parameters                       | 8,095,264                    |
| Best Epoch                                 | 6 (early stopped at epoch 9) |
| Validation Perplexity                      | 155.64                       |
| Test Cross-Entropy                         | 5.0117                       |
| Test Perplexity                            | 150.15                       |
| Training and Validation Time               | 690.0 s                      |
| Checkpoint Size                            | 30.88 MiB                    |
| Evaluation Throughput                      | 119,519 tokens/s             |
| Generation Throughput (original Top-k)     | 439.6 tokens/s               |
| Generation Throughput (UNK-filtered Top-k) | 440.9 tokens/s               |

The model uses 128-dimensional trainable embeddings, a 256-dimensional hidden state, and independent sequence windows of up to 64 tokens. The best checkpoint was selected using validation cross-entropy. Training used a maximum of 15 epochs with early stopping (patience 3, minimum improvement 0.001 in validation cross-entropy).

The LSTM achieved lower test perplexity than the Vanilla RNN (150.15 vs. 169.41), although it used more trainable parameters. Validation performance was best at epoch 6 and worsened during the following three epochs.

Text generation used the same three prompts and Top-k sampling settings as the RNN. UNK-filtered sampling improved readability, but generated text still showed repetition and semantic inconsistencies.

Training time includes nine measured training epochs and validation passes, excluding preprocessing and initialization. Generation throughput was measured after MPS warmup. Reported efficiency measurements are specific to the current hardware and implementation.

#### Causal Transformer Baseline Results

A two-layer causal Transformer language model was implemented with manually constructed multi-head self-attention, causal masking, residual connections, and feed-forward blocks. Training and evaluation were performed on WikiText-2 using PyTorch and the Apple Silicon MPS backend.

| Metric                                     | Result                               |
| ------------------------------------------ | ------------------------------------ |
| Trainable Parameters                       | 5,544,992                            |
| Best Epoch                                 | 15 (maximum training budget reached) |
| Validation Perplexity                      | 138.37                               |
| Test Cross-Entropy                         | 4.8994                               |
| Test Perplexity                            | 134.21                               |
| Training and Validation Time               | 798.9 s                              |
| Checkpoint Size                            | 21.16 MiB                            |
| Evaluation Throughput                      | 153,416 tokens/s                     |
| Generation Throughput (original Top-k)     | 203.3 tokens/s                       |
| Generation Throughput (UNK-filtered Top-k) | 202.8 tokens/s                       |

The Transformer uses 128-dimensional token and learned positional embeddings, two Transformer blocks, four attention heads, a feed-forward dimension of 512, and dropout of 0.1. Independent sequence windows of up to 64 tokens were used. The best checkpoint was selected using validation cross-entropy, under a maximum budget of 15 epochs and early stopping (patience 3, minimum improvement 0.001 in validation cross-entropy).

Causal multi-head self-attention was implemented directly using PyTorch tensor operations, without relying on `nn.MultiheadAttention` or `nn.TransformerEncoder`. Correctness tests verified forward and backward computation, causal next-token behavior, and masking of future positions.

The Transformer achieved the lowest test perplexity among the four evaluated models (134.21), below the LSTM (150.15), while using fewer trainable parameters. Validation cross-entropy was lowest at epoch 15, so convergence beyond the maximum training budget has not been established. These findings are specific to the tested configurations and do not establish general architectural superiority.

Text generation used the same three prompts, top-k value (20), temperature (1.0), random seed (42), and maximum generation length (40 tokens) as the recurrent models. UNK-filtered sampling improved readability, although repetition, grammatical errors, and semantic inconsistencies remained.

Training time includes the measured 15 training epochs and validation passes, excluding preprocessing and initialization. Generation throughput includes repeated context inference and top-k sampling without KV caching.

#### From-Scratch Recurrent Model Implementations

In addition to the benchmark models, this project includes implementations of recurrent architectures using PyTorch tensor operations.

**Vanilla RNN (completed)**

- Implemented the recurrent update manually using learnable input-to-hidden and hidden-to-hidden parameters.
- Used the tanh activation and explicitly iterated over sequence positions.
- Verified numerical consistency with PyTorch's `nn.RNN` after copying identical weights and biases.
- Confirmed that forward outputs, final hidden states, input gradients, hidden-state gradients, and parameter gradients match within numerical tolerance.
- Verified the default zero-initialized hidden state.

The from-scratch implementation is intended for algorithm understanding and correctness verification. The main language-model benchmark continues to use PyTorch's optimized `nn.RNN` for training and efficiency measurements.

**LSTM (completed)**

- Implemented the input, forget, candidate, and output gates manually using PyTorch tensor operations.
- Explicitly computed hidden-state and cell-state updates across sequence positions.
- Verified numerical consistency with PyTorch's `nn.LSTM` after copying identical parameters.
- Confirmed matching forward outputs, final hidden and cell states, input and initial-state gradients, and parameter gradients within numerical tolerance.
- Verified the default zero-initialized hidden and cell states.

The main LSTM benchmark uses PyTorch's optimized `nn.LSTM` for training and efficiency measurements. The manual implementation is used for algorithm understanding and correctness verification.

**Implementation files:**

- `src/models/rnn_from_scratch.py`
- `tests/test_rnn_from_scratch.py`
- `src/models/lstm_from_scratch.py`
- `tests/test_lstm_from_scratch.py`

The manual implementations are separate from the benchmark models and do not change the reported training or evaluation results.

#### Training and Inference Efficiency

The table below uses one consistent definition of each time metric rather than mixing training-only and training-plus-validation totals.

| Model | Completed epochs (best) | Training only (s) | Validation (s) | Training + validation (s) | Test targets/s |
|---|---:|---:|---:|---:|---:|
| RNN | 8 (5) | 615.5 | 17.6 | 633.1 | 99,931 |
| LSTM | 9 (6) | 672.5 | 17.5 | 690.0 | 119,519 |
| Transformer | 15 (15) | 778.8 | 20.2 | 798.9 | 153,416 |

The same test scoring workload contains **238,352 target tokens** for every Part I model. Different effective epoch counts mean the total running times are not equal-training-budget comparisons. Transformer scoring can evaluate tokens in parallel, while generation is autoregressive; the present Transformer generator does not cache past K/V projections.

#### Language Model Comparison Summary

All four models were evaluated on the same processed WikiText-2 splits, a vocabulary of 20,000 entries (including four special tokens), and the same 238,352 valid test prediction targets.

| Model                | Validation PPL | Test PPL   | Test CE    | Model Size    |
| -------------------- | -------------- | ---------- | ---------- | ------------- |
| Interpolated Trigram | 264.90         | 258.38     | 5.5544     | 24.01 MiB     |
| Vanilla RNN          | 175.77         | 169.41     | 5.1323     | 29.75 MiB     |
| LSTM                 | 155.64         | 150.15     | 5.0117     | 30.88 MiB     |
| Causal Transformer   | **138.37**     | **134.21** | **4.8994** | **21.16 MiB** |

**Extended training and checkpoint selection.** The three neural models were trained from scratch using Adam (learning rate 0.001), batch size 16, gradient clipping at 1.0, and random seed 42, with a maximum of 15 epochs. Early stopping used patience 3 and minimum improvement 0.001 in validation cross-entropy. The checkpoint with the lowest observed validation cross-entropy was retained for test evaluation, even when an improvement was smaller than the early-stopping threshold.

The Vanilla RNN stopped after epoch 8 (best epoch 5); the LSTM stopped after epoch 9 (best epoch 6); and the Transformer ran all 15 epochs (best epoch 15). The Transformer's validation performance was still improving at the training-budget limit, so convergence is not established. Measured training plus validation times were 633.1 s, 690.0 s, and 798.9 s, respectively; these exclude preprocessing and initialization.

Neural language models achieved substantially lower test perplexity than the Trigram baseline. The LSTM outperformed the Vanilla RNN, and the Transformer achieved the lowest perplexity. Longer training substantially improved Transformer test perplexity, but qualitative samples from all three neural models still showed repetition, grammatical inconsistencies, and topic drift. UNK-filtered decoding improved readability without changing test perplexity.

These comparisons control the dataset, vocabulary, target coverage, and core evaluation procedure. Model parameter counts, architectures, context mechanisms, effective training durations, and implementation backends differ; the results represent comparisons of the evaluated systems, not strictly parameter-matched architectural ablations. Generation observations are based on three fixed prompts and are not a definitive quality ranking.

### 2.2 Embedding Ablation — Completed

Part II compares three 100-dimensional embedding strategies in a shared single-layer LSTM language model (hidden size 256):

- **Random Trainable:** randomly initialized embeddings, updated during language model training.
- **Self-trained Skip-gram Frozen:** Gensim Word2Vec Skip-gram trained only on the WikiText-2 training split, then frozen in the LSTM.
- **Public Pretrained GloVe Frozen:** `glove-wiki-gigaword-100` mapped into the project vocabulary and frozen in the LSTM.

The vocabulary (20,000 tokens), 64-token context window, train/validation/test splits, optimizer (Adam, learning rate 0.001), batch size (16), gradient clipping (1.0), random seed (42), validation-based checkpoint selection, and maximum training budget (15 epochs with early stopping, patience 3 and min_delta 0.001) are consistent across the three configurations. Embeddings are 100-dimensional in Part II versus 128-dimensional in Part I's LSTM benchmark.

#### Embedding Construction and Correctness

The self-trained Skip-gram uses embedding dimension 100, window size 5, five negative samples, minimum count 1, five epochs, one worker, and seed 42. All 19,996 ordinary vocabulary entries had exact Skip-gram vector matches (100% coverage).

GloVe vector lookup first uses an exact match, then a lowercase fallback, and finally a seeded random vector for missing entries; PAD remains zero. Of the 19,996 ordinary vocabulary entries, 12,954 matched exactly, 6,784 matched by lowercase fallback, and 258 were missing, for **98.71% coverage**.

A separate PyTorch implementation of Skip-gram with Negative Sampling (SGNS) verifies the objective and gradients, including numerical finite-difference tests and negative-sampling checks. The official Skip-gram embeddings used for the LSTM experiments are trained with Gensim. The embedding-aware LSTM also passed forward-shape, trainability, freezing, and invalid-input tests.

#### Final Embedding Ablation Results

All three models were selected by minimum validation cross-entropy and then evaluated on the same **238,352 test prediction targets**.

| Metric | Random Trainable | Skip-gram Frozen | GloVe Frozen |
|---|---:|---:|---:|
| Trainable parameters | 7,506,592 | 5,506,592 | 5,506,592 |
| Best epoch | 6 | 8 | 8 |
| Epochs completed | 9 | 11 | 11 |
| Validation perplexity | 158.68 | **140.74** | 141.05 |
| Test cross-entropy | 5.0341 | **4.9217** | 4.9224 |
| Test perplexity | 153.57 | **137.24** | 137.34 |
| Evaluation throughput (tokens/s) | 127,376 | 123,852 | 121,717 |

Both frozen pretrained configurations achieved lower test perplexity than the random trainable baseline: approximately **10.63%** lower for Skip-gram and **10.57%** lower for GloVe. Skip-gram and GloVe differ by just 0.10 test perplexity points; one seed does not establish a statistically reliable difference.

#### Training Cost and Vocabulary Coverage

| Setting | Epochs (best) | LM training only (s) | LM validation (s) | LM training + validation (s) | Extra embedding pretraining |
|---|---:|---:|---:|---:|---:|
| Random trainable | 9 (6) | 640.4 | 16.9 | 657.4 | None |
| Skip-gram frozen | 11 (8) | 726.0 | 20.2 | 746.2 | 23.28 s (gensim) |
| GloVe frozen | 11 (8) | 782.6 | 22.6 | 805.2 | Public model (not locally trained) |

GloVe lookup is measured over **19,996 ordinary vocabulary entries**: 12,954 exact matches, 6,784 lowercase matches, and 258 unmatched entries (98.71% coverage). The unmatched entries use seeded random vectors; the special-token handling is separate. Skip-gram has 19,996 exact matches (100% coverage). The three setups are a comparison of **complete embedding choices**, because the randomly initialized embedding matrix is trainable while both pretrained matrices are frozen.

#### Text Generation

Text generation reuses Part I's LSTM generation procedure with three fixed prompts (`The history of`, `In the city of`, and `The first`), random seed 42, temperature 1.0, top-k 20, and a maximum of 40 new tokens. Both original Top-k sampling and UNK-filtered Top-k sampling were evaluated, yielding 18 samples in total.

| Generation throughput (tokens/s) | Random Trainable | Skip-gram Frozen | GloVe Frozen |
|---|---:|---:|---:|
| Original Top-k | 279.8 | 648.6 | 731.6 |
| UNK-filtered Top-k | 377.7 | 740.2 | 746.4 |

Filtering `<unk>` improves readability, but generated text can still contain repetition, grammatical errors, and topic drift. Generation speeds are implementation- and run-dependent and should not be attributed solely to embedding freezing; noticeable timing variability occurred, especially in the first Random prompt.

**Interpretation and limitations:** This is a comparison of complete embedding strategies, **not** an isolated causal test of pretraining. Random embeddings are trainable while Skip-gram/GloVe embeddings are frozen, producing a difference of two million trainable parameters. The training corpora also differ for self-trained Skip-gram and public GloVe. The reported results are from a single random seed and limited qualitative generation samples.

### 2.3 Downstream Representation Analysis — Completed

**Task.** Binary sentiment classification on IMDb using frozen representations extracted from the LSTM and causal Transformer trained in Part I. The official IMDb training split (25,000 reviews) was divided into 20,000 training and 5,000 validation examples using a stratified split with random seed 42. The official test split (25,000 reviews) was reserved for final evaluation.

**Representation extraction.** Reviews were cleaned and tokenized using the Part I tokenizer and vocabulary. Each review was truncated to at most 768 tokens and processed in windows of 64 tokens. The pretrained language-model weights remained frozen throughout downstream experiments. We evaluated final hidden-state representations, mean pooling, token-weighted mean pooling, mean-plus-max pooling, and overlapping windows with strides of 32 and 16 tokens.

**Classification.** Extracted features were standardized using a `StandardScaler` fitted on training data only. Logistic Regression classifiers were trained on the frozen features, with regularization strength selected from `C ∈ {0.01, 0.1, 1, 10, 100}` using validation macro F1. Additional small MLP experiments were conducted as exploratory classifier comparisons; they were not included in the final saved-probe test benchmark.

#### Final IMDb Classification Results

| Model and Representation | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| LSTM — Final state | 58.06% | 57.83% | 57.83% |
| LSTM — Mean pooling | 71.00% | 70.83% | 70.83% |
| LSTM — Token-weighted mean | 71.34% | **71.82%** | 71.82% |
| LSTM — Mean + Max | 70.30% | 70.49% | 70.49% |
| LSTM — Overlap, stride 32 | **72.16%** | 71.56% | 71.56% |
| LSTM — Overlap, stride 16 | 71.32% | 71.34% | 71.33% |
| Transformer — Mean pooling | 71.16% | 69.88% | 69.87% |
| Transformer — Token-weighted mean | 70.48% | 70.42% | 70.41% |
| Transformer — Mean + Max | 70.82% | 69.74% | 69.73% |
| Transformer — Overlap, stride 32 | **71.20%** | 70.60% | 70.59% |
| Transformer — Overlap, stride 16 | 70.90% | 70.77% | 70.76% |

All test results were evaluated on the same 25,000 official IMDb test examples. Saved predictions were independently checked against the reported accuracy, macro F1, and confusion matrices.

**Findings.** Mean pooling substantially improved LSTM downstream accuracy over using the final hidden state. A moderate window overlap (stride 32) improved validation accuracy for the LSTM, but increasing overlap to stride 16 did not provide further gains. Although the Transformer achieved lower language-model perplexity in Part I, this did not consistently translate into better frozen-representation sentiment classification.

The LSTM stride-32 configuration was selected by validation performance, whereas token-weighted mean pooling achieved the highest numerical test accuracy. Test results were not used to revise the model-selection decision.

#### Selected-model confusion matrices

Rows give the true IMDb sentiment and columns the predicted sentiment (the official test set has 12,500 reviews per class).

| Frozen feature extractor | True negative → negative | True negative → positive | True positive → negative | True positive → positive |
|---|---:|---:|---:|---:|
| LSTM, stride 32 | 9,009 | 3,491 | 3,619 | 8,881 |
| Transformer, stride 32 | 9,045 | 3,455 | 3,895 | 8,605 |

Thus Transformer correctly identifies **36 more negative reviews**, but **276 fewer positive reviews**, than LSTM. Across the same 25,000 test examples, both are correct on 14,737; both are incorrect on 4,197; only LSTM is correct on 3,153; and only Transformer is correct on 2,913. These paired counts sum to 25,000 and match the final accuracy difference.

#### Error Analysis

We compared the stride-32 LSTM and Transformer predictions on the official IMDb test set. Both models were correct on 14,737 reviews and wrong on 4,197. The LSTM alone was correct on 3,153 reviews, while the Transformer alone was correct on 2,913.

Qualitative inspection revealed challenging cases involving explicit sentiment words, mixed positive and negative expressions, and negation. These observations are illustrative rather than causal evidence of specific model failures.

**Limitations.** The language models were pretrained on WikiText-2 rather than IMDb and were not fine-tuned for sentiment classification. Reviews were truncated at 768 tokens, and each 64-token window was encoded independently. Multiple representation variants were explored on the same validation set; consequently, small performance differences should be interpreted cautiously.

## 3. Dataset and Preprocessing

**Corpus:** `Salesforce/wikitext`  

**Configuration:** `wikitext-2-raw-v1`

The official training, validation, and test splits are retained.

### Preprocessing Decisions

- Word-level tokenization using a shared regular expression
- Original capitalization retained
- WikiText formatting markers normalized
- Vocabulary built exclusively from training data
- Vocabulary size: 20,000
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- One `<bos>` token prepended to each document
- One `<eos>` token appended to each document
- Document-internal punctuation preserved
- Level-1 headings used to identify document boundaries
- Higher-level headings excluded from the language-model corpus
- Documents encoded and stored independently

The initial fixed-window next-token dataset uses a maximum of 64 target positions per sample. Padding labels are set to `-100` for exclusion from the loss.

### Processed Dataset Statistics

| Statistic                      | Train     | Validation | Test    |
| ------------------------------ | --------- | ---------- | ------- |
| Documents                      | 629       | 60         | 62      |
| Token IDs                      | 2,032,891 | 212,129    | 238,414 |
| OOV rate                       | 5.07%     | 7.57%      | 8.10%   |
| Next-token samples (length 64) | 32,085    | 3,347      | 3,755   |
| Prediction targets             | 2,032,262 | 212,069    | 238,352 |

OOV rates exclude special tokens. Dataset statistics reflect the current preprocessing implementation.

## 4. Repository Structure

```text
nlp-language-model-comparison/
├── README.md
├── .gitignore
├── reports/
│   └── nlp-comparison-report.pdf
├── results/
│   ├── ngram_*.json
│   ├── rnn_*.json
│   ├── lstm_*.json
│   ├── transformer_*.json
│   ├── embedding_ablation_*.json
│   ├── skipgram_training.json
│   ├── imdb_*_probe.json
│   ├── imdb_*_test.json
│   ├── figures/
│   ├── samples/
│   └── tables/
├── scripts/
│   ├── run_ngram.py
│   ├── evaluate_ngram.py
│   ├── generate_ngram.py
│   ├── train_rnn.py
│   ├── evaluate_rnn.py
│   ├── generate_rnn.py
│   ├── train_lstm.py
│   ├── evaluate_lstm.py
│   ├── generate_lstm.py
│   ├── train_transformer.py
│   ├── evaluate_transformer.py
│   ├── generate_transformer.py
│   ├── train_word2vec.py
│   ├── train_embedding_ablation.py
│   ├── evaluate_embedding_ablation.py
│   ├── generate_embedding_ablation.py
│   ├── prepare_imdb.py
│   ├── extract_imdb_features.py
│   ├── extract_imdb_overlap_features.py
│   ├── train_imdb_probe.py
│   ├── train_imdb_mlp.py
│   ├── evaluate_imdb_test.py
│   ├── experiment_imdb_overlap.py
│   └── smoke_test_*.py
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── analyze_data.py
│   │   ├── inspect_headings.py
│   │   ├── preprocess.py
│   │   ├── lm_dataset.py
│   │   └── imdb_preprocess.py
│   ├── models/
│   │   ├── ngram.py
│   │   ├── rnn.py
│   │   ├── rnn_from_scratch.py
│   │   ├── lstm.py
│   │   ├── lstm_from_scratch.py
│   │   ├── transformer.py
│   │   ├── embedding_lstm.py
│   │   ├── checkpoint_loader.py
│   │   ├── representation_extractor.py
│   │   └── imdb_mlp.py
│   └── embeddings/
│       ├── word2vec.py
│       ├── skipgram_from_scratch.py
│       └── embedding_utils.py
└── tests/
    ├── test_lm_dataset.py
    ├── test_ngram.py
    ├── test_rnn.py
    ├── test_rnn_from_scratch.py
    ├── test_lstm.py
    ├── test_lstm_from_scratch.py
    ├── test_transformer.py
    ├── test_embedding_lstm.py
    └── test_skipgram_from_scratch.py
```

### Local Files and Reproducibility

Large datasets, processed features, model checkpoints, and private development notes are stored locally and excluded from Git.

The repository includes the saved experiment metrics in `results/`. For Part III, `lstm_overlap` and `transformer_overlap` refer to stride-32 features, while `lstm_stride16` and `transformer_stride16` refer to stride-16 features.

## 5. Environment

Experiments were conducted on macOS with Apple Silicon.

- Python 3.11
- PyTorch 2.14.0 (MPS backend)
- Hugging Face Datasets 5.1.0
- Conda environment: `torch_env`

Other dependencies include NumPy, scikit-learn, gensim,
and matplotlib.

### Data Preparation

Run the following commands from the repository root:

```bash
conda activate torch_env

python src/data/prepare_data.py
python src/data/analyze_data.py
python src/data/inspect_headings.py
python src/data/preprocess.py
python tests/test_lm_dataset.py
```

The preprocessing pipeline builds the vocabulary from the WikiText-2 training split and saves the processed datasets under `data/processed/`

### Trigram Experiment Commands

Run the correctness tests:

```bash

python tests/test_ngram.py

```

Run validation hyperparameter search:

```bash

python scripts/run_ngram.py

```

Evaluate the selected model and save or load the local checkpoint:

```bash

python scripts/evaluate_ngram.py

```

Generate text using the saved checkpoint:

```bash

python scripts/generate_ngram.py

```

Model checkpoints are stored locally under `checkpoints/` and excluded from Git.

**Note:** The current validation script only runs its configured candidate set, rather than automatically reproducing all stages of the original hyperparameter search.

### Vanilla RNN Experiment Commands

Run model correctness tests:

```bash

python tests/test_rnn.py

```

Run the small-scale training test:

```bash

python scripts/smoke_test_rnn.py

```

Train the RNN and select the best checkpoint using validation cross-entropy:

```bash

python scripts/train_rnn.py

```

Evaluate the saved checkpoint on the test set:

```bash

python scripts/evaluate_rnn.py

```

Generate text using the saved checkpoint:

```bash

python scripts/generate_rnn.py

```

Set `EXCLUDE_UNK = False` in `scripts/generate_rnn.py` for original Top-k sampling, or `EXCLUDE_UNK = True` for UNK-filtered Top-k sampling.

Generated results are saved under `results/`. Model checkpoints are saved locally under `checkpoints/` and excluded from Git.

### LSTM Experiment Commands

Run the model correctness tests:

```bash

python tests/test_lstm.py

```

Run the small-scale training test:

```bash

python scripts/smoke_test_lstm.py

```

Train the LSTM and select the best checkpoint using validation cross-entropy:

```bash

python scripts/train_lstm.py

```

Evaluate the saved checkpoint on the test set:

```bash

python scripts/evaluate_lstm.py

```

Generate text using the saved checkpoint:

```bash

python scripts/generate_lstm.py

```

Set `EXCLUDE_UNK = False` in `scripts/generate_lstm.py` for original Top-k sampling, or `EXCLUDE_UNK = True` for UNK-filtered Top-k sampling.

The two generation modes save results to separate JSON files under `results/`. Model checkpoints are stored locally under `checkpoints/` and excluded from Git.

### From-Scratch Recurrent Model Verification

Verify the manually implemented recurrent architectures against PyTorch reference implementations:

```bash

python tests/test_rnn_from_scratch.py

python tests/test_lstm_from_scratch.py

```

These tests compare forward outputs, recurrent states, and backward gradients without retraining the benchmark models.

### Causal Transformer Experiment Commands

Run model correctness tests, including causal masking and gradient verification:

```bash

python tests/test_transformer.py

```

Run the small-scale MPS training test:

```bash

python scripts/smoke_test_transformer.py

```

Train the Transformer and select the best checkpoint using validation cross-entropy:

```bash

python scripts/train_transformer.py

```

Evaluate the saved best checkpoint on the test split:

```bash

python scripts/evaluate_transformer.py

```

Generate text using the saved checkpoint:

```bash

python scripts/generate_transformer.py

```

Set `EXCLUDE_UNK = False` in `scripts/generate_transformer.py` for original Top-k sampling, or `EXCLUDE_UNK = True` for UNK-filtered Top-k sampling.

Generation results are saved separately under `results/`. The best model checkpoint is stored locally under `checkpoints/` and excluded from Git.

The Transformer implements causal multi-head self-attention manually using PyTorch tensor operations. The generation procedure recomputes the most recent context without KV caching.

### Part II — Embedding Ablation Commands

Run the handwritten SGNS and LSTM embedding tests:

```bash
python -m tests.test_skipgram_from_scratch
python -m tests.test_embedding_lstm
```

Train the self-trained Skip-gram word vectors using the training split:

```bash
python -m scripts.train_word2vec
```

Train all three embedding strategies separately (do not overwrite the Part I LSTM checkpoint):

```bash
python scripts/train_embedding_ablation.py --embedding random
python scripts/train_embedding_ablation.py --embedding skipgram
python scripts/train_embedding_ablation.py --embedding glove
```

Evaluate the saved best checkpoints on the common test split:

```bash
python -u scripts/evaluate_embedding_ablation.py --embedding all
```

Generate the same prompts and Top-k samples used in Part I:

```bash
python -u scripts/generate_embedding_ablation.py
```

Set `EXCLUDE_UNK = False` in `scripts/generate_embedding_ablation.py` for original Top-k sampling, or `EXCLUDE_UNK = True` for UNK-filtered Top-k sampling; the modes write separate JSON files. The three training histories, test summaries, and two generation-mode results per model are saved under `results/`. Best checkpoints and the locally trained Gensim model are stored under `checkpoints/` and are excluded from Git.

### Part III — IMDb Downstream Classification Commands

Part III uses frozen LSTM and Transformer checkpoints from Part I. All experiments use the same IMDb train/validation split and evaluate classification using accuracy and macro F1.

**1. Prepare IMDb data and split indices**

```bash
python -m scripts.prepare_imdb
```

**2. Extract frozen language-model features**

Examples for the original non-overlapping representations:

```bash
python -m scripts.extract_imdb_features --model lstm_mean --split train
python -m scripts.extract_imdb_features --model lstm_mean --split validation

python -m scripts.extract_imdb_features --model transformer --split train
python -m scripts.extract_imdb_features --model transformer --split validation
```

For overlapping representations, extract the training and validation features using the same stride:

```bash
python -m scripts.extract_imdb_overlap_features --model lstm --split train --stride 32
python -m scripts.extract_imdb_overlap_features --model lstm --split validation --stride 32
```

For overlapping-window representations, the window size is 64 tokens.

- `lstm_overlap` and `transformer_overlap` use stride 32.
- `lstm_stride16` and `transformer_stride16` use stride 16.

These identifiers correspond to four configurations within the 11 evaluated IMDb representations.

**3. Train linear probes**

```bash
python -m scripts.train_imdb_probe --model lstm_mean
python -m scripts.train_imdb_probe --model transformer
```

The same script supports the other saved representation variants, including weighted pooling, mean-plus-max pooling, and stride-specific overlapping windows.

**4. Evaluate on the official IMDb test set**

```bash
python -m scripts.evaluate_imdb_test --model lstm_mean
python -m scripts.evaluate_imdb_test --model lstm_overlap
python -m scripts.evaluate_imdb_test --model transformer
python -m scripts.evaluate_imdb_test --model transformer_overlap
```

The evaluation script loads the corresponding saved classifier without refitting it, extracts or reuses cached test features, and saves accuracy, macro F1, confusion matrices, and prediction records.

**Reproducibility notes:** Checkpoints and processed feature matrices are generated locally and excluded from Git. Training and feature-extraction scripts protect existing outputs from accidental overwriting. To reproduce an experiment from scratch, follow the data-preparation, feature-extraction, probe-training, and test-evaluation order using the appropriate representation name.

## 6. Current Progress

- [x] Initialize GitHub repository
- [x] Configure Python and PyTorch environment
- [x] Load WikiText-2
- [x] Analyze vocabulary coverage
- [x] Implement vocabulary construction and preprocessing
- [x] Correct document boundary detection
- [x] Implement next-token dataset and padding
- [x] Validate dataset samples and target coverage
- [x] Train and evaluate n-gram model
- [x] Train and evaluate RNN
- [x] Train and evaluate LSTM
- [x] Train and evaluate causal Transformer
- [x] Complete embedding ablation experiments
- [x] Evaluate downstream representations
- [x] Record and audit final experiment metrics
- [x] Finalize the submitted report


## 7. Reproducibility and Limitations

### Reproducibility

All three project parts use documented preprocessing, model configurations, evaluation procedures, and fixed random seeds where applicable. The WikiText-2 vocabulary is constructed from training data only, and document boundaries are preserved when generating next-token samples.

Part III uses the official IMDb dataset. The original training split is divided into 20,000 training and 5,000 validation reviews using stratified sampling with random seed 42. The official 25,000-review test split is reserved for final evaluation.

The LSTM and Transformer checkpoints trained in Part I remain frozen during downstream representation extraction. The feature standardizer is fitted exclusively on the IMDb training features. Logistic Regression regularization is selected using validation macro F1, and the official test set is not used for hyperparameter tuning.

Final test predictions and metrics are saved for reproducibility. Accuracy, macro F1, confusion matrices, and sample alignment were independently checked across all 11 probe configurations.

Large processed datasets, feature matrices, and trained checkpoints are excluded from Git. Reproducing the experiments requires obtaining the datasets and running the documented preprocessing, training, feature extraction, and evaluation commands.

### Limitations

**Language modeling.** The models use a word-level vocabulary of 20,000 tokens and a maximum context length of 64 tokens. Fixed windows are processed independently, and recurrent hidden states are reset between windows. Heading text is discarded, and document-level beginning and end markers are used. These design choices may limit language modeling and generation quality.

**Embedding ablation.** The Part II experiments use a single random seed and do not fully isolate embedding initialization from whether embeddings are frozen or trainable.

**Downstream classification.** The language models are pretrained on WikiText-2 rather than sentiment-specific data and are not fine-tuned on IMDb. Reviews are truncated to 768 tokens, and individual windows are encoded independently. The downstream experiments compare frozen representations rather than end-to-end sentiment fine-tuning.

Multiple representation extraction strategies were explored using the same validation set. Consequently, small differences between validation results may reflect model-selection variability. Test results are reported without further hyperparameter tuning.

**Compute environment.** Reported training and inference times reflect the Apple Silicon MPS environment and may differ on other hardware.

## 8. Data Sources and Acknowledgments

- WikiText-2 dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- PyTorch: [Official Website](https://pytorch.org/)
- Hugging Face Datasets: [Documentation](https://huggingface.co/docs/datasets/)

This project uses PyTorch, Hugging Face Datasets, gensim, scikit-learn, and related numerical libraries. ChatGPT was used for development assistance, explanations, debugging, and report editing. Experiments were run locally, and the final numeric metrics in this README were verified against saved results. Public GloVe embeddings are used only in Part II.


## 8. License

No license has been assigned to this repository. All rights are reserved unless a license is added.
