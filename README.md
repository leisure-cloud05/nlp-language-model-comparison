# Language Model Benchmarking and Representation Learning

A reproducible NLP project comparing language modeling architectures, word embedding strategies, and learned representations for downstream text classification.

**Topic:** Natural Language Processing for Data Science  
**Framework:** PyTorch  
**Dataset:** WikiText-2 (raw)  
**Status:** Data preprocessing and dataset validation completed; model experiments in progress.

## 1. Project Overview

This project investigates three main questions:

1. How do n-gram, RNN, LSTM, and causal Transformer language models compare in predictive performance, computational efficiency, and text generation quality?
2. How do different input embedding strategies affect language model performance?
3. Do representations learned from language modeling transfer effectively to downstream classification tasks?

The project emphasizes controlled experimental comparisons, reproducibility, and evidence-based interpretation rather than model size alone.

## 2. Planned Experiments



### 2.1 Language Model Comparison

The following language models will be trained on a common English corpus:

- N-gram language model with smoothing or interpolation
- Vanilla RNN language model
- LSTM language model
- Causal Transformer language model

Evaluation will cover next-token cross-entropy or perplexity, training and inference costs, learning curves, and generated text.

#### Trigram Baseline Results

The interpolated Trigram language model has been implemented and evaluated on WikiText-2.

| Metric | Result |
|---|---:|
| Validation Perplexity | 264.90 |
| Test Perplexity | 258.38 |
| Training Time | 1.61 s |
| Serialized Model Size | 24.01 MiB |
| Evaluation Throughput | 934,666 tokens/s |
| Generation Throughput (original Top-k) | 49.3 tokens/s |
| Generation Throughput (UNK-filtered Top-k) | ~46.6 tokens/s |

The model uses recursive interpolated smoothing with document-level BOS padding. Smoothing parameters were selected using the validation split.

Training time measures n-gram counting on preprocessed data. Evaluation throughput measures CPU-based token scoring, while generation throughput includes full-vocabulary probability computation and top-k sampling.

Text generation was evaluated using both standard Top-k sampling and UNK-filtered Top-k sampling. Filtering `<unk>` improved readability but did not eliminate grammatical errors or topic drift. Both decoding strategies use the same trained model, and the decoding changes do not affect the model's test perplexity.

#### Vanilla RNN Baseline Results

A single-layer Vanilla RNN with trainable embeddings was trained from scratch on WikiText-2 using PyTorch and the Apple Silicon MPS backend.

| Metric | Result |
|---|---:|
| Trainable Parameters | 7,798,816 |
| Best Epoch | 4 |
| Validation Perplexity | 176.53 |
| Test Perplexity | 169.98 |
| Training and Validation Time | ~309.4 s |
| Checkpoint Size | 29.75 MiB |
| Evaluation Throughput | 90,159 tokens/s |
| Generation Throughput (original Top-k) | 356.1 tokens/s |
| Generation Throughput (UNK-filtered Top-k) | 354.6 tokens/s |

The model uses 128-dimensional trainable embeddings, a 256-dimensional hidden state, and independent sequence windows of up to 64 tokens. The best checkpoint was selected using validation cross-entropy.

UNK-filtered sampling improved the readability of generated text, although grammatical inconsistencies and topic drift remained. Decoding changes do not affect the model's test perplexity.

Training time includes the measured four training epochs and validation passes, excluding preprocessing and initialization. Generation throughput was measured after an initial MPS warmup. Throughput comparisons with the CPU-based Trigram reflect different implementations and hardware backends.

#### LSTM Baseline Results

A single-layer LSTM with trainable embeddings was trained on WikiText-2 using PyTorch and the Apple Silicon MPS backend, following the same preprocessing and training configuration as the Vanilla RNN.

| Metric | Result |
|---|---:|
| Trainable Parameters | 8,095,264 |
| Best Epoch | 4 |
| Validation Perplexity | 161.03 |
| Test Cross-Entropy | 5.0448 |
| Test Perplexity | 155.21 |
| Training and Validation Time | ~292.4 s |
| Checkpoint Size | 30.88 MiB |
| Evaluation Throughput | 117,314 tokens/s |
| Generation Throughput (original Top-k) | 440.9 tokens/s |
| Generation Throughput (UNK-filtered Top-k) | 430.9 tokens/s |

The model uses 128-dimensional trainable embeddings, a 256-dimensional hidden state, and independent sequence windows of up to 64 tokens. The best checkpoint was selected using validation cross-entropy.

The LSTM achieved lower test perplexity than the Vanilla RNN (155.21 vs. 169.98), although it used more trainable parameters.

Text generation was evaluated using the same three prompts and Top-k sampling settings as the RNN. UNK-filtered sampling improved readability, but the generated text still showed repetition and semantic inconsistencies.

Training time includes four training epochs and validation passes. Generation throughput was measured after MPS warmup. Reported efficiency measurements are specific to the current hardware and implementation.

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

### 2.2 Embedding Ablation

One neural architecture from the language model comparison will be evaluated with three embedding configurations:

- Randomly initialized, trainable embeddings
- Self-trained, frozen word embeddings (e.g., Skip-gram)
- Public pretrained, frozen embeddings (e.g., GloVe)

Embedding dimensions, architecture, training data, and training procedures will be controlled across configurations.

### 2.3 Downstream Representation Analysis

Representations from previously trained language models will be evaluated using a supervised text classification task.

At least two representation methods will be compared using a consistent classifier and evaluation protocol.

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
├── configs/
├── data/
│   └── processed/       # Generated locally, excluded from Git
├── reports/             # Final report
├── results/             # Evaluation results and figures
├── scripts/
│   ├── run_ngram.py
│   ├── evaluate_ngram.py
│   ├── generate_ngram.py
│   ├── smoke_test_rnn.py
│   ├── train_rnn.py
│   ├── evaluate_rnn.py
│   ├── generate_rnn.py
│   ├── smoke_test_lstm.py
│   ├── train_lstm.py
│   ├── evaluate_lstm.py
│   └── generate_lstm.py
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── analyze_data.py
│   │   ├── inspect_headings.py
│   │   ├── preprocess.py
│   │   └── lm_dataset.py
│   ├── models/
│   │   ├── ngram.py
│   │   ├── rnn.py
│   │   ├── rnn_from_scratch.py
│   │   ├── lstm.py
│   │   └── lstm_from_scratch.py
│   ├── embeddings/
│   ├── training/
│   └── evaluation/
└── tests/
    ├── test_lm_dataset.py
    ├── test_ngram.py
    ├── test_rnn.py
    ├── test_rnn_from_scratch.py
    ├── test_lstm.py
    └── test_lstm_from_scratch.py
```

Some directories and experiment modules are reserved for later development.

## 5. Environment

Development environment:

- macOS, Apple Silicon
- Python 3.11
- PyTorch 2.14.0
- Hugging Face Datasets 5.1.0
- Conda environment: `torch_env`
- PyTorch MPS backend available

Core numerical and data-processing libraries are managed through conda-forge. PyTorch is installed through pip inside the Conda environment.

### Current Data Preparation Commands

From the repository root:

```bash
conda activate torch_env

python src/data/prepare_data.py
python src/data/analyze_data.py
python src/data/inspect_headings.py
python src/data/preprocess.py

python tests/test_lm_dataset.py
```

The scripts use Hugging Face Datasets to obtain the corpus. Running the preprocessing script generates the encoded documents and vocabulary under `data/processed/`.

Complete dependency specifications and additional model training commands will be documented as experiments are implemented.

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
- [ ] Train and evaluate causal Transformer
- [ ] Complete embedding ablation experiments
- [ ] Evaluate downstream representations
- [ ] Finalize reproducibility documentation and report



## 7. Reproducibility and Limitations

The vocabulary is built only from the training split, and document boundaries are preserved in next-token sample construction.

The current dataset implementation treats fixed windows independently. Recurrent model states will not automatically carry across windows. Model-specific context limitations and evaluation procedures will be documented before final comparisons.

The preprocessing pipeline discards heading text and uses document-level beginning and end markers. This is a deliberate simplification and may affect language modeling results.

Random seeds, final hyperparameters, dependency versions, model checkpoints, and evaluation settings will be recorded as model experiments are completed.

## 8. Data Sources and Acknowledgments

- WikiText-2 dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- PyTorch: [Official Website](https://pytorch.org/)
- Hugging Face Datasets: [Documentation](https://huggingface.co/docs/datasets/)

This project explores language modeling, word embeddings, and representation learning through reproducible experiments. Standard libraries, documentation, and AI-assisted coding support are used during development. External assistance and implementation references will be documented in the project documentation.

## 9. License

No license has been assigned at this stage. The repository is currently maintained as a private research project.