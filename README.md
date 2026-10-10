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



### Part I — Language Model Comparison

The following language models will be trained on a common English corpus:

- N-gram language model with smoothing or interpolation
- Vanilla RNN language model
- LSTM language model
- Causal Transformer language model

Evaluation will cover next-token cross-entropy or perplexity, training and inference costs, learning curves, and generated text.

### Part II — Embedding Ablation

One neural architecture from Part I will be evaluated with three embedding configurations:

- Randomly initialized, trainable embeddings
- Self-trained, frozen word embeddings (e.g., Skip-gram)
- Public pretrained, frozen embeddings (e.g., GloVe)

Embedding dimensions, architecture, training data, and training procedures will be controlled across configurations.

### Part III — Downstream Representation Analysis

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
- Special tokens: `<pad>`, `<unk>`, `<eos>`
- Level-1 headings used to identify document boundaries
- Higher-level headings excluded from the language-model corpus
- `<eos>` appended to non-empty text lines
- Documents encoded and stored independently

The initial fixed-window next-token dataset uses a maximum of 64 target positions per sample. Padding labels are set to `-100` for exclusion from the loss.

### Processed Dataset Statistics


| Statistic                      | Train     | Validation | Test    |
| ------------------------------ | --------- | ---------- | ------- |
| Documents                      | 629       | 60         | 62      |
| Token IDs                      | 2,049,189 | 213,850    | 240,475 |
| OOV rate                       | 5.07%     | 7.57%      | 8.10%   |
| Next-token samples (length 64) | 32,325    | 3,367      | 3,790   |
| Prediction targets             | 2,048,560 | 213,790    | 240,413 |


OOV rates exclude special tokens. Dataset statistics reflect the current preprocessing implementation.

## 4. Repository Structure

```text
nlp-individual-project/
├── README.md
├── configs/
├── data/
│   └── processed/       # Generated locally, excluded from Git
├── docs/                # Experiment logs and documentation
├── reports/             # Final report
├── results/             # Evaluation results and figures
├── scripts/             # Experiment entry points
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── analyze_data.py
│   │   ├── inspect_headings.py
│   │   ├── preprocess.py
│   │   └── lm_dataset.py
│   ├── models/
│   ├── embeddings/
│   ├── training/
│   └── evaluation/
└── tests/
    └── test_lm_dataset.py
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

The current environment uses a pthreads-based OpenBLAS build to avoid an observed OpenMP runtime initialization conflict.

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

Complete dependency specifications and model training commands will be documented as experiments are implemented.

## 6. Current Progress

- [x] Initialize GitHub repository
- [x] Configure Python and PyTorch environment
- [x] Load WikiText-2
- [x] Analyze vocabulary coverage
- [x] Implement vocabulary construction and preprocessing
- [x] Correct document boundary detection
- [x] Implement next-token dataset and padding
- [x] Validate dataset samples and target coverage
- [ ] Train and evaluate n-gram model
- [ ] Train and evaluate RNN
- [ ] Train and evaluate LSTM
- [ ] Train and evaluate causal Transformer
- [ ] Complete embedding ablation experiments
- [ ] Evaluate downstream representations
- [ ] Finalize reproducibility documentation and report



## 7. Reproducibility and Limitations

The vocabulary is built only from the training split, and document boundaries are preserved in next-token sample construction.

The current dataset implementation treats fixed windows independently. Recurrent model states will not automatically carry across windows. Model-specific context limitations and evaluation procedures will be documented before final comparisons.

The preprocessing pipeline discards heading text and uses line-level end markers. This is a deliberate simplification and may affect language modeling results.

Random seeds, final hyperparameters, dependency versions, model checkpoints, and evaluation settings will be recorded as model experiments are completed.

## 8. Data Sources and Acknowledgments

- WikiText-2 dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Hugging Face Datasets: [https://huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)

This project is developed for an individual university assignment. Standard libraries, documentation, and AI-assisted coding support are used during development. External assistance and implementation references will be documented in the final submission.

## 9. License

No license has been assigned at this stage. The repository is maintained as a private academic project during assignment development.