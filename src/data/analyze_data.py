from collections import Counter
import re

from datasets import load_dataset

# 1. Load WikiText-2
dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

# 2. Use the same tokenizer for all splits
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

# 3. Tokenize train, validation, and test sets
tokens_by_split = {}

for split in ["train", "validation", "test"]:
    tokens = []

    for text in dataset[split]["text"]:
        if text.strip():
            tokens.extend(tokenize(text))

    tokens_by_split[split] = tokens

# 4. Count vocabulary frequencies using training data only
train_counter = Counter(tokens_by_split["train"])

print("Training tokens:", len(tokens_by_split["train"]))
print("Unique training tokens:", len(train_counter))

# 5. Compare candidate vocabulary sizes
vocab_sizes = [10000, 20000, 30000]

num_special_tokens = 3  # <pad>, <unk>, <eos>

print("\nVocabulary size comparison")
print("-" * 75)
print(
    f"{'Vocab Size':<15}"
    f"{'Train Coverage':<20}"
    f"{'Validation OOV':<20}"
    f"{'Test OOV':<15}"
)
print("-" * 75)

for vocab_size in vocab_sizes:

    # Reserve space for special tokens
    num_words = vocab_size - num_special_tokens

    # Build vocabulary using training data only
    vocab = {
        word for word, _ in train_counter.most_common(num_words)
    }

    # Training coverage
    train_tokens = tokens_by_split["train"]
    train_covered = sum(token in vocab for token in train_tokens)
    train_coverage = train_covered / len(train_tokens)

    # Validation OOV
    val_tokens = tokens_by_split["validation"]
    val_oov = sum(token not in vocab for token in val_tokens)
    val_oov_rate = val_oov / len(val_tokens)

    # Test OOV
    test_tokens = tokens_by_split["test"]
    test_oov = sum(token not in vocab for token in test_tokens)
    test_oov_rate = test_oov / len(test_tokens)

    print(
        f"{vocab_size:<15}"
        f"{train_coverage:<20.2%}"
        f"{val_oov_rate:<20.2%}"
        f"{test_oov_rate:<15.2%}"
    )
