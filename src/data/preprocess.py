import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


# Configuration

SEED = 42
VOCAB_SIZE = 20000

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEADING_PATTERN = re.compile(
    r"^((?:=\s*)+)(.+?)\s+((?:=\s*)+)$"
)


# Text processing

def normalize_text(text):
    """Normalize WikiText formatting consistently."""
    text = text.replace("@-@", "-")
    text = text.replace("@,@", ",")
    text = text.replace("@.@", ".")
    return text.strip()


def tokenize(text):
    """Split text into word and punctuation tokens."""
    text = normalize_text(text)
    return TOKEN_PATTERN.findall(text)



def extract_documents(text_rows):
    """
    Split WikiText into documents using level-1 headings.

    Level-1 headings start new documents.
    Other heading levels are skipped.
    Non-empty text lines end with <eos>.
    """
    documents = []
    current_document = []

    for raw_text in text_rows:
        text = raw_text.strip()

        if not text:
            continue

        match = HEADING_PATTERN.fullmatch(text)

        if match:
            left_level = match.group(1).count("=")
            right_level = match.group(3).count("=")

            # Only treat valid headings with matching levels
            if left_level == right_level:
                if left_level == 1:
                    if current_document:
                        documents.append(current_document)

                    current_document = []

                # Skip all heading text
                continue

        tokens = tokenize(text)

        if tokens:
            current_document.extend(tokens)
            current_document.append(EOS_TOKEN)

    if current_document:
        documents.append(current_document)

    return documents



# =========================
# Vocabulary
# =========================

def build_vocabulary(train_documents):
    counter = Counter()

    for document in train_documents:
        counter.update(
            token for token in document
            if token not in SPECIAL_TOKENS
        )

    vocab = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    for token, _ in counter.most_common(
        VOCAB_SIZE - len(SPECIAL_TOKENS)
    ):
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_documents(documents, vocab):
    unk_id = vocab[UNK_TOKEN]

    return [
        [vocab.get(token, unk_id) for token in document]
        for document in documents
    ]


def calculate_oov(documents, vocab):
    total = 0
    unknown = 0

    for document in documents:
        for token in document:
            if token in SPECIAL_TOKENS:
                continue

            total += 1

            if token not in vocab:
                unknown += 1

    return unknown / total if total else 0.0


# =========================
# Main pipeline
# =========================

def main():
    print("Loading WikiText-2...")

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents_by_split = {}

    for split in ["train", "validation", "test"]:
        documents = extract_documents(
            dataset[split]["text"]
        )

        documents_by_split[split] = documents

        print(
            f"{split}: {len(documents)} documents"
        )

    # Build vocabulary ONLY from training data
    vocab = build_vocabulary(
        documents_by_split["train"]
    )

    assert len(vocab) <= VOCAB_SIZE

    with open(
        OUTPUT_DIR / "vocab.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"\nVocabulary size: {len(vocab)}")

    statistics = {
        "vocab_size": len(vocab),
        "seed": SEED,
        "splits": {}
    }

    for split, documents in documents_by_split.items():

        encoded_documents = encode_documents(
            documents, vocab
        )

        oov_rate = calculate_oov(documents, vocab)

        total_tokens = sum(
            len(doc) for doc in encoded_documents
        )

        output_path = OUTPUT_DIR / f"{split}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(encoded_documents, f)

        statistics["splits"][split] = {
            "documents": len(documents),
            "tokens": total_tokens,
            "oov_rate": oov_rate
        }

        print(
            f"{split}: "
            f"{total_tokens} tokens, "
            f"OOV = {oov_rate:.2%}"
        )

    with open(
        OUTPUT_DIR / "statistics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(statistics, f, indent=2)

    print("\nPreprocessing completed!")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
