from datasets import load_dataset

# Load WikiText-2
dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

# Display dataset information
print(dataset)

for split in ["train", "validation", "test"]:
    texts = dataset[split]["text"]

    non_empty = [text for text in texts if text.strip()]

    print(f"\n{split}:")
    print(f"Total rows: {len(texts)}")
    print(f"Non-empty rows: {len(non_empty)}")
    print(f"Example: {non_empty[0][:200]}")