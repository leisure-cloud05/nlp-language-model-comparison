
import re
from collections import Counter
from datasets import load_dataset

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

# WikiText headings use separated equals signs:
# = Article =
# = = Section = =
# = = = Subsection = = =

heading_counts = Counter()
examples = {}

for text in dataset["train"]["text"]:
    line = text.strip()

    if not line.startswith("="):
        continue

    # Count all '=' characters at the beginning,
    # including spaces between them.
    match = re.match(r"^((?:=\s*)+)", line)

    if match:
        level = match.group(1).count("=")
        heading_counts[level] += 1

        if level not in examples:
            examples[level] = line

print("Heading counts by level:")

for level in sorted(heading_counts):
    print(f"Level {level}: {heading_counts[level]}")

print("\nExamples:")

for level in sorted(examples):
    print(f"Level {level}: {examples[level]}")
