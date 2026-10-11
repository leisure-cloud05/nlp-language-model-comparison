"""Generate text from the trained interpolated trigram model."""

import json
import random
import re
import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lm_dataset import load_documents, load_vocabulary
from src.models.ngram import NGramLanguageModel


PROMPTS = [
    "The history of",
    "In the city of",
    "The first",
]

SEED = 42
MAX_NEW_TOKENS = 40
TEMPERATURE = 1.0
TOP_K = 20


def sample_next_token(model, context, rng):
    """Sample a token using temperature and top-k filtering."""
    candidates = []

    for token_id in range(model.vocab_size):
        if token_id in model.excluded_targets:
            continue

        probability = model.probability(token_id, context)
        candidates.append((token_id, probability))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    candidates = candidates[:TOP_K]

    weights = [
        probability ** (1.0 / TEMPERATURE)
        for _, probability in candidates
    ]

    return rng.choices(
        [token_id for token_id, _ in candidates],
        weights=weights,
        k=1,
    )[0]


def generate(model, prompt, vocab, id_to_token, rng):
    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt)
    prompt_ids = [
        vocab.get(token, vocab["<unk>"])
        for token in prompt_tokens
    ]

    # Two BOS markers are used only as initial n-gram context.
    history = [vocab["<bos>"], vocab["<bos>"]] + prompt_ids
    generated_ids = []

    start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        next_id = sample_next_token(
            model,
            history[-2:],
            rng,
        )

        generated_ids.append(next_id)
        history.append(next_id)

        if next_id == vocab["<eos>"]:
            break

    elapsed = time.perf_counter() - start

    generated_tokens = [
        id_to_token[token_id]
        for token_id in generated_ids
        if token_id != vocab["<eos>"]
    ]

    return {
        "prompt": prompt,
        "prompt_oov_tokens": [
            token for token in prompt_tokens
            if token not in vocab
        ],
        "generated_text": " ".join(
            prompt_tokens + generated_tokens
        ),
        "num_generated_tokens": len(generated_ids),
        "generation_seconds": elapsed,
        "tokens_per_second": (
            len(generated_ids) / elapsed
        ),
        "stopped_at_eos": (
            bool(generated_ids)
            and generated_ids[-1] == vocab["<eos>"]
        ),
    }


def main():
    if TEMPERATURE <= 0 or TOP_K < 1:
        raise ValueError("Invalid generation settings.")

    vocab = load_vocabulary()
    id_to_token = {
        int(token_id): token
        for token, token_id in vocab.items()
    }

    checkpoint_path = ROOT / "checkpoints" / "ngram.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Trigram checkpoint not found. "
            "Run python scripts/evaluate_ngram.py first."
        )

    print("Loading saved trigram model...")

    with checkpoint_path.open("rb") as file:
        model = pickle.load(file)

    assert isinstance(model, NGramLanguageModel)
    assert model.is_fitted
    assert model.vocab_size == len(vocab)
    assert model.alpha == 10.0
    assert model.tau2 == 320.0
    assert model.tau3 == 320.0

    print("Checkpoint loaded successfully.")

    rng = random.Random(SEED)
    samples = []

    for prompt in PROMPTS:
        result = generate(
            model, prompt, vocab, id_to_token, rng
        )

        samples.append(result)

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result['generated_text']}")
        print(
            f"Speed: {result['tokens_per_second']:.1f} tokens/s"
        )
        print(
            f"Stopped at EOS: {result['stopped_at_eos']}"
        )

        if result["prompt_oov_tokens"]:
            print(
                "Prompt OOV tokens:",
                result["prompt_oov_tokens"],
            )

    total_tokens = sum(
        item["num_generated_tokens"] for item in samples
    )
    total_seconds = sum(
        item["generation_seconds"] for item in samples
    )

    output = {
        "model": "interpolated_trigram",
        "hyperparameters": {
            "alpha": 10.0,
            "tau2": 320.0,
            "tau3": 320.0,
        },
        "generation_config": {
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        },
        "samples": samples,
        "total_generated_tokens": total_tokens,
        "total_generation_seconds": total_seconds,
        "overall_tokens_per_second": (
            total_tokens / total_seconds
        ),
    }

    output_path = ROOT / "results" / "ngram_generation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nOverall generation speed: "
          f"{output['overall_tokens_per_second']:.1f} tokens/s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
