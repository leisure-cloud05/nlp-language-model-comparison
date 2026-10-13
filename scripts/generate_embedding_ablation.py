"""Generate text for Part II embedding ablation using Part I LSTM decoding."""

import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data.lm_dataset import load_vocabulary
from src.models.embedding_lstm import EmbeddingAblationLSTM


PROMPTS = [
    "The history of",
    "In the city of",
    "The first",
]

STRATEGIES = ("random", "skipgram", "glove")

SEED = 42
MAX_NEW_TOKENS = 40
TEMPERATURE = 1.0
TOP_K = 20

# False: original model sampling; True: filter <unk>.
EXCLUDE_UNK = False


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def sample_next_token(logits, rng, pad_id, bos_id, unk_id):
    """Apply the exact Part I temperature + top-k sampling procedure."""
    logits = logits.float().clone()

    # PAD and BOS are not valid generated targets.
    logits[pad_id] = float("-inf")
    logits[bos_id] = float("-inf")

    if EXCLUDE_UNK:
        logits[unk_id] = float("-inf")

    logits = logits / TEMPERATURE

    k = min(TOP_K, logits.numel())
    top_logits, top_indices = torch.topk(logits, k=k)

    probabilities = torch.softmax(top_logits, dim=-1)

    # Use a local Python RNG for reproducible sampling.
    weights = probabilities.cpu().tolist()
    selected_position = rng.choices(
        range(k),
        weights=weights,
        k=1,
    )[0]

    return int(top_indices[selected_position].item())


@torch.inference_mode()
def generate(model, prompt, vocab, id_to_token, device, rng, seq_len):
    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt)

    prompt_ids = [
        vocab.get(token, vocab["<unk>"])
        for token in prompt_tokens
    ]

    # One real BOS token at document beginning.
    history = [vocab["<bos>"]] + prompt_ids
    generated_ids = []

    synchronize(device)
    start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        # Match the maximum training window length.
        context = history[-seq_len:]

        inputs = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        # Recompute context rather than carrying LSTM states.
        logits, _ = model(inputs)
        next_logits = logits[0, -1, :]

        next_id = sample_next_token(
            next_logits,
            rng,
            pad_id=vocab["<pad>"],
            bos_id=vocab["<bos>"],
            unk_id=vocab["<unk>"],
        )

        generated_ids.append(next_id)
        history.append(next_id)

        if next_id == vocab["<eos>"]:
            break

    synchronize(device)
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
        "tokens_per_second": len(generated_ids) / elapsed,
        "stopped_at_eos": (
            bool(generated_ids)
            and generated_ids[-1] == vocab["<eos>"]
        ),
    }


def load_model(strategy, vocab, device):
    checkpoint_filename = f"embedding_ablation_{strategy}_best.pt"
    checkpoint_path = ROOT / "checkpoints" / checkpoint_filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    frozen = checkpoint["freeze_embeddings"]

    assert checkpoint["embedding_strategy"] == strategy
    assert checkpoint["vocab_size"] == len(vocab)
    assert checkpoint["pad_id"] == vocab["<pad>"]
    assert frozen == (strategy != "random")

    # Placeholder only: actual embedding values are restored from checkpoint.
    pretrained = (
        torch.zeros(len(vocab), config["embedding_dim"])
        if frozen else None
    )

    model = EmbeddingAblationLSTM(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_id=checkpoint["pad_id"],
        pretrained_embeddings=pretrained,
        freeze_embeddings=frozen,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    assert model.embedding.weight.requires_grad == (not frozen)
    return model, checkpoint, checkpoint_filename


def main():
    if TEMPERATURE <= 0 or TOP_K < 1 or MAX_NEW_TOKENS < 1:
        raise ValueError("Invalid generation settings.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    vocab = load_vocabulary()
    id_to_token = {
        int(token_id): token
        for token, token_id in vocab.items()
    }

    print(f"Device: {device}", flush=True)
    print(
        f"Generation: temperature={TEMPERATURE}, top_k={TOP_K}, "
        f"max_new_tokens={MAX_NEW_TOKENS}, seed={SEED}, "
        f"exclude_unk={EXCLUDE_UNK}",
        flush=True,
    )

    for strategy in STRATEGIES:
        print(f"\n{'=' * 60}")
        print(f"Embedding strategy: {strategy.upper()}", flush=True)

        model, checkpoint, checkpoint_filename = load_model(
            strategy, vocab, device
        )
        config = checkpoint["config"]
        print(f"Checkpoint loaded. Best epoch: {checkpoint['best_epoch']}")

        # Warm up MPS inference before measuring generation.
        print("Warming up model inference...", flush=True)
        with torch.inference_mode():
            warmup_input = torch.tensor(
                [[vocab["<bos>"], vocab.get("The", vocab["<unk>"])]],
                dtype=torch.long,
                device=device,
            )
            for _ in range(5):
                model(warmup_input)

        synchronize(device)

        # Restart same Part I RNG sequence for each model: fair comparison.
        rng = random.Random(SEED)
        samples = []

        for prompt in PROMPTS:
            result = generate(
                model=model,
                prompt=prompt,
                vocab=vocab,
                id_to_token=id_to_token,
                device=device,
                rng=rng,
                seq_len=config["seq_len"],
            )
            samples.append(result)

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {result['generated_text']}")
            print(f"Speed: {result['tokens_per_second']:.1f} tokens/s")
            print(f"Stopped at EOS: {result['stopped_at_eos']}")
            if result["prompt_oov_tokens"]:
                print("Prompt OOV tokens:", result["prompt_oov_tokens"])

        total_tokens = sum(s["num_generated_tokens"] for s in samples)
        total_seconds = sum(s["generation_seconds"] for s in samples)

        output = {
            "model": "embedding_ablation_lstm",
            "embedding_strategy": strategy,
            "checkpoint": checkpoint_filename,
            "best_epoch": checkpoint["best_epoch"],
            "generation_config": {
                "seed": SEED,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "exclude_unk": EXCLUDE_UNK,
            },
            "samples": samples,
            "total_generated_tokens": total_tokens,
            "total_generation_seconds": total_seconds,
            "overall_tokens_per_second": total_tokens / total_seconds,
            "device": str(device),
        }

        suffix = "_generation_no_unk.json" if EXCLUDE_UNK else "_generation.json"
        output_path = ROOT / "results" / f"embedding_ablation_{strategy}{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)

        print(
            f"\nOverall generation speed: "
            f"{output['overall_tokens_per_second']:.1f} tokens/s"
        )
        print(f"Results saved to: {output_path}", flush=True)

    print("\nAll Part II generation experiments completed.")


if __name__ == "__main__":
    main()
