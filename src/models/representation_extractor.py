"""Extract fixed-length representations from frozen language models."""

import torch


@torch.inference_mode()
def extract_lstm_chunk(model, input_ids):
    """Extract the final hidden state of an unpadded text chunk.

    Args:
        model: Frozen LSTMLanguageModel in evaluation mode.
        input_ids: LongTensor of shape [1, T], with 1 <= T <= 64.
                   Contains real tokens only, without padding.

    Returns:
        Tensor of shape [1, hidden_dim].
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected input_ids with shape [1, T].")

    if not 1 <= input_ids.size(1) <= 64:
        raise ValueError("Chunk length must be between 1 and 64.")

    if (input_ids == model.pad_id).any().item():
        raise ValueError("Input chunk must not contain padding.")

    embeddings = model.embedding(input_ids)

    outputs, (h_n, c_n) = model.lstm(embeddings)

    # h_n: [num_layers, batch_size, hidden_dim]
    # Select the final LSTM layer.
    representation = h_n[-1]

    return representation


@torch.inference_mode()
def extract_transformer_chunk(model, input_ids):
    """Mean-pool final hidden states from an unpadded chunk.

    Args:
        model: Frozen CausalTransformerLanguageModel in eval mode.
        input_ids: LongTensor [1, T], with 1 <= T <= max_seq_len.
                   Must not contain padding.

    Returns:
        Tensor [1, embedding_dim].
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected input_ids with shape [1, T].")

    seq_len = input_ids.size(1)

    if not 1 <= seq_len <= model.max_seq_len:
        raise ValueError("Invalid chunk length.")

    pad_id = model.token_embedding.padding_idx

    if (input_ids == pad_id).any().item():
        raise ValueError("Input chunk must not contain padding.")

    positions = torch.arange(
        seq_len,
        device=input_ids.device,
    )

    x = (
        model.token_embedding(input_ids)
        + model.position_embedding(positions)[None, :, :]
    )

    x = model.embedding_dropout(x)

    for block in model.blocks:
        x = block(x)

    x = model.final_norm(x)

    # All positions are valid: input chunks contain no padding.
    representation = x.mean(dim=1)

    return representation


@torch.inference_mode()
def extract_review_representation(
    model,
    token_ids,
    model_type,
    device,
):
    """Extract a fixed-size representation for one IMDb review."""
    from src.data.imdb_preprocess import split_into_chunks

    chunks = split_into_chunks(token_ids)

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    chunk_representations = []

    for chunk in chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            representation = extract_lstm_chunk(model, input_ids)
        elif model_type == "transformer":
            representation = extract_transformer_chunk(model, input_ids)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        chunk_representations.append(representation)

    return torch.cat(chunk_representations, dim=0).mean(
        dim=0, keepdim=True
    )



@torch.inference_mode()
def extract_review_representation_batched(
    model,
    token_ids,
    model_type,
    device,
):
    """Extract one review representation with batched full chunks."""
    from src.data.imdb_preprocess import split_into_chunks, CHUNK_SIZE

    chunks = split_into_chunks(token_ids)

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    if model_type not in ("lstm", "transformer"):
        raise ValueError(f"Unknown model type: {model_type}")

    full_chunks = [c for c in chunks if len(c) == CHUNK_SIZE]
    short_chunks = [c for c in chunks if len(c) < CHUNK_SIZE]

    representations = []

    if full_chunks:
        batch = torch.tensor(
            full_chunks,
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            embeddings = model.embedding(batch)
            _, (h_n, _) = model.lstm(embeddings)
            full_representations = h_n[-1]

        else:
            positions = torch.arange(
                CHUNK_SIZE,
                device=device,
            )

            x = (
                model.token_embedding(batch)
                + model.position_embedding(positions)[None, :, :]
            )

            x = model.embedding_dropout(x)

            for block in model.blocks:
                x = block(x)

            x = model.final_norm(x)
            full_representations = x.mean(dim=1)

        representations.append(full_representations)

    for chunk in short_chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            rep = extract_lstm_chunk(model, input_ids)
        else:
            rep = extract_transformer_chunk(model, input_ids)

        representations.append(rep)

    return torch.cat(representations, dim=0).mean(
        dim=0, keepdim=True
    )



@torch.inference_mode()
def extract_lstm_mean_chunk(model, input_ids):
    """Mean-pool LSTM hidden states over an unpadded chunk.

    Args:
        model: Frozen LSTMLanguageModel in eval mode.
        input_ids: LongTensor [1, T], with 1 <= T <= 64.
                   Must not contain padding.

    Returns:
        Tensor [1, hidden_dim].
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected input_ids with shape [1, T].")

    if not 1 <= input_ids.size(1) <= 64:
        raise ValueError("Chunk length must be between 1 and 64.")

    if (input_ids == model.pad_id).any().item():
        raise ValueError("Input chunk must not contain padding.")

    embeddings = model.embedding(input_ids)

    # outputs: [1, T, 256]
    outputs, _ = model.lstm(embeddings)

    # Mean over the token dimension, not the hidden dimension.
    representation = outputs.mean(dim=1)

    return representation



@torch.inference_mode()
def extract_lstm_mean_review_batched(model, token_ids, device):
    """Extract one review using mean-pooled LSTM chunk representations."""
    from src.data.imdb_preprocess import split_into_chunks, CHUNK_SIZE

    chunks = split_into_chunks(token_ids)

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    full_chunks = [chunk for chunk in chunks if len(chunk) == CHUNK_SIZE]
    short_chunks = [chunk for chunk in chunks if len(chunk) < CHUNK_SIZE]

    representations = []

    # Process all full-length chunks together.
    if full_chunks:
        batch = torch.tensor(
            full_chunks,
            dtype=torch.long,
            device=device,
        )

        embeddings = model.embedding(batch)
        outputs, _ = model.lstm(embeddings)

        # [K, 64, 256] -> [K, 256]
        full_representations = outputs.mean(dim=1)

        representations.append(full_representations)

    # Process remaining shorter chunks without padding.
    for chunk in short_chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        representation = extract_lstm_mean_chunk(
            model, input_ids
        )

        representations.append(representation)

    # Equal-weight mean across chunks.
    return torch.cat(representations, dim=0).mean(
        dim=0, keepdim=True
    )



def token_weighted_chunk_pooling(chunk_representations, chunk_lengths):
    """Aggregate chunk vectors weighted by valid token counts.

    Args:
        chunk_representations: Tensor [K, D].
        chunk_lengths: List of K positive token counts.

    Returns:
        Tensor [1, D].
    """
    if chunk_representations.ndim != 2:
        raise ValueError("Expected representations with shape [K, D].")

    if len(chunk_lengths) != chunk_representations.size(0):
        raise ValueError("Chunk counts and representation counts differ.")

    if not chunk_lengths or any(length <= 0 for length in chunk_lengths):
        raise ValueError("All chunk lengths must be positive.")

    weights = torch.tensor(
        chunk_lengths,
        dtype=chunk_representations.dtype,
        device=chunk_representations.device,
    )

    weights = weights / weights.sum()

    representation = (
        chunk_representations * weights.unsqueeze(1)
    ).sum(dim=0, keepdim=True)

    return representation


@torch.inference_mode()
def extract_lstm_mean_weighted_review_batched(
    model,
    token_ids,
    device,
):
    """Extract a review with token-weighted LSTM mean pooling."""
    from src.data.imdb_preprocess import (
        split_into_chunks,
        CHUNK_SIZE,
    )

    chunks = split_into_chunks(token_ids)

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    full_chunks = [
        chunk for chunk in chunks
        if len(chunk) == CHUNK_SIZE
    ]
    short_chunks = [
        chunk for chunk in chunks
        if len(chunk) < CHUNK_SIZE
    ]

    representations = []
    lengths = []

    # Batch full-length chunks.
    if full_chunks:
        batch = torch.tensor(
            full_chunks,
            dtype=torch.long,
            device=device,
        )

        embeddings = model.embedding(batch)
        outputs, _ = model.lstm(embeddings)

        full_representations = outputs.mean(dim=1)

        representations.append(full_representations)
        lengths.extend([CHUNK_SIZE] * len(full_chunks))

    # Handle short chunks without padding.
    for chunk in short_chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        representation = extract_lstm_mean_chunk(
            model,
            input_ids,
        )

        representations.append(representation)
        lengths.append(len(chunk))

    chunk_representations = torch.cat(
        representations,
        dim=0,
    )

    return token_weighted_chunk_pooling(
        chunk_representations,
        lengths,
    )


@torch.inference_mode()
def extract_transformer_mean_weighted_review_batched(
    model,
    token_ids,
    device,
):
    """Extract token-weighted Transformer review representation."""
    from src.data.imdb_preprocess import (
        split_into_chunks,
        CHUNK_SIZE,
    )

    chunks = split_into_chunks(token_ids)

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    full_chunks = [
        chunk for chunk in chunks
        if len(chunk) == CHUNK_SIZE
    ]
    short_chunks = [
        chunk for chunk in chunks
        if len(chunk) < CHUNK_SIZE
    ]

    representations = []
    lengths = []

    if full_chunks:
        batch = torch.tensor(
            full_chunks,
            dtype=torch.long,
            device=device,
        )

        positions = torch.arange(
            CHUNK_SIZE,
            device=device,
        )

        x = (
            model.token_embedding(batch)
            + model.position_embedding(positions)[None, :, :]
        )

        x = model.embedding_dropout(x)

        for block in model.blocks:
            x = block(x)

        x = model.final_norm(x)

        representations.append(x.mean(dim=1))
        lengths.extend([CHUNK_SIZE] * len(full_chunks))

    for chunk in short_chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        representations.append(
            extract_transformer_chunk(model, input_ids)
        )
        lengths.append(len(chunk))

    chunk_representations = torch.cat(
        representations,
        dim=0,
    )

    return token_weighted_chunk_pooling(
        chunk_representations,
        lengths,
    )



def mean_max_chunk_pooling(chunk_representations):
    """Combine mean and max statistics across chunk representations.

    Args:
        chunk_representations: Tensor [K, D], K >= 1.

    Returns:
        Tensor [1, 2 * D].
    """
    if chunk_representations.ndim != 2:
        raise ValueError("Expected chunk representations with shape [K, D].")

    if chunk_representations.size(0) == 0:
        raise ValueError("Cannot pool zero chunks.")

    mean_representation = chunk_representations.mean(
        dim=0, keepdim=True
    )

    max_representation = chunk_representations.max(
        dim=0, keepdim=True
    ).values

    return torch.cat(
        [mean_representation, max_representation],
        dim=1,
    )


@torch.inference_mode()
def extract_mean_max_review_batched(
    model,
    token_ids,
    model_type,
    device,
):
    """Extract a review using mean + max pooling across chunks."""
    from src.data.imdb_preprocess import (
        split_into_chunks,
        CHUNK_SIZE,
    )

    if model_type not in ("lstm", "transformer"):
        raise ValueError(f"Unknown model type: {model_type}")

    chunks = split_into_chunks(token_ids)

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    full_chunks = [
        chunk for chunk in chunks
        if len(chunk) == CHUNK_SIZE
    ]

    short_chunks = [
        chunk for chunk in chunks
        if len(chunk) < CHUNK_SIZE
    ]

    representations = []

    # Batch the full-length chunks.
    if full_chunks:
        batch = torch.tensor(
            full_chunks,
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            embeddings = model.embedding(batch)
            outputs, _ = model.lstm(embeddings)
            full_reps = outputs.mean(dim=1)

        else:
            positions = torch.arange(
                CHUNK_SIZE,
                device=device,
            )

            x = (
                model.token_embedding(batch)
                + model.position_embedding(positions)[None, :, :]
            )

            x = model.embedding_dropout(x)

            for block in model.blocks:
                x = block(x)

            x = model.final_norm(x)
            full_reps = x.mean(dim=1)

        representations.append(full_reps)

    # Handle shorter chunks without padding.
    for chunk in short_chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            rep = extract_lstm_mean_chunk(
                model, input_ids
            )
        else:
            rep = extract_transformer_chunk(
                model, input_ids
            )

        representations.append(rep)

    chunk_representations = torch.cat(
        representations,
        dim=0,
    )

    return mean_max_chunk_pooling(
        chunk_representations
    )


@torch.inference_mode()
def extract_overlapping_review_batched(
    model,
    token_ids,
    model_type,
    device,
    stride=32,
):
    """Extract a review representation using overlapping windows."""
    from src.data.imdb_preprocess import (
        split_into_overlapping_chunks,
        CHUNK_SIZE,
    )

    if model_type not in ("lstm", "transformer"):
        raise ValueError(f"Unknown model type: {model_type}")

    chunks = split_into_overlapping_chunks(
        token_ids,
        chunk_size=CHUNK_SIZE,
        stride=stride,
    )

    if not chunks:
        raise ValueError("Cannot extract representation from empty review.")

    representations = []

    full_chunks = [
        chunk for chunk in chunks if len(chunk) == CHUNK_SIZE
    ]
    short_chunks = [
        chunk for chunk in chunks if len(chunk) < CHUNK_SIZE
    ]

    if full_chunks:
        batch = torch.tensor(
            full_chunks,
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            embeddings = model.embedding(batch)
            outputs, _ = model.lstm(embeddings)
            representations.append(outputs.mean(dim=1))

        else:
            positions = torch.arange(
                CHUNK_SIZE, device=device
            )

            x = (
                model.token_embedding(batch)
                + model.position_embedding(positions)[None, :, :]
            )

            x = model.embedding_dropout(x)

            for block in model.blocks:
                x = block(x)

            x = model.final_norm(x)
            representations.append(x.mean(dim=1))

    for chunk in short_chunks:
        input_ids = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        if model_type == "lstm":
            representation = extract_lstm_mean_chunk(
                model, input_ids
            )
        else:
            representation = extract_transformer_chunk(
                model, input_ids
            )

        representations.append(representation)

    return torch.cat(
        representations, dim=0
    ).mean(dim=0, keepdim=True)