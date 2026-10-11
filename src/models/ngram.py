"""Trigram language model with interpolated smoothing."""

from collections import Counter
import math


class NGramLanguageModel:
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        bos_id: int = 2,
        alpha: float = 0.1,
        tau2: float = 10.0,
        tau3: float = 10.0,
    ):
        if vocab_size <= 2:
            raise ValueError("Vocabulary is too small.")
        if alpha <= 0 or tau2 <= 0 or tau3 <= 0:
            raise ValueError("Smoothing parameters must be positive.")
        if pad_id == bos_id:
            raise ValueError("PAD and BOS must have different IDs.")
        if not all(
            0 <= token < vocab_size
            for token in (pad_id, bos_id)
        ):
            raise ValueError("Special token IDs are invalid.")

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id

        self.alpha = alpha
        self.tau2 = tau2
        self.tau3 = tau3

        self.excluded_targets = {pad_id, bos_id}
        self.prediction_vocab_size = (
            vocab_size - len(self.excluded_targets)
        )

        self.unigram = Counter()
        self.bigram = Counter()
        self.trigram = Counter()

        self.bigram_context = Counter()
        self.trigram_context = Counter()

        self.total_targets = 0
        self.is_fitted = False

    def _validate_document(self, document):
        if len(document) < 2:
            raise ValueError("Document is too short.")
        if document[0] != self.bos_id:
            raise ValueError("Document must begin with BOS.")

        for token in document:
            if not 0 <= int(token) < self.vocab_size:
                raise ValueError("Token ID outside vocabulary.")

        for token in document[1:]:
            if token in self.excluded_targets:
                raise ValueError(
                    "PAD/BOS cannot appear as prediction targets."
                )

    def fit(self, documents):
        """Count all prediction events, with virtual BOS padding."""
        self.unigram.clear()
        self.bigram.clear()
        self.trigram.clear()
        self.bigram_context.clear()
        self.trigram_context.clear()
        self.total_targets = 0
        self.is_fitted = False

        for document in documents:
            self._validate_document(document)

            # The document already starts with one BOS.
            # Add one more only inside the n-gram model.
            padded = [self.bos_id] + list(document)

            # The first TWO entries are context, never targets.
            for i in range(2, len(padded)):
                u = int(padded[i - 2])
                v = int(padded[i - 1])
                w = int(padded[i])

                self.unigram[w] += 1
                self.bigram[(v, w)] += 1
                self.trigram[(u, v, w)] += 1

                self.bigram_context[v] += 1
                self.trigram_context[(u, v)] += 1

                self.total_targets += 1

        if self.total_targets == 0:
            raise ValueError("No training targets found.")

        self.is_fitted = True
        return self

    def probability(self, word, context):
        """Return interpolated probability of word given context."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        word = int(word)

        if not 0 <= word < self.vocab_size:
            raise ValueError("Target ID outside vocabulary.")

        if word in self.excluded_targets:
            return 0.0

        context = tuple(int(t) for t in context)[-2:]

        if any(
            not 0 <= token < self.vocab_size
            for token in context
        ):
            raise ValueError("Context ID outside vocabulary.")

        # 1. Add-alpha smoothed unigram probability
        p1 = (
            self.unigram[word] + self.alpha
        ) / (
            self.total_targets
            + self.alpha * self.prediction_vocab_size
        )

        if len(context) == 0:
            return p1

        # 2. Interpolated bigram probability
        v = context[-1]
        p2 = (
            self.bigram[(v, word)]
            + self.tau2 * p1
        ) / (
            self.bigram_context[v] + self.tau2
        )

        if len(context) == 1:
            return p2

        # 3. Interpolated trigram probability
        u = context[-2]
        p3 = (
            self.trigram[(u, v, word)]
            + self.tau3 * p2
        ) / (
            self.trigram_context[(u, v)] + self.tau3
        )

        return p3

    def evaluate(self, documents):
        """Compute token-level cross-entropy and perplexity."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        total_nll = 0.0
        total_targets = 0

        for document in documents:
            self._validate_document(document)
            padded = [self.bos_id] + list(document)

            for i in range(2, len(padded)):
                context = (padded[i - 2], padded[i - 1])
                target = padded[i]

                p = self.probability(target, context)
                total_nll -= math.log(p)
                total_targets += 1

        if total_targets == 0:
            raise ValueError("No evaluation targets found.")

        cross_entropy = total_nll / total_targets

        return {
            "cross_entropy": cross_entropy,
            "perplexity": math.exp(cross_entropy),
            "num_targets": total_targets,
        }

    def model_statistics(self):
        """Summarize observed n-gram counts."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        return {
            "unigrams": len(self.unigram),
            "bigrams": len(self.bigram),
            "trigrams": len(self.trigram),
            "training_targets": self.total_targets,
        }
