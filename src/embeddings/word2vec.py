
"""Train Skip-gram word embeddings on the existing WikiText-2 corpus."""

from gensim.models import Word2Vec

from src.data.lm_dataset import load_documents, load_vocabulary


SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>"}


class Word2VecCorpus:
    """Re-iterable training corpus for Gensim Word2Vec."""

    def __init__(self, documents, vocabulary):
        self.documents = documents
        self.id_to_token = {
            token_id: token
            for token, token_id in vocabulary.items()
        }

    def __iter__(self):
        for document in self.documents:
            sentence = [
                self.id_to_token[token_id]
                for token_id in document
                if self.id_to_token[token_id] not in SPECIAL_TOKENS
            ]

            if sentence:
                yield sentence


def train_skipgram(
    vector_size=100,
    window=5,
    epochs=5,
    seed=42,
):
    """Train Skip-gram using WikiText-2 training documents only."""

    documents = load_documents("train")
    vocabulary = load_vocabulary()

    corpus = Word2VecCorpus(documents, vocabulary)

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        negative=5,
        workers=1,
        seed=seed,
        epochs=epochs,
    )

    return model
