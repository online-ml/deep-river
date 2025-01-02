from typing import Callable, Union

import torch.nn as nn
from river import base


class WordEmbeddingTransformer(base.Transformer):
    class WordEmbeddingModule(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            """
            Simple word embedding model using PyTorch's nn.Embedding.

            Parameters
            ----------
            vocab_size : int
                Size of the vocabulary.
            embedding_dim : int
                Dimensionality of the word embeddings.
            """
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            """
            Forward pass of the model.

            Parameters
            ----------
            x : torch.Tensor
                Tensor containing word indices.

            Returns
            -------
            torch.Tensor
                Embedded word vectors.
            """
            return self.embedding(x)

    def __init__(
        self,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):

        if "module" in kwargs:
            del kwargs["module"]

        super().__init__(
            module=WordEmbeddingTransformer.WordEmbeddingModule,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            seed=seed,
            **kwargs,
        )
