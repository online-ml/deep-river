from collections import defaultdict
from typing import Callable, Dict, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim
from river import base
from torchtext.data import get_tokenizer

from deep_river.base import DeepEstimator


class EmbeddingTransformer(DeepEstimator, base.transformer.BaseTransformer):
    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        tokenizer=None,
        language: str = "en",
        lr=0.01,
        device: str = "cpu",
        seed=42,
        **kwargs,
    ):

        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )
        self.tokenizer = get_tokenizer(tokenizer, language=language)

        # Vocabulary with an <UNK> token
        self.vocab = defaultdict(lambda: self.vocab["<UNK>"])
        self.vocab["<UNK>"] = 0  # Reserve index 0 for <UNK>

    def learn_one(self, x: dict) -> None:
        """
        Learn from a single instance by updating the module parameters.

        Parameters
        ----------
        x : dict
            Dictionary of text data to be processed.

        Returns
        -------
        None
        """

        token_list = []
        for text in x.values():
            token_list.extend(self.tokenizer(text))
            
        if not self.module_initialized:
            self.initialize_module(x=x, n_features=len(self.vocab), **self.kwargs)

        # Map tokens to indices, adding new tokens to the vocabulary
        token_indices = [self.vocab[token] for token in token_list]

        if not token_indices:
            raise ValueError("Input text contains no valid tokens.")

        input_tensor = torch.tensor([token_indices], dtype=torch.long)

        # Forward pass
        embedding = self.module(input_tensor)

        # Generate a "target" embedding by shifting the input
        shifted_tensor = input_tensor.roll(shifts=1, dims=1)

        # Forward pass for the target embedding
        target_embedding = self.module(shifted_tensor)

        # Compute cosine similarity loss
        loss = 1 - nn.functional.cosine_similarity(embedding, target_embedding).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def transform_one(self, x: dict) -> dict:
        """
        Transform a single instance (x) into embeddings for each key.

        Parameters
        ----------
        x : dict
            Dictionary where the keys are labels (e.g., field names) and values are text (strings).

        Returns
        -------
        dict
            A dictionary where keys are the input keys and values are their embeddings.
        """
        if not self.module_initialized:
            self.initialize_module(x=x, n_features=len(self.vocab), **self.kwargs)
        self.module.eval()

        embeddings = {}
        for key, text in x.items():
            token_list = self.tokenizer(text)

            # Map tokens to indices, using <UNK> for unknown tokens
            token_indices = [self.vocab[token] for token in token_list]

            if not token_indices:
                raise ValueError(
                    f"Input text for key '{key}' contains no valid tokens."
                )

            input_tensor = torch.tensor([token_indices], dtype=torch.long)

            # Compute embeddings
            with torch.no_grad():
                embedding = self.module(input_tensor).squeeze(0)

            # Mean Pooling: Average over the word embeddings to get a fixed-size vector
            pooled_embedding = embedding.mean(dim=0)

            # Add to the output dictionary
            print(pooled_embedding)
            embeddings[key] = float(pooled_embedding.numpy())

        self.module.train()
        return embeddings


class WordEmbeddingModel(nn.Module):
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

