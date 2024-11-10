import torch
import torch.nn as nn
import torch.optim as optim
from river import base
from torchtext.vocab import build_vocab_from_iterator
from collections import defaultdict


class EmbeddingTransformer(base.Transformer):
    def __init__(self, vocab_size, embedding_dim, tokenizer, learning_rate=0.01):
        """
        Transformer that generates word embeddings and transforms variable-length input into a
        fixed-size vector using mean pooling. The input is a dictionary of sentences/words,
        which are tokenized and transformed into fixed-size vectors.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        embedding_dim : int
            Dimensionality of the word embeddings.
        tokenizer : callable
            Function that tokenizes a sentence/word into tokens.
        learning_rate : float
            Learning rate for the optimizer.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tokenizer  # Tokenizer that converts sentences/words into tokens
        self.learning_rate = learning_rate

        # PyTorch model
        self.model = WordEmbeddingModel(vocab_size, embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Call the PyTorch model's train mode
        self.model.train()

    def learn_one(self, x):
        """
        Update the model with a single instance (x).

        Parameters
        ----------
        x : dict
            Dictionary where the keys are words/sentences and values are text (strings).

        Returns
        -------
        self
        """
        # Tokenize and concatenate token lists from the dictionary
        token_list = []
        for text in x.values():
            token_list.extend(self.tokenizer(text))  # Tokenize each sentence/word

        input_tensor = torch.tensor([token_list], dtype=torch.long)

        # Forward pass
        embedding = self.model(input_tensor)

        # Loss computation and backward pass
        loss = self.criterion(embedding, input_tensor)  # Target and input should be aligned for the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self

    def transform_one(self, x):
        """
        Transform a single instance (x) into a fixed-length embedding.

        Parameters
        ----------
        x : dict
            Dictionary where the keys are words/sentences and values are text (strings).

        Returns
        -------
        torch.Tensor
            A fixed-length vector representing the input text.
        """
        self.model.eval()  # Switch to eval mode

        # Tokenize and concatenate token lists from the dictionary
        token_list = []
        for text in x.values():
            token_list.extend(self.tokenizer(text))  # Tokenize each sentence/word

        input_tensor = torch.tensor([token_list], dtype=torch.long)

        # Forward pass to get embeddings
        with torch.no_grad():
            embedding = self.model(input_tensor).squeeze(0)

        # Mean Pooling: Average over the word embeddings to get a fixed-size vector
        pooled_embedding = embedding.mean(dim=0)

        self.model.train()  # Switch back to train mode
        return pooled_embedding.numpy()


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
        super(WordEmbeddingModel, self).__init__()
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

# Example Tokenizer Function
def simple_tokenizer(text):
    """
    Example tokenizer function that splits text by spaces and maps tokens to arbitrary IDs.
    Replace this with a real tokenizer as needed.
    """
    tokens = text.lower().split()  # Basic split by space
    token_to_id = defaultdict(lambda: len(token_to_id))
    token_ids = [token_to_id[token] for token in tokens]
    return token_ids
