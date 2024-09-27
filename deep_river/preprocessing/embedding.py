from typing import Dict

import numpy as np
import river.base as base
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingTransformer(base.Transformer):
    """
    A class for processing text input and generating embeddings using a pre-trained transformer model.

    This class leverages the Hugging Face Transformers library to load a specified model and its tokenizer.
    It can concatenate text from input dictionaries, tokenize the text, and generate embeddings, which can
    be utilized for various natural language processing tasks such as semantic similarity, clustering, etc.

    Attributes:
        model_name (str): The name of the pre-trained transformer model to be used.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initializes the EmbeddingTransformer with the specified model and tokenization length.

        Args:
            model_name (str): The name of the pre-trained transformer model to use (default is "bert-base-uncased").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _concat_dict_values(self, x: Dict) -> str:
        """
        Helper function to concatenate all values from the input dictionary into a single string.

        Args:
            x (dict): A dictionary containing values to be concatenated.

        Returns:
            str: A single string composed of all values in the dictionary, separated by spaces.
        """
        return " ".join(str(value) for value in x.values())

    def transform_one(self, x: Dict) -> Dict:
        """
        Processes a single input dictionary to generate a text embedding.

        This method concatenates all values from the input dictionary, tokenizes the resulting text,
        and generates embeddings using the pre-trained transformer model. It returns a dictionary with
        the embedding values indexed starting from 1.

        Args:
            x (Dict): A dictionary containing text data for embedding generation.

        Returns:
            Dict: A dictionary containing the generated embedding values indexed from 1.

        Example:
            >>> # Example usage of TextEmbeddingProcessor
            >>> processor = EmbeddingTransformer(model_name="bert-base-uncased")
            >>> input_data = {
            ...     "title": "Transformers in NLP",
            ...     "description": "An overview of transformer models in natural language processing."
            ... }
            >>> embeddings = processor.transform_one(input_data)

        """
        # Concatenate all values from the input dictionary
        concatenated_text = self._concat_dict_values(x)

        # Tokenize the concatenated text
        inputs = self.tokenizer(
            concatenated_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Generate embeddings using the pre-trained model
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy()
        return dict(enumerate(embedding, 1))
