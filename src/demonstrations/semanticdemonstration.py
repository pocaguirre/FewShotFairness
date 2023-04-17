from typing import List 

from .demonstration import Demonstration

from sentence_transformers import SentenceTransformer

import pandas as pd

class SemanticDemonstration(Demonstration):

    def __init__(self, shots: int = 16) -> None:
        super().__init__(shots)

        self.embedding = SentenceTransformer('all-mpnet-base-v2')

        self.train_vectors = None

        self.test_vectors = None

        self.type = "semantic"
    
    def embed(self, train_df: pd.DataFrame, test_df: pd.DataFrame):

        self.train_vectors = self.embedding.encode(train_df['prompts'].tolist(), batch_size = 32, show_progress_bar=True)

        self.test_vectors = self.embedding.encode(test_df['prompts'].tolist(), batch_size = 32, show_progress_bar=True)

