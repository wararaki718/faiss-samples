import faiss
import numpy as np

from .base import BaseANN


class LinearL2(BaseANN):
    def __init__(self, dim: int):
        self._index = faiss.IndexFlatL2(dim)

    def add(self, x: np.ndarray):
        self._index.add(x)
    
    def search(self, x: np.ndarray, k: int=10) -> tuple:
        distances, indices = self._index.search(x, k)
        return distances, indices
