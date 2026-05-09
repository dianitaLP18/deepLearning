import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


class GRUModel:
    def __init__(self, dropout: float, k: int,
                 units_1: int = 64, units_2: int = 32) -> None:
        """
        :param dropout: dropout rate for regularization.
        :param k: lookback window length for input sequences.
        :param units_1: hidden units in the first GRU layer.
        :param units_2: hidden units in the second GRU layer.
        """
        self.dropout = dropout
        self.k = k
        self.units_1 = units_1
        self.units_2 = units_2
        self.model = None

    def create(self) -> Sequential:
        return Sequential([
            GRU(self.units_1, input_shape=(self.k, 1), return_sequences=True),
            Dropout(self.dropout),
            GRU(self.units_2),
            Dropout(self.dropout),
            Dense(1),
        ])

    def compile(self, **kwargs) -> None:
        self.model = self.create()
        self.model.compile(**kwargs)

    def summary(self) -> None:
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        self.model.summary()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        return self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        return self.model.predict(X)
