"""Main module for the LSTM model."""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LSTMModel:
    
    def __init__(self, dropout: float, k: int) -> None:
        """Initialize the LSTM model with given hyperparameters.
        
        :param dropout: dropout rate for regularization.
        :param k: lookback window length for input sequences.
        """
        # hyperparams
        self.dropout = dropout
        self.k = k
        self.model = None

    def create(self) -> Sequential:
        """Build the LSTM model architecture."""
        model = Sequential([
            LSTM(64, input_shape=(self.k, 1),
                  return_sequences=True), 
                  Dropout(self.dropout), 
                  LSTM(32), 
                  Dropout(self.dropout), 
                  Dense(1)
        ])
        return model

    def compile(self, **kwargs) -> None:
        """Compile the model with given loss and optimizer."""
        self.model = self.create()
        self.model.compile(**kwargs)

    def summary(self) -> None:
        """Print the model architecture summary."""
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        self.model.summary()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model on the given data.
        
        :param X: input features of shape (n_samples, k, 1).
        :param y: target values of shape (n_samples,).
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        :param X: input features of shape (n_samples, k, 1).
        :return: predicted values of shape (n_samples,).
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        return self.model.predict(X)
