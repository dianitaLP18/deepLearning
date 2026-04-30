from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

"""
    Long Short Term Memory (LSTM) Model

to add:
- refactor the class functions to be modular and reusable, as well as add type hints and error handling
- experiment with different values for the layers and compare results
- add more metrics for evaluation
"""

class LSTMModel:
    
    def __init__(self, dropout, k):
        # hyperparams
        self.dropout = dropout
        self.k = k
        self.model = None

    def create(self):
        model = Sequential([
            LSTM(64, input_shape=(self.k, 1),
                  return_sequences=True), 
                  Dropout(self.dropout), 
                  LSTM(32), 
                  Dropout(self.dropout), 
                  Dense(1)
                ])
        return model

    def compile(self, **kwargs):
        self.model = self.create()
        self.model.compile(**kwargs)

    def summary(self):
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        self.model.summary()

    def fit(self, X, y, **kwargs):
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not created yet. Call compile() first.")
        return self.model.predict(X)
