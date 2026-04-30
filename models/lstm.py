from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

"""
    Long Short Term Memory (LSTM) Model
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
                  return_sequence=True), 
                  Dropout(self.dropout), 
                  LSTM(32), 
                  Dropout(self.dropout), 
                  Dense(1)
                ])
        return model
