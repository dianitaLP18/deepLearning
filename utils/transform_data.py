import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view


def make_sequences(series: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for the model.

    :param series: 1D array of shape (n,).
    :param k: lookback window length.
    :return: (X, y) with shapes (n-k, k, 1) and (n-k,).
    """
    windows = sliding_window_view(series, window_shape=k + 1)
    X, y = windows[:, :-1], windows[:, -1]
    return X.reshape(-1, k, 1), y


if __name__ == "__main__":
    # load
    raw = scipy.io.loadmat('data/Xtrain.mat')['Xtrain'].ravel().astype(np.float32)

    # chronological split for timeseries data
    n = len(raw)
    split = int(0.8 * n)
    train_raw, test_raw = raw[:split], raw[split:]

    # fit scaler on training and transform both
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).ravel()
    test_scaled  = scaler.transform(test_raw.reshape(-1, 1)).ravel()

    # build sequences, the shape changes to 3D
    k = 20
    X_train, y_train = make_sequences(train_scaled, k)
    X_test, y_test = make_sequences(test_scaled,  k)

    print(X_train.shape, y_train.shape)
    print(X_test.shape,  y_test.shape)

    # inverting measurements
    # y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
