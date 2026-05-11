import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from numpy.lib.stride_tricks import sliding_window_view
from typing import Generator


def make_sequences(series: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for the model.

    :param series: 1D array of shape (n,).
    :param k: lookback window length.
    :return: (X, y) with shapes (n-k, k, 1) and (n-k,).
    """
    windows = sliding_window_view(series, window_shape=k + 1)
    X, y = windows[:, :-1], windows[:, -1]
    return X.reshape(-1, k, 1), y


def prepare_data(test_fraction: float, k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Load the laser series, held out a final test set, and scale.

    :param test_frac: fraction of the series held out as the final test set.
    :param k: lookback window length.
    :return: (X_train, y_train, X_test, y_test, scaler)
    """
    raw = scipy.io.loadmat('data/Xtrain.mat')['Xtrain'].ravel().astype(np.float32)

    # chronological split for timeseries data
    n = len(raw)
    split = int((1 - test_fraction) * n)
    dev_raw, test_raw = raw[:split], raw[split:]

    # fit scaler on development set and transform both
    scaler = MinMaxScaler()
    dev_scaled = scaler.fit_transform(dev_raw.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).ravel()

    X_test, y_test = make_sequences(test_scaled,  k)

    print(f"Development set: {len(dev_raw)} samples.")
    print(f"Testing set: {len(test_raw)} samples. | Test sequences: {X_test.shape[0]}")

    return dev_scaled, X_test, y_test, scaler


def get_cv_folds(
        dev_scaled: np.ndarray,
        n_splits: int, k: int) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Yield expanding-window CV folds with sequences built per fold.

    This helps in creating more realistic validation folds for time series data.

    :param dev_scaled: 1D scaled development series.
    :param k: lookback window length.
    :param n_splits: number of CV folds.
    :yield: (fold_idx, X_train, y_train, X_val, y_val)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(dev_scaled)):
        train_series = dev_scaled[train_idx]
        val_series = dev_scaled[val_idx]

        if len(train_series) <= k or len(val_series) <= k:
            print(f"Fold {fold_idx}: too short for k={k}. Skipping.")
            continue

        X_train, y_train = make_sequences(train_series, k)
        X_val, y_val = make_sequences(val_series, k)

        print(f"Fold {fold_idx}: train idx [{train_idx[0]}:{train_idx[-1]+1}]"
              f"({X_train.shape[0]} sequences)"
              f" val idx [{val_idx[0]}:{val_idx[-1]+1}] ({X_val.shape[0]} sequences)")
        yield fold_idx, X_train, y_train, X_val, y_val
