import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
from tensorflow.keras.callbacks import History  # type: ignore


def plot_original_laser_data() -> None:
    """Load the original laser time series data and plot it."""
    # load data
    data = scipy.io.loadmat('data/Xtrain.mat')

    # extract dataset
    array_data = data['Xtrain']

    # flatten it
    X = array_data.flatten()

    plt.plot(X)
    plt.title("Laser Time Series Data")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.savefig("images/laser_plot.png")
    plt.show()


def plot_predictions_actuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
    """Plot the predicted values against the actual values on the same time axis.
    
    :param y_true: 1D array of ground-truth values, in original units.
    :param y_pred: 1D array of predicted values, same length as y_true
    :param title: title for the plot
    :param save_path: path to save the plot
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(y_true, label="Actual", color="#017442", linewidth=1.2)
    ax1.plot(y_pred, label="Predicted", color="#f4bc13", linewidth=1.2, alpha=0.85)
    ax1.set_ylabel("Laser intensity")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    residuals = y_true - y_pred
    ax2.plot(residuals, color="#534ab7", linewidth=0.9)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Time step")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_autocorrelation(series: np.ndarray, lags: int = 50) -> None:
    """Plot the autocorrelation and partial autocorrelation of the given time series.
    
    :param series: 1D array of time series values.
    :param lags: number of lags to display in the plots.
    """
    series = np.asarray(series).ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    plot_acf(series, lags=lags, ax=ax1, zero=False)
    plot_pacf(series, lags=lags, ax=ax2, zero=False, method="ywm")

    ax1.set_title("Autocorrelation Function (ACF)")
    ax2.set_title("Partial Autocorrelation Function (PACF)")

    for ax in (ax1, ax2):
        ax.set_xlabel('Lag')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/acf_pacf.png", dpi=120, bbox_inches="tight")
    plt.show()


def plot_training_history(histories: list, save_path: str) -> None:
    """Plot training/validation loss across CV folds.

    :param histories: list of Keras History objects from each CV fold.
    :param save_path: path to save the plot.
    """
    max_len = max(len(h.history["loss"]) for h in histories)

    def _pad(values, length):
        arr = np.full(length, np.nan)
        arr[:len(values)] = values
        return arr

    train = np.stack([_pad(h.history["loss"], max_len) for h in histories])
    val = np.stack([_pad(h.history["val_loss"], max_len) for h in histories])

    epochs = np.arange(1, max_len + 1)
    train_mean, train_std = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    val_mean,   val_std   = np.nanmean(val,   axis=0), np.nanstd(val,   axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_mean, label="Training Loss (mean)", color="#C942C7", linewidth=1.4)
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="#C942C7", alpha=0.2)
    ax.plot(epochs, val_mean, label="Validation Loss (mean)", color="#d8602e", linewidth=1.4)
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="#d8602e", alpha=0.2)

    # mark median best epoch across folds
    best_epochs = [int(np.argmin(h.history["val_loss"])) + 1 for h in histories]
    median_best = int(np.median(best_epochs))
    ax.axvline(median_best, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.scatter([median_best], [val_mean[median_best - 1]], color="#d8602e", zorder=5, s=40)
    ax.annotate(f"Median best epoch: {median_best}", xy=(median_best, val_mean[median_best - 1]),
                xytext=(8, 10), textcoords="offset points", fontsize=10, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"CV learning curves (mean ± std across {len(histories)} folds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
