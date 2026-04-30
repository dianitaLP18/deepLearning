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


def plot_training_history(history: History) -> None:
    """Plot the training loss curve from history of the model."""
    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, h['loss'], label='Training Loss', color="#C942C7", linewidth=1.4)
    if "val_loss" in h:
        ax.plot(epochs, h["val_loss"], label="Validation loss",
                color="#d8602e", linewidth=1.4)

        # mark the best validation epoch
        best_epoch = int(np.argmin(h["val_loss"])) + 1
        best_val = min(h["val_loss"])
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.scatter([best_epoch], [best_val], color="#d8602e", zorder=5, s=40)
        ax.annotate(f"Best: epoch {best_epoch}",
                    xy=(best_epoch, best_val),
                    xytext=(8, 10), textcoords="offset points",
                    fontsize=10, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Learning curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/training_history.png", dpi=120, bbox_inches="tight")
    plt.show()
