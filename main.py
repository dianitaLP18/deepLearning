import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from joblib import dump
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.forecasting import plot_recursive_forecast, run_recursive_forecast
from utils.transform_data import prepare_data
from models.lstm import LSTMModel
from models.gru import GRUModel
from utils.plotting_helpers import plot_predictions_actuals, plot_autocorrelation, plot_training_history


def _set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def _evaluate(model, X_train, y_train, X_test, y_test, scaler, name: str) -> dict:
    """Shared evaluation: predictions, metrics, plots, recursive forecast.
    
    :param model: trained model to evaluate.
    :param X_train: training input data.
    :param y_train: training target data.
    :param X_test: testing input data.
    :param y_test: testing target data.
    :param scaler: fitted scaler to inverse transform predictions and targets.
    :param name: name of the model.
    """
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    trainY_real = scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    testY_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    trainPred_real = scaler.inverse_transform(trainPredict).ravel()
    testPred_real = scaler.inverse_transform(testPredict).ravel()

    train_rmse = np.sqrt(mean_squared_error(trainY_real, trainPred_real))
    test_rmse = np.sqrt(mean_squared_error(testY_real, testPred_real))
    train_mae = mean_absolute_error(trainY_real, trainPred_real)
    test_mae = mean_absolute_error(testY_real, testPred_real)

    print(f"[{name}] Train RMSE: {train_rmse:.2f}  MAE: {train_mae:.2f}")
    print(f"[{name}] Test  RMSE: {test_rmse:.2f}  MAE: {test_mae:.2f}")
    print(f"[{name}] Test/Train RMSE ratio: {test_rmse / train_rmse:.2f}")

    plot_predictions_actuals(
        trainY_real, trainPred_real,
        title=f"{name} Train: Predicted vs Actual",
        save_path=f"images/{name.lower()}_train_pred.png",
    )
    plot_predictions_actuals(
        testY_real, testPred_real,
        title=f"{name} Test: Predicted vs Actual",
        save_path=f"images/{name.lower()}_test_pred.png",
    )

    forecast_scaled = run_recursive_forecast(model, X_test[-1], steps=200)
    recursive_real = scaler.inverse_transform(forecast_scaled).ravel()
    plot_recursive_forecast(
        testY_real, recursive_real,
        save_path=f"images/{name.lower()}_recursive_forecast.png",
    )

    return {
        "name": name,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "train_mae": train_mae, "test_mae": test_mae,
        "n_params": model.model.count_params(),
    }


def train_lstm(X_train, y_train, X_test, y_test, scaler) -> dict:
    _set_seeds(42)
    model = LSTMModel(dropout=0.2, k=20)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_split=0.2, shuffle=False, verbose=2,
    )
    model.model.save("models/final_lstm.h5")
    plot_training_history(history, save_path="images/lstm_training_history.png")

    return _evaluate(model, X_train, y_train, X_test, y_test, scaler, name="LSTM")


def train_gru(X_train, y_train, X_test, y_test, scaler,
              units_1: int = 64, units_2: int = 32) -> dict:
    _set_seeds(42)
    model = GRUModel(dropout=0.2, k=20, units_1=units_1, units_2=units_2)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_split=0.2, shuffle=False, verbose=2,
    )
    model.model.save("models/final_gru.h5")
    plot_training_history(history, save_path="images/gru_training_history.png")

    return _evaluate(model, X_train, y_train, X_test, y_test, scaler, name="GRU")


if __name__ == "__main__":
    _set_seeds(42)

    X_train, y_train, X_test, y_test, scaler = prepare_data()
    dump(scaler, "models/scaler.save")

    raw_train_series = scaler.inverse_transform(
        X_train[:, :, 0].reshape(-1, 1)
    ).ravel()
    plot_autocorrelation(raw_train_series, lags=60)

    lstm_metrics = train_lstm(X_train, y_train, X_test, y_test, scaler)
    gru_metrics = train_gru(X_train, y_train, X_test, y_test, scaler)

    print("\n=== Comparison ===")
    for m in (lstm_metrics, gru_metrics):
        print(f"{m['name']:6s} | params={m['n_params']:>7d} | "
              f"test RMSE={m['test_rmse']:.2f} | test MAE={m['test_mae']:.2f}")
