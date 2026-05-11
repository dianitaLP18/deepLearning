from xml.parsers.expat import model

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from joblib import dump
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from utils.forecasting import plot_recursive_forecast, run_recursive_forecast
from utils.transform_data import prepare_data, get_cv_folds, make_sequences
from models.lstm import LSTMModel
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


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
               *, k: int, max_epochs: int, patience: int,
               fixed_epochs: int | None = None) -> tuple[LSTMModel, list[float]]:
    """Train an LSTM model with early stopping and return the trained model and training history."""
    model = LSTMModel(dropout=0.2, k=k)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    model.summary()

    if fixed_epochs is not None:
        history = model.fit(
            X_train, y_train, 
            epochs=fixed_epochs, batch_size=32,
            shuffle=False, verbose=2
        )
    else:
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs, batch_size=32,
            shuffle=False, callbacks=[es], verbose=2
        )

    return model, history


"""
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
"""


if __name__ == "__main__":
    # set the values for the hyperparameters
    K = 20
    N_SPLITS = 5
    MAX_EPOCHS = 100
    PATIENCE = 15

    _set_seeds(42)

    dev_scaled, X_test, y_test, scaler = prepare_data(test_fraction=0.2, k=K)
    dump(scaler, "models/scaler.save")

    raw_train_series = scaler.inverse_transform(dev_scaled.reshape(-1, 1)).ravel()
    plot_autocorrelation(raw_train_series, lags=60)

    # cross-validation
    cv_results = []
    cv_histories = []
    for fold_idx, X_train, y_train, X_val, y_val in get_cv_folds(dev_scaled, n_splits=N_SPLITS, k=K):
        tf.keras.backend.clear_session()
        _set_seeds(42)

        model, history = train_lstm(
            X_train, y_train,X_val, y_val,
            patience=PATIENCE, max_epochs=MAX_EPOCHS, k=K
        )
        cv_histories.append(history)

        # per fold metrics
        val_pred = scaler.inverse_transform(model.predict(X_val)).ravel()
        val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        val_mae = mean_absolute_error(val_true, val_pred)
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1

        cv_results.append({
            "fold": fold_idx, "val_rmse": val_rmse,
            "val_mae": val_mae, "best_epoch": best_epoch,
        })
        print(f"-> fold {fold_idx}: val RMSE={val_rmse:.2f}, "
              f"val MAE={val_mae:.2f}, best epoch={best_epoch}")

    plot_training_history(cv_histories, save_path="images/cv_training_history.png")

    rmses = np.array([r["val_rmse"] for r in cv_results])
    maes = np.array([r["val_mae"]  for r in cv_results])
    best_epochs = [r["best_epoch"] for r in cv_results]
    print("\n=== CV summary ===")
    print(f"Val RMSE: {rmses.mean():.2f} ± {rmses.std():.2f}")
    print(f"Val MAE:  {maes.mean():.2f} ± {maes.std():.2f}")
    print(f"Best epochs: {best_epochs}")

    # final model on dev data
    final_epochs = int(np.median(best_epochs))
    print(f"\nRetraining final model on full dev set for {final_epochs} epochs...")

    tf.keras.backend.clear_session()
    _set_seeds(42)
    X_dev, y_dev = make_sequences(dev_scaled, k=K)

    final_model, _ = train_lstm(
        X_dev, y_dev,
        k=K, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        fixed_epochs=final_epochs
    )
    final_model.model.save("models/final_lstm.h5")

    # evaluation
    lstm_metrics = _evaluate(
        final_model, X_dev, y_dev, X_test, y_test, scaler, name="LSTM",
    )

    print("\n=== Final results ===")
    print(f"{lstm_metrics['name']:6s} | params={lstm_metrics['n_params']:>7d} | "
          f"test RMSE={lstm_metrics['test_rmse']:.2f} | "
          f"test MAE={lstm_metrics['test_mae']:.2f}")
    print(f"(CV val RMSE for comparison: {rmses.mean():.2f} ± {rmses.std():.2f})")
