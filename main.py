from utils.transform_data import prepare_data
from models.lstm import LSTMModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.plotting_helpers import plot_predictions_actuals, plot_autocorrelation, plot_training_history


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, y_train, X_test, y_test, scaler = prepare_data()

    raw_train_series = scaler.inverse_transform(
        X_train[:, :, 0].reshape(-1, 1)
    ).ravel()
    plot_autocorrelation(raw_train_series, lags=60)

    model = LSTMModel(dropout=0.2, k=20)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mae'])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_split=0.2, shuffle=False,
        verbose=2,
    )

    trainPredict = model.predict(X_train)
    testPredict  = model.predict(X_test)

    trainY_real = scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    testY_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    trainPred_real = scaler.inverse_transform(trainPredict).ravel()
    testPred_real = scaler.inverse_transform(testPredict).ravel()

    train_rmse = np.sqrt(mean_squared_error(trainY_real, trainPred_real))
    test_rmse = np.sqrt(mean_squared_error(testY_real, testPred_real))
    train_mae = mean_absolute_error(trainY_real, trainPred_real)
    test_mae = mean_absolute_error(testY_real, testPred_real)

    print(f"Train RMSE: {train_rmse:.2f}  MAE: {train_mae:.2f}")
    print(f"Test  RMSE: {test_rmse:.2f}  MAE: {test_mae:.2f}")
    print(f"Test/Train RMSE ratio: {test_rmse / train_rmse:.2f}")

    plot_predictions_actuals(trainY_real, trainPred_real, title="Train: Predicted vs Actual",
                             save_path="images/train_pred.png")
    plot_predictions_actuals(testY_real, testPred_real, title="Test: Predicted vs Actual",
                             save_path="images/test_pred.png")
