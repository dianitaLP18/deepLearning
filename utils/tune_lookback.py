import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from models.lstm import LSTMModel   

def create_dataset(series, look_back):
    X, Y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        Y.append(series[i + look_back])
    return np.array(X), np.array(Y)


data = pd.read_csv("data/xtrain.csv", header=None).values.flatten().astype(float)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

look_backs = [5, 10, 20, 30, 40, 50]
results = []
for lb in look_backs:
    print(f"\n Testing look_back = {lb}")

    X, Y = create_dataset(data_scaled, lb)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )

    model = LSTMModel(k=lb, dropout=0.2)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=False,
        verbose=0,
        callbacks=[es]
    )

    val_mse = history.history["val_loss"][-1]
    val_mae = history.history["val_mae"][-1]

    print(f"look_back={lb} → MSE={val_mse:.6f}, MAE={val_mae:.6f}")
    results.append((lb, val_mse, val_mae))

# the best look_back is the one with the lowest validation MSE
best = min(results, key=lambda x: x[1])
print("\n BEST LOOK-BACK WINDOW:")
print(f"look_back = {best[0]},  MSE = {best[1]:.6f},  MAE = {best[2]:.6f}")
