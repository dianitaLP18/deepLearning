import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from joblib import load
from statsmodels.graphics.tsaplots import plot_acf

# Load test dataset
test_data = sio.loadmat("data/Xtest.mat")["Xtest"].flatten().astype(float)

#  Load scaler 
scaler = load("models/scaler.save")

# Scale test data
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

#  Load trained LSTM model
model = load_model("models/final_lstm.h5")

# Set your tuned look_back
LOOK_BACK = 20   # change if your tuning result differs

#  Load training data to get last window
train_data = sio.loadmat("data/Xtrain.mat")["Xtrain"].flatten().astype(float)
train_scaled = scaler.transform(train_data.reshape(-1, 1)).flatten()

# Initial window = last LOOK_BACK points of training data
window = train_scaled[-LOOK_BACK:].tolist()

#  Recursive forecasting
pred_scaled = []

for _ in range(len(test_scaled)):
    x_input = np.array(window[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
    y_pred = model.predict(x_input, verbose=0)[0][0]
    pred_scaled.append(y_pred)
    window.append(y_pred)

# Inverse transform predictions
predictions = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()

#  Compute metrics

mse = mean_squared_error(test_data, predictions)
mae = mean_absolute_error(test_data, predictions)

print("Test MSE:", mse)
print("Test MAE:", mae)

#  Plot: Test vs Predicted
plt.figure(figsize=(10, 4))
plt.plot(test_data, label="True Test Data")
plt.plot(predictions, label="Predicted")
plt.title("Test vs Predicted")
plt.legend()
plt.grid()
plt.show()

#  Residual plot
residuals = test_data - predictions

plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Residuals (Test - Predicted)")
plt.grid()
plt.show()

# Autocorrelation of residuals
plot_acf(residuals, lags=40)
plt.title("Autocorrelation of Residuals")
plt.show()

# Recursive forecast plot (200 points)
plt.figure(figsize=(10, 4))
plt.plot(predictions[:200])
plt.title("Recursive Forecast (First 200 Steps)")
plt.grid()
plt.show()
