import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def run_recursive_forecast(model, start_window, steps=200):
    """
    Performs recursive forecasting using the trained model.
    """
    # Ensure starting window is 3D: (1, Time Steps, Features)
    current_batch = start_window.reshape(1, start_window.shape[0], start_window.shape[1])
    current_batch = tf.convert_to_tensor(current_batch, dtype=tf.float32)

    recursive_predictions = []

    for _ in range(steps):
        # Predict
        current_pred = model.model(current_batch, training=False)
        pred_scalar = current_pred[0, 0].numpy()
        recursive_predictions.append(pred_scalar)
        
        # Shift and Update the input window
        temp_np = current_batch.numpy()
        new_input = np.roll(temp_np, -1, axis=1)
        new_input[0, -1, 0] = pred_scalar
        
        # Re-wrap
        current_batch = tf.convert_to_tensor(new_input, dtype=tf.float32)

    return np.array(recursive_predictions).reshape(-1, 1)

def plot_recursive_forecast(actual_data, forecast_data, save_path="images/forecast.png"):
    """
    Handles the visualization of the forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data, label="Actual Test Data", color='blue')
    
    # Calculate x-axis for forecast starting where actual data ends
    x_axis_forecast = np.arange(len(actual_data), len(actual_data) + len(forecast_data))
    
    plt.plot(x_axis_forecast, forecast_data, 
             label=f"Recursive Forecast ({len(forecast_data)} pts)", 
             linestyle='--', color='red')
    
    plt.title("Recursive Out-of-Sample Forecast")
    plt.legend()
    plt.savefig(save_path)
    plt.show()