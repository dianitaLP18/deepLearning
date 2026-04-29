import scipy.io
import matplotlib.pyplot as plt

# Load data
data = scipy.io.loadmat('Xtrain.mat')

# Extract dataset
array_data = data['Xtrain']

# Flatten it
X = array_data.flatten()
# Plot
plt.plot(X)
plt.title("Laser Time Series Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
# Save image
plt.savefig("laser_plot.png")
# Show plot
plt.show()