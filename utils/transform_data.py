import scipy.io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_data(data: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Scales the data using MinMaxScaler."""
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    # save the scaled data
    scaled_df.to_csv('data/xtrain_scaled.csv', index=False)
    return scaled_df


def descale_data(scaled_data: pd.DataFrame, original_data: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """De-scales the data using the same MinMaxScaler."""
    scaler.fit(original_data)
    descaled_data = scaler.inverse_transform(scaled_data)
    descaled_df = pd.DataFrame(descaled_data, columns=scaled_data.columns)
    return descaled_df


data = scipy.io.loadmat('data/Xtrain.mat')

# print(data.keys())

array_data = data['Xtrain']
df = pd.DataFrame(array_data)
df.to_csv('data/xtrain.csv', index=False)

scaler = MinMaxScaler()
scaled_df = scale_data(df, scaler)

# print(data)