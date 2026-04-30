import scipy.io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data = scipy.io.loadmat('data/Xtrain.mat')

# print(data.keys())

array_data = data['Xtrain']
df = pd.DataFrame(array_data)
df.to_csv('data/xtrain.csv', index=False)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df.to_csv('data/xtrain_scaled.csv', index=False)

# print(data)