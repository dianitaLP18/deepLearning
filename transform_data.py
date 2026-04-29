import scipy.io
import pandas as pd

data = scipy.io.loadmat('Xtrain.mat')

print(data.keys())

array_data = data['Xtrain']
df = pd.DataFrame(array_data)
df.to_csv('xtrain.csv', index=False)

print(data)