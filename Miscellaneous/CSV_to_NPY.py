import pandas as pd
import numpy as np

df = pd.read_csv('./InitData/Kaggle/labels.csv')

X = np.array(df.iloc[:, 1:])

print(X)
print(X.shape)

np.save('./PreparedData/Kaggle/votes', X)

#print(X[60891])