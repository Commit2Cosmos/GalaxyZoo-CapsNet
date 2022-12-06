import pandas as pd
import numpy as np

df = pd.read_csv('../Data/paths_votes.csv')

X = np.array(df.iloc[:, 1:])

print(X)
print(X.shape)

np.save('../ReadyFile/votes', X)

#print(X[60891])