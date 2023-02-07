import pandas as pd
import numpy as np

# Simard or Kaggle
DATASET = 'Simard'

df = pd.read_csv(f'./PreparedData/{DATASET}/paths_votes_test.csv')

X = np.array(df.iloc[:, 1:])

print(X)
print(X.shape)

np.save(f'./PreparedData/{DATASET}/RGB/votes_test', X)
# np.save('./PreparedData/Kaggle/Grey/votes', X)