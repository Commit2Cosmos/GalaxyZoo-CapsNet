import pandas as pd
import numpy as np

# Simard or Kaggle
DATASET = 'Simard'
NUM_CLASSES = 6

df = pd.read_csv(f'./PreparedData/{DATASET}/paths_votes_{NUM_CLASSES}.csv')

X = np.array(df.iloc[:, 1:])

print(X)
print(X.shape)

np.save(f'./PreparedData/{DATASET}/RGB/votes_{NUM_CLASSES}', X)
# np.save('./PreparedData/Kaggle/Grey/votes', X)