import pandas as pd
import numpy as np

# Simard or Kaggle
DATASET = 'Kaggle'
NUM_CLASSES = 2

# df = pd.read_csv(f'./PreparedData/{DATASET}/paths_votes_{NUM_CLASSES}.csv')
df = pd.read_csv(f'./PreparedData/{DATASET}/paths_votes_deepcaps.csv')

X = np.array(df.iloc[:, 1:])
# X = df['ng'].tolist()

print(X)
# print(X.shape)
print(len(X))
# np.save(f'./PreparedData/{DATASET}/Grey/votes_{NUM_CLASSES}', X)
np.save(f'./PreparedData/{DATASET}/Grey/votes_deepcaps_{DATASET}', X)