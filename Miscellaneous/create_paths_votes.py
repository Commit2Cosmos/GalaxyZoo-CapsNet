import pandas as pd
import os


# ids = pd.read_csv('gz2_filename_mapping.csv', usecols=[0,2])
# ids['asset_id'] = ids.iloc[:,1].map(lambda x: f'{x}.jpg')
# ids = ids.rename(columns={'asset_id' : 'paths'})
# votes = pd.read_csv('gz2_vote_fractions.csv', usecols=[0,18])
# votes.columns = ['objid', 'votes']

# merged = pd.merge(left=ids, right=votes, left_on='objid', right_on='objid')
# merged.drop('objid', axis=1, inplace=True)



# merged = pd.read_csv('./InitData/Kaggle/labels_train.csv', usecols=[0,1,2])
# merged = pd.read_csv('./InitData/Kaggle/labels_train.csv', usecols=[0,1])
merged = pd.read_csv('./InitData/Kaggle/labels_train.csv')

merged['GalaxyID'] = merged.iloc[:,0].map(lambda x:f'{x}.jpg')

# filenames = os.listdir('./InitData/Kaggle/images_train')
filenames = os.listdir('../Data/images_train_kaggle')
filenames = pd.DataFrame(filenames, columns=['filenames'])
merged = pd.merge(left=merged, right=filenames, left_on='GalaxyID', right_on='filenames')
merged.drop('filenames', axis=1, inplace=True)

# for binary labels
# merged['Class1.2'] = merged.iloc[:,1].map(lambda x:round(1-x, 6))



print(merged)
merged.to_csv('./PreparedData/Kaggle/paths_votes.csv', encoding='utf-8', index=False)