import pandas as pd
import os
import math


#* MATCH SDSS & IMAGE IDS FOR SIMARD
ids = pd.read_csv(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\Data\\match_test_kaggle.csv', usecols=[0,1])


#* match ids to previous dataset by usage (either this or ids_required)
# ids = pd.read_csv(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\Data\\match_test_kaggle.csv')
# ids = ids[ids['Usage'] == 'training']
# ids.drop('Usage', axis=1, inplace=True)
# print(ids)


#* match kaggle ids to previous dataset
ids_required = pd.read_csv('./InitData/Kaggle/labels_train.csv', usecols=[0])
ids = pd.merge(left=ids, right=ids_required, on='GalaxyID', how='inner')
# print(ids)


#* load votes
votes = pd.read_csv(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\Data\\simard_nfree.csv', 
usecols=['objID', 'Scale', 'rg2d', 'Rhlr', 'e', 'phi', 'S2r', 'ng'])


#* match kaggle & SDSS ids
votes = pd.merge(left=votes, right=ids, left_on='objID', right_on='dr7objid', how='inner')
votes.drop(['objID', 'dr7objid'], axis=1, inplace=True)
cols = votes.columns.tolist()
cols = cols[-1:] + cols[:-1]
votes = votes[cols]


#* convert Rhlr to arsec
votes['Rhlr'] = votes['Rhlr'] * votes['Scale']
votes.drop('Scale', axis=1, inplace=True)


#* convert phi to rads
votes['phi'] = votes['phi'] * math.pi/180


#* match labels to images
votes['GalaxyID'] = votes.iloc[:,0].map(lambda x:f'{x}.jpg')
filenames = os.listdir('../Data/images_train_kaggle')
filenames = pd.DataFrame(filenames, columns=['filenames'])
votes = pd.merge(left=votes, right=filenames, left_on='GalaxyID', right_on='filenames')
votes.drop('filenames', axis=1, inplace=True)

votes = votes.iloc[:200]

print(votes)

#* save csv
# votes.to_csv('./PreparedData/Simard/paths_votes_test.csv', encoding='utf-8', index=False)