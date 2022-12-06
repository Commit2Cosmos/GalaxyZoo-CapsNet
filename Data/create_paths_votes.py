import pandas as pd
import numpy as np


ids = pd.read_csv('gz2_filename_mapping.csv', usecols=[0,2])
ids['asset_id'] = ids.iloc[:,1].map(lambda x: f'{x}.jpg')
ids = ids.rename(columns={'asset_id' : 'paths'})
votes = pd.read_csv('gz2_vote_fractions.csv', usecols=[0,18])
votes.columns = ['objid', 'votes']

merged = pd.merge(left=ids, right=votes, left_on='objid', right_on='objid')
merged.drop('objid', axis=1, inplace=True) 

merged.to_csv('paths_votes.csv', encoding='utf-8')