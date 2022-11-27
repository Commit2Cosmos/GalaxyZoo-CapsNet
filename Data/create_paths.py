"""
Input:

images_gz2 (galaxy images; dowload from Zenodo),
gz2_vote_fractions.csv (only need 1st (dr7objid) and 19th (t01_smooth_or_features_a02_features_or_disk_weighted_fraction) columns; download from GZ2 Table 1),
gz2_filename_mapping.csv (only need 1st (objid) and 3rd (asset_id) columns)

Todo:

Create a csv file with relative file paths of images (1st column) and vote fractions (2nd column) (./file_paths.csv) for Dataloader.py

Method:

For each image create relative path (append asset_id at the end) and find its vote fraction (by matching objid to dr7objid)
"""


import csv
import pandas as pd
# import numpy as np

asset_id = pd.read_csv('./gz2_filename_mapping.csv', usecols=[2])

# def add_file_name(name='./images_gz2'):
    

# print(asset_id)


# header = ['relative_path', 'vote_fraction']


# with open('./image_paths.csv', 'w') as file:
#     writer = csv.writer(file)

