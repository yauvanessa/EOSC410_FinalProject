import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## EOSC410 Project
## Authors: Vanessa Yau + Christina Rutherford

# import avalanche data for sea-to-sky region
avy_risk = pd.read_csv('sea-to-sky_Jan_to_Apr_2012_to_2017.csv')

# import era5 2m temp data
temp_2m = np.load('Sea-to-Sky_Jan2017_Temp2m.npy')
snowdep = np.load('Sea-to-Sky_Jan2017_SnowDepth.npy')

# import era5 coordinates
coord = np.load('Sea-to-Sky_Jan2017_LatLon.npy', allow_pickle=True)
lat = coord[0]
lon = coord[1]



