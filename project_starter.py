import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

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

#Current Temp and Snow Depth data are (31, 7, 13). I averaged them below to be (31, 13), but this still doesn't give 1 mean line on the
#graph when plotting. I could average from (31, 7, 13) to (31, 7) and then (31,) but then it requires to reshape it to (31,1) and the 
#variance doesn't look quite right. 

#averaging temp from (31, 7, 13) to (31, 13)
temp_mean = np.mean(temp_2m,axis=1)
np.shape(temp_mean)
#averaging snow depth from(31, 7, 13) to (31, 13)
snowdep_mean = np.mean(snowdep,axis=1)
np.shape(snowdep_mean)

#visualize current data
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(temp_2m[:,1])
plt.plot(temp_mean, color = 'k',label = 'mean', linewidth = 3) #this isn't just 1 line, I think it's 13
plt.xlabel('Days of January')
plt.ylabel('Temp (K)')
plt.title('Temperature at noon everyday in January')
plt.subplot(1,2,2)
plt.plot(snowdep[:,1])
plt.plot(snowdep_mean, color = 'k',label = 'mean', linewidth = 3) #this isn't just 1 line, I think it's 13
plt.xlabel('Days of January')
plt.ylabel('Snow Depth')
plt.title('Snow depth at noon everyday in January')
plt.show()



#define RMSE as a function, since we'll use this in the NN model 
def rmse(target,prediction):
    return(np.sqrt(((target - prediction)**2).sum()/len(target)))
  
#normalize data
temp_norm = (temp_mean - temp_mean.mean())/temp_mean.std()
snowdep_norm = (snowdep_mean - snowdep_mean.mean())/snowdep_mean.std()

#visualize normalized data
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(temp_norm)
#plt.plot(temp_mean, color = 'k',label = 'mean', linewidth = 3)
plt.xlabel('Days of January')
plt.ylabel('Temp (K)')
plt.title('Normalized Temp data through January')
plt.subplot(1,2,2)
plt.plot(snowdep_norm)
#plt.plot(snowdep_mean, color = 'k',label = 'mean', linewidth = 3)
plt.xlabel('Days of January')
plt.ylabel('Snow Depth')
plt.title('Normalized Snow depth data through January')
plt.show()

#first do PCA for TEMP data, then use PCs as predictors
n_modesT = np.min(np.shape(temp_norm))
pcaT = PCA(n_components = n_modesT)
PCsT = pcaT.fit_transform(temp_norm)
eigvecsT = pcaT.components_
fracVarT = pcaT.explained_variance_ratio_

#first do PCA for SNOW data, then use PCs as predictors
n_modesS = np.min(np.shape(snowdep_norm))
pcaS = PCA(n_components = n_modesS)
PCsS = pcaS.fit_transform(snowdep_norm)
eigvecsS = pcaS.components_
fracVarS = pcaS.explained_variance_ratio_

#plot fraction of variance explained by each mode
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(range(len(fracVarT)),fracVarT)
plt.xlabel('Mode Number')
plt.ylabel('Fraction Variance Explained')
plt.title('Variance Explained by All Modes for Temp')
plt.subplot(1,2,2)
plt.scatter(range(len(fracVarS)),fracVarS)
plt.xlabel('Mode Number')
plt.ylabel('Fraction Variance Explained')
plt.title('Variance Explained by All Modes for Snow Depth')
plt.tight_layout()
plt.show()

#ready for MLP, see MLP file
