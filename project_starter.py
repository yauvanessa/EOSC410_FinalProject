import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from functions import *

## EOSC410 Project
## Authors: Vanessa Yau + Christina Rutherford
saveIt = 0 # set to 1 to save all figs

# import avalanche data for sea-to-sky region
avy_risk = pd.read_csv('sea-to-sky_Jan2017.csv') #TODO do we need to normalize this data?
avy_date = avy_risk['Date']
below_tree = avy_risk['Below Treeline']
tree = avy_risk['Treeline']
above_tree = avy_risk['Above Treeline']

# import era5 2m temp data
temp_2m = np.load('Sea-to-Sky_Jan2017_Temp2m.npy')
snowdep = np.load('Sea-to-Sky_Jan2017_SnowDepth.npy')

# import era5 coordinates
coord = np.load('Sea-to-Sky_Jan2017_LatLon.npy', allow_pickle=True)
lat = coord[0]
lon = coord[1]

#Current Temp and Snow Depth data are (31, 7, 13). I averaged them below to be (31,) and eventually reshape them to (31,1), but this still 
#causes the variance to only be 1 mode in the center. Therefore maybe we don't do PCA 

# averaging TEMP from (31,7,13) to (31, 7)
temp_station_mean = np.mean(temp_2m,axis=2)
# averaging TEMP from (31,7) to (31,1)
temp_mean = np.mean(temp_station_mean,axis=1)
# averaging SNOW from (31, 7, 13) to (31, 7)
snowdep_station_mean = np.mean(snowdep,axis=2)
# averaging SNOW from (31, 7) to (31, 1)
snowdep_mean = np.mean(snowdep_station_mean,axis=1)

# visualize current temp and snow depth data
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(temp_2m[:,1])
plt.plot(temp_mean, color = 'k',label = 'mean', linewidth = 3)
plt.xlabel('Day in January')
plt.ylabel('Temp (K)')
plt.title('Daily Temperature at noon, January 2017')

plt.subplot(1,2,2)
plt.plot(snowdep[:,1])
plt.plot(snowdep_mean, color = 'k',label = 'mean', linewidth = 3)
plt.xlabel('Day in January')
plt.ylabel('Snow Depth (m of water equivalent)')
plt.title('Daily Snow Depth at noon, January 2017')

plt.tight_layout()
# plt.show()
if saveIt == 1:
    plt.savefig('fig_Jan2017_temp_snowdep.png')
  
# normalize data
temp_norm = (temp_mean - temp_mean.mean())/temp_mean.std()
snowdep_norm = (snowdep_mean - snowdep_mean.mean())/snowdep_mean.std()
# reshape data from (31,) to (31, 1)
temp_norm=temp_norm.reshape(-1, 1)
snowdep_norm=snowdep_norm.reshape(-1, 1)

#visualize normalized data
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(temp_norm)
#plt.plot(temp_mean, color = 'k',label = 'mean', linewidth = 3)
plt.xlabel('Day in January')
plt.ylabel('Temp (K)')
plt.title('Normalized Temp January 2017')

plt.subplot(1,2,2)
plt.plot(snowdep_norm)
#plt.plot(snowdep_mean, color = 'k',label = 'mean', linewidth = 3)
plt.xlabel('Day in January')
plt.ylabel('Snow Depth (m of water equivalent)')
plt.title('Normalized Snow Depth January 2017')

plt.tight_layout()
# plt.show()
if saveIt == 1:
    plt.savefig('fig_Jan2017_normalized_temp_snowdep.png')

# plot avy risk data
days = np.linspace(1,31,31)
plt.figure(figsize=(6,6))
plt.scatter(days,above_tree,color='blue',label='Above Treeline',alpha=0.5)
plt.scatter(days,tree,color='green',label='Treeline',alpha=0.5)
plt.scatter(days,below_tree,color='orange',label='Below Treeline',alpha=0.5)
plt.legend()
plt.xlabel('Day in January')
plt.ylabel('Avalanche Risk Index Value')
plt.title('Avalanche Risk By Terrain in January 2017')

plt.tight_layout()
if saveIt == 1:
    plt.savefig('fig_Jan2017_avyrisk.png')

#COMMENTED OUT PCA because I don't think we need it
##first do PCA for TEMP data, then use PCs as predictors
# n_modesT = np.min(np.shape(temp_norm))
# pcaT = PCA(n_components = n_modesT)
# PCsT = pcaT.fit_transform(temp_norm)
# eigvecsT = pcaT.components_
# fracVarT = pcaT.explained_variance_ratio_

##first do PCA for SNOW data, then use PCs as predictors
# n_modesS = np.min(np.shape(snowdep_norm))
# pcaS = PCA(n_components = n_modesS)
# PCsS = pcaS.fit_transform(snowdep_norm)
# eigvecsS = pcaS.components_
# fracVarS = pcaS.explained_variance_ratio_

##plot fraction of variance explained by each mode
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.scatter(range(len(fracVarT)),fracVarT)
# plt.xlabel('Mode Number')
# plt.ylabel('Fraction Variance Explained')
# plt.title('Variance Explained by All Modes for Temp')
# plt.subplot(1,2,2)
# plt.scatter(range(len(fracVarS)),fracVarS)
# plt.xlabel('Mode Number')
# plt.ylabel('Fraction Variance Explained')
# plt.title('Variance Explained by All Modes for Snow Depth')
# plt.tight_layout()
# plt.show()

# ready for MLP, see MLP file
# define variables for MLP
N = len(days) # size of data
n_predictors = 2
predictors = np.vstack([temp_norm,snowdep_norm]).T

target_bt = below_tree
target_t = tree
target_at = above_tree
target = [target_bt, target_t, target_at]

# loop through the 3 terrain categories for MLP on each
fracTrain = 0.8 # 80% of data used for training
NTrain = int(np.floor(fracTrain*N))
for m in range(0,2): 
    target_m = target[m]
    x_train = predictors[:NTrain]
    y_train = target_m[:NTrain]

    x_test = predictors[NTrain:,:]
    y_test = target_m[NTrain:]

    y_out_ensemble_mean, y_out_ensemble, RMSE_ensemble_mean, RMSE_ensemble, nhn_best, nhl_best = MLP(x_train,y_train,x_test,y_test)

    #visualize
    plt.figure(figsize=(12,8))

    plt.subplot(241)
    plt.scatter(len(RMSE_ensemble),RMSE_ensemble_mean,c='k',marker='*')
    plt.scatter(range(len(RMSE_ensemble)),RMSE_ensemble)
    plt.xlabel('Model #')
    plt.ylabel('RMSE')
    plt.title('Error')

    plt.subplot(242)
    plt.scatter(range(len(nhn_best)),nhn_best)
    plt.xlabel('Model #')
    plt.ylabel('# Hidden Neurons')
    plt.title('Hidden Neurons')

    plt.subplot(243)
    plt.scatter(range(len(nhl_best)),nhl_best)
    plt.xlabel('Model #')
    plt.ylabel('# Hidden Layers')
    plt.title('Hidden Layers')

    plt.subplot(244)
    plt.scatter(y_test,y_out_ensemble_mean)
    #plt.plot((np.min(y_test),np.max(y_test)),'k--')
    plt.xlabel('y_test')
    plt.ylabel('y_model')
    plt.title('Ensemble')

    plt.subplot(212)
    plt.plot(y_out_ensemble_mean)
    plt.plot(np.array(y_test),alpha = 0.5)

    plt.tight_layout()
    if saveIt:
        plt.savefig('fig_Jan2017_model_overview.png')

    #visualize individual model runs
    plt.figure(figsize = (12,5))
    plt.scatter(range(len(y_test)),y_test,label='Observations',zorder = 0,alpha = 0.3)
    plt.plot(range(len(y_test)),np.transpose(y_out_ensemble[0]),color = 'k',alpha = 0.4,label='Individual Models',zorder=1) #plot first ensemble member with a label
    plt.plot(range(len(y_test)),np.transpose(y_out_ensemble[1:]),color = 'k',alpha = 0.4,zorder=1) #plot remaining ensemble members without labels for a nicer legend
    plt.plot(range(len(y_test)),y_out_ensemble_mean,color = 'k',label = 'Ensemble',zorder=2, linewidth = 3)
    plt.xlabel('Time', fontsize = 20)
    plt.ylabel('y', fontsize = 20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title('MLP Model Results', fontsize = 24)
    plt.legend(fontsize = 16, loc = 'best')
    plt.tight_layout()
    if saveIt:
        plt.savefig('fig_Jan2017_model_results.png')

    #visualize performance metrics/etc
    plt.figure(figsize=(16,4))

    plt.subplot(131)
    plt.scatter(len(RMSE_ensemble),RMSE_ensemble_mean,c='k',marker='*', s = 150)
    plt.scatter(range(len(RMSE_ensemble)),RMSE_ensemble, s = 150)
    plt.xlabel('Model #', fontsize = 20)
    plt.ylabel('RMSE', fontsize = 20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    #plt.ylim((np.min(RMSE_ensemble) - 0.005, np.max(RMSE_ensemble)+0.005))
    plt.title('Error', fontsize = 20)

    plt.subplot(132)
    plt.scatter(range(len(nhn_best)),nhn_best, s = 150)
    plt.xlabel('Model #', fontsize = 20)
    plt.ylabel('# Hidden Neurons', fontsize = 20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title('Hidden Neurons', fontsize = 20)

    plt.subplot(133)
    plt.scatter(range(len(nhl_best)),nhl_best, s = 150)
    plt.xlabel('Model #', fontsize = 20)
    plt.ylabel('# Hidden Layers', fontsize = 20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title('Hidden Layers', fontsize = 20)

    plt.tight_layout()
    if saveIt:
        plt.savefig(f'fig_Jan2017_model_hidden.png')
