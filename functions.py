import numpy as np
from sklearn.neural_network import MLPRegressor

# SCRIPT FOR RUNNING MLP
# define RMSE as a function, since we'll use this in the NN model 
def rmse(target,prediction):
    return(np.sqrt(((target - prediction)**2).sum()/len(target)))

def MLP(x_train,y_train,x_test,y_test):
    num_models = 10 #number of models to build for the ensemble
    min_nhn = 1 #minimum number of hidden neurons to loop through (nhn = 'number hidden neurons')
    max_nhn = 9 #maximum number of hidden neurons to loop through
    max_hidden_layers = 1 #maximum number of hidden layers to loop through (nhl = 'number hidden layers')
    batch_size = 32
    solver = 'adam' #use stochastic gradient descent as an optimization method (weight updating algorithm)
    activation = 'relu'
    learning_rate_init = 0.01

    max_iter = 1500 #max number of epochs to run
    early_stopping = True #True = stop early if validation error begins to rise
    validation_fraction = 0.1 #fraction of training data to use as validation

    y_out_all_nhn = []
    y_out_ensemble = []
    RMSE_ensemble = [] #RMSE for each model in the ensemble
    RMSE_ensemble_cumsum = [] #RMSE of the cumulative saltation for each model
    nhn_best = []
    nhl_best = []

    for model_num in range(num_models): #for each model in the ensemble
        
        print('Model Number: ' + str(model_num))
        
        RMSE = []
        y_out_all_nhn = []
        nhn = []
        nhl = []
        
        for num_hidden_layers in range(1,max_hidden_layers+1):
        
            print('\t # Hidden Layers = ' + str(num_hidden_layers))
        
            for num_hidden_neurons in range(min_nhn,max_nhn+1): #for each number of hidden neurons

                print('\t\t # hidden neurons = ' + str(num_hidden_neurons))
                
                hidden_layer_sizes = (num_hidden_neurons,num_hidden_layers)
                model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                    verbose=False,
                                    max_iter=max_iter, 
                                    early_stopping = early_stopping,
                                    validation_fraction = validation_fraction,
                                    batch_size = batch_size,
                                    solver = solver,
                                    activation = activation,
                                    learning_rate_init = learning_rate_init)

                model.fit(x_train,y_train) #train the model

                y_out_this_nhn = model.predict(x_test) #model prediction for this number of hidden neurons (nhn)
                y_out_all_nhn.append(y_out_this_nhn) #store all models -- will select best one best on RMSE

                RMSE.append(rmse(y_test,y_out_this_nhn)) #RMSE between cumulative curves
                
                nhn.append(num_hidden_neurons)
                nhl.append(num_hidden_layers)
            
        indBest = RMSE.index(np.min(RMSE)) #index of model with lowest RMSE
        RMSE_ensemble.append(np.min(RMSE))
        nhn_best.append(nhn[indBest])
        nhl_best.append(nhl[indBest])
        #nhn_best.append(indBest+1) #the number of hidden neurons that achieved best model performance of this model iteration
        y_out_ensemble.append(y_out_all_nhn[indBest])
        
        print('\t BEST: ' + str(nhl_best[model_num]) + ' hidden layers, '+ str(nhn_best[model_num]) + ' hidden neurons')
        
    y_out_ensemble_mean = np.mean(y_out_ensemble,axis=0)
    RMSE_ensemble_mean = rmse(y_out_ensemble_mean,y_test)

    return y_out_ensemble_mean, y_out_ensemble, RMSE_ensemble_mean, RMSE_ensemble, nhn_best, nhl_best