# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:24 2021

This scripts trains a BEGRS surrogate on the simulated data produced by the 
Dosi et. al (2015) financial K+S model.

@author: Sylvain Barde, University of Kent

"""

import numpy as np
import pickle
import zlib
import os

from begrs import begrs

KSfolder = 'K+S'
dataPath = 'simData'
modelPath = 'models'
modelTags = ['KS_sobol4000_base',   # modelChoice == 0
             'KS_sobol4000_exp1',   # modelChoice == 1
             'KS_sobol4000_exp2',   # modelChoice == 2
             'KS_sobol4000_exp3',   # modelChoice == 3
             'KS_sobol4000_exp4',   # modelChoice == 4
             'KS_sobol4000_exp5',   # modelChoice == 5
             'KS_sobol4000_exp7']   # modelChoice == 6

# Setup training data parameters
modelChoice = 6     # Choose model from list above
numSamples = 4000   # Choose No samples (has to be less than available in data)
numObs = 300        # Choose No obs (has to be less than available in data)
dropExit = True     # Drop the firm exit variable (not used)

# Set learning hyper-parameters here
numLatents = 7         # Identify by pca? - 7 seems good here
numInducingPts = 250     # Number of inducing points
batchSize = 20000        # Size of training minibatches (mainly a speed issue)
numIter = 100            # Number of epoch iterations 
learning_rate = 0.0005   # Learning rate

# Load samples & parameter ranges
modelTag = modelTags[modelChoice]
fil = open(KSfolder + '/' + 
           modelTag + '/' + 
           dataPath + '/parampool.pkl','rb')
datas = fil.read()
fil.close()
params = pickle.loads(datas,encoding="bytes")
samples = params['samples'][0:numSamples,:]
parameter_range = params['parameter_range']

# Load & repackage simulation data
fil = open(KSfolder + '/' + 
           modelTag + '/' + 
           dataPath + '/' + 
           modelTag + '_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
simData = pickle.loads(datas,encoding="bytes")

numSamples = max(numSamples,len(simData))
numTasks = simData[0].shape[1]

modelData = np.zeros([numObs,numTasks,numSamples])
for i in range(numSamples):
    
    if len(simData[i]) == 0:
        modelData[:,:,i] = np.nan
    else:    
        checkObs = simData[i].shape[0]
        if checkObs < numObs:
            paddedSimData = np.empty([numObs,numTasks])
            paddedSimData[:] = np.nan
            paddedSimData[0:checkObs,:] = simData[i]
            modelData[:,:,i] = paddedSimData
        else:
            modelData[:,:,i] = simData[i][0:numObs,:]
        
if dropExit is True:
    modelData = np.delete(modelData,9,axis = 1)
    namePad = '_cut'
    
else:
    namePad = ''

#%%
# Create a begrs estimation object, set training data and train GP
begrsEst = begrs()
begrsEst.setTrainingData(modelData, samples, parameter_range, wins = 0.05)
begrsEst.train(numLatents, numInducingPts, batchSize, numIter, 
                learning_rate)

# Save trained model
savePath = modelPath + '/' + modelTag + namePad
if not os.path.exists(savePath):
    os.makedirs(savePath,mode=0o777)

saveName = '/begrs_batch_{:d}_ind_{:d}_lr_{:s}_ep_{:d}'.format(
                batchSize,
                numInducingPts,
                str(learning_rate)[2:],
                numIter) 

begrsEst.save(savePath + saveName)
#-----------------------------------------------------------------------------