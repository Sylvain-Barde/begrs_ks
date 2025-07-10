# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:24 2021

This script runs a simulation-based calibration diagnostic on a trained BEGRS 
model on the full simulated K+ testing data.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import zlib

from begrs import begrs, begrsNutsSampler, begrsSbc

# Create a begrs estimation object, load existing model
KSfolder = 'K+S'
simDataPath = 'simData'
savePath = 'sbc'
modelPath = 'models'
modelTags = ['KS_sobol4000_base',   # modelChoice == 0
             'KS_sobol4000_exp1',   # modelChoice == 1
             'KS_sobol4000_exp2',   # modelChoice == 2
             'KS_sobol4000_exp3',   # modelChoice == 3
             'KS_sobol4000_exp4',   # modelChoice == 4
             'KS_sobol4000_exp5',   # modelChoice == 5
             'KS_sobol4000_exp7']   # modelChoice == 6

# Select training data parameters (to identify trained model)
modelChoice = 1     # Choose model from list above
dropExit = True     # Drop the firm exit variable (not used in dew data)
numInducingPts = 250     # Number of inducing points
batchSize = 20000        # Size of training minibatches (mainly a speed issue)
numIter = 100            # Number of epoch iterations 
learning_rate = 0.0005   # Learning rate

# Select Testing dataset size for loading BEGRS model
numSbcSamples = 1000
numObs = 300        # Choose No obs (has to be less than available in data)
wins = 0.01         # Winsorise @ 1% (K+S data can take extreme values)
saveName = '/begrs_batch_{:d}_ind_{:d}_lr_{:s}_ep_{:d}'.format(
                batchSize,
                numInducingPts,
                str(learning_rate)[2:],
                numIter) 

if dropExit is True:
    namePad = '_cut'
else:
    namePad = ''

# Load trained Begrs model
modelTag = modelTags[modelChoice] 
begrsEst = begrs()
begrsEst.load( modelPath + '/' + modelTag + namePad + saveName)

#-----------------------------------------------------------------------------
# Load SBC testing data
# Load samples & parameter ranges
fil = open(KSfolder + '/' + modelTag + '_sbc' + '/' + simDataPath
            + '/parampool.pkl','rb')
datas = fil.read()
fil.close()
params = pickle.loads(datas,encoding="bytes")
testSamples = params['samples']

# Load & repackage simulation data
fil = open(KSfolder + '/' + modelTag + '_sbc' + '/' + simDataPath 
           + '/' + modelTag + '_sbc_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
simData = pickle.loads(datas,encoding="bytes")
numTasks = simData[0].shape[1]

testData = np.zeros([numObs,numTasks,numSbcSamples])
for i in range(numSbcSamples):
    
    if len(simData[i]) == 0:
        testData[:,:,i] = np.nan
    else:    # pad if needed
        checkObs = simData[i].shape[0]
        if checkObs < numObs:
            paddedSimData = np.empty([numObs,numTasks])
            paddedSimData[:] = np.nan
            paddedSimData[0:checkObs,:] = simData[i]
            if checkObs == 1:       # protect against single obs case (lag!)
                paddedSimData[1,:] = simData[i]
            testData[:,:,i] = paddedSimData
        else:
            testData[:,:,i] = simData[i][0:numObs,:]

if dropExit is True:
    testData = np.delete(testData,9,axis = 1)
    namePad = '_cut'
    
else:
    namePad = ''

# Winsorise the test data
# - Simulated K+S data is used as synthetic 'empirical' data, needs winsorising
#   to protect against extreme outliers caused by from pathological simulations
#   These values can stall the NUTS sampler.

LB = np.nanquantile(np.vstack(np.moveaxis(testData,-1,-2)),
                    wins/2,axis = 0)[None,:,None]
UB = np.nanquantile(np.vstack(np.moveaxis(testData,-1,-2)),
                    1-wins/2,axis = 0)[None,:,None]

LBCheck = np.less(testData, LB, where =~ np.isnan(testData))
UBCheck = np.greater(testData, UB, where =~ np.isnan(testData))

testData[LBCheck] = np.tile(LB,[numObs,1,numSbcSamples])[LBCheck]
testData[UBCheck] = np.tile(UB,[numObs,1,numSbcSamples])[UBCheck]

#-----------------------------------------------------------------------------
# Define prior and posteriors
def logP(sample):
    
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

#-----------------------------------------------------------------------------
# Run SBC analsysis

# Create SBC object, load data & sampler
SBC = begrsSbc()
SBC.setTestData(testSamples,testData)
SBC.setPosteriorSampler(begrsNutsSampler(begrsEst, logP))

# Run and save results
N = 500             # Number of draws  - aim is to generate sample of 400
burn = 100          # Burn-in, substracted from Number of draws

init = np.zeros(begrsEst.num_param)
SBC.run(N, burn, init, autoThin = False)  # SBC without auto-thin (faster)
SBC.saveData(savePath + 
          '/{:s}_ind_{:d}_lr_{:s}_ep_{:d}.pkl'.format(modelTag,
                                                       numInducingPts,
                                                      str(learning_rate)[2:],
                                                      numIter)) 

#-----------------------------------------------------------------------------
