# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:24 2021

This script estimates parameter values for the K+S model on US macroeconomic 
data, using a trained BEGRS surrograte model.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle

from begrs import begrs, begrsNutsSampler

# Create a begrs estimation object, load existing model
KSfolder = 'K+S'
simDataPath = 'simData'
empDataPath = 'us_data/empirical_dataset_new.txt'
modelPath = 'models'
modelTags = ['KS_sobol4000_base',   # modelChoice == 0
             'KS_sobol4000_exp1',   # modelChoice == 1
             'KS_sobol4000_exp2',   # modelChoice == 2
             'KS_sobol4000_exp3',   # modelChoice == 3
             'KS_sobol4000_exp4',   # modelChoice == 4
             'KS_sobol4000_exp5',   # modelChoice == 5
             'KS_sobol4000_exp7']   # modelChoice == 6

empSettings = ['estimates_full',    # Full dataset
               'estimates_crisis',  # Crisis period
               'estimates_gm']      # Great moderation

# Select training data parameters (to identify trained model)
modelChoice = 6     # Choose model from list above
dropExit = True     # Drop the firm exit variable (not used in new data)
numInducingPts = 250     # Number of inducing points
batchSize = 20000        # Size of training minibatches 
numIter = 100            # Number of epoch iterations 
learning_rate = 0.0005   # Learning rate

# Generate path to Begrs model
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
# Define prior and posteriors
def logP(sample):
    
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

#-----------------------------------------------------------------------------
# Run Estimation on US data, for the 3 empirical settings

# for iterNo, empSetting in enumerate(empSettings):
for iterNo, estimateName in enumerate(empSettings):
    
    # Load US data and truncate sample as needed
    xEmp = np.loadtxt(empDataPath, delimiter="\t")
    if iterNo == 1:
        xEmp = xEmp[29:,:]
    elif iterNo == 2:
        xEmp = xEmp[0:93,:]

    # Print setting (for log)
    print('\n Empirical setting: {:s}'.format(estimateName))

    # Create & configure NUTS sampler for BEGRS
    posteriorSampler = begrsNutsSampler(begrsEst, logP)
    init = np.zeros(begrsEst.num_param)
    posteriorSampler.setup(xEmp, init)
    
    # Run sampler
    N = 10000
    burn = 100
    posteriorSamples = posteriorSampler.run(N, burn)
    sampleESS = posteriorSampler.minESS(posteriorSamples)
    print('Minimal sample ESS: {:.2f}'.format(sampleESS))
    
    # Extract results and save
    results = {'mode' : begrsEst.uncenter(posteriorSampler.mode),
               'samples': posteriorSamples,
               'ess': sampleESS}
    
    fil = open(modelPath + '/' + modelTag + namePad  + saveName + 
                '/' + estimateName + '.pkl','wb')
    fil.write(pickle.dumps(results, protocol=2))
    fil.close()
#-----------------------------------------------------------------------------