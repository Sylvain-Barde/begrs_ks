# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:24 2021

This script estimates parameter values for the K+S model on US macroeconomic 
data, using a trained BEGRS surrograte model.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pandas as pd
import pickle

from begrs import begrs, begrsNutsSampler

KSfolder = 'K+S'
simDataPath = 'simData'
empDataPathBase = 'us_data/empirical_dataset_'
modelPath = 'models'

dataFreqs = {'annual':{'tag':'a',
                        'path':'annual'},
             'quarterly':{'tag':'q',
                       'path':'quarterly'}
            }

modelTags = ['KS_sobol4000_base',   # modelChoice == 0
             'KS_sobol4000_exp1',   # modelChoice == 1
             'KS_sobol4000_exp2',   # modelChoice == 2
             'KS_sobol4000_exp3',   # modelChoice == 3
             'KS_sobol4000_exp4',   # modelChoice == 4
             'KS_sobol4000_exp5',   # modelChoice == 5
             'KS_sobol4000_exp7']   # modelChoice == 6

# Select training data parameters (to identify trained model)
frequency = 'quarterly'  # Set to 'quarterly' or 'annual'
modelChoice = 6          # Choose model from list above
dropExit = True          # Drop the firm exit variable (not used in new data)
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
modelTag = modelTags[modelChoice] +'_{:s}'.format(dataFreqs[frequency]['tag'])
begrsEst = begrs()
begrsEst.load( modelPath + '/{:s}/'.format(dataFreqs[frequency]['path']) + 
               modelTag + namePad + saveName)

#-----------------------------------------------------------------------------
# Define prior and posteriors
def logP(sample):
    
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

#-----------------------------------------------------------------------------
# Run Estimation on US data, for the selected empirical settings
estimateName = empDataPathBase + dataFreqs[frequency]['path'] + '.csv'
xEmp = pd.read_csv(estimateName, index_col=0).to_numpy()

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

fil = open(modelPath + '/{:s}/'.format(dataFreqs[frequency]['path']) + 
           modelTag + namePad  + saveName + '/' + 
           'estimates.pkl','wb')
fil.write(pickle.dumps(results, protocol=2))
fil.close()
#-----------------------------------------------------------------------------