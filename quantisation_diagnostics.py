# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:25:52 2018

This script provides quantisation diagnostics for the MIC comparison of K+S
specifications

@author: Sylvain Barde, University of Kent
"""
import numpy as np
import pandas as pd
import os
import zlib
import pickle
import mic.toolbox as mt

#       0   1   2   3   4   5   6   7   8   9
#      dy  dc  di  dL dnw  dw   u   r dpi  lr
lb = [-10,-10,-60,-20,-12,-10,  0,  0,-10,  0]
ub = [ 10, 10, 60, 20, 12, 10, 50, 10, 10, 12]
r_vec = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

dropExit = True
frequency = 'annual'    # Set to 'quarterly' or 'annual'

# Load empirical data, run quantisation test
empDataPathBase = 'us_data/empirical_dataset_'
dataFreqs = {'annual':{'tag':'a',
                        'path':'annual'},
             'quarterly':{'tag':'q',
                       'path':'quarterly'}
            }

empPath = empDataPathBase + dataFreqs[frequency]['path'] + '.csv'
data = pd.read_csv(empPath, index_col=0).to_numpy()

empMax = np.percentile(data, 97.5, axis = 0)
empMin = np.percentile(data, 2.5, axis = 0)

print('{:s} data'.format(frequency))
data_struct = mt.bin_quant(data,lb,ub,r_vec,'')

# Load simulation data
KSfolder = 'K+S'
dataPath = 'simData'
policyFlag = 'F_norule_M_tr2'
modelTags = ['KS_calib_us',
             'KS_sobol4000_base',
             'KS_sobol4000_exp1',
             'KS_sobol4000_exp2',
             'KS_sobol4000_exp3',
             'KS_sobol4000_exp4',
             'KS_sobol4000_exp5',
             'KS_sobol4000_exp7']

overallMaxTab = np.zeros([len(modelTags),len(r_vec)])
overallMinTab = np.zeros([len(modelTags),len(r_vec)])

for j, modelTagBase in enumerate(modelTags):
    modelTag = modelTagBase +'_{:s}_mc'.format(dataFreqs[frequency]['tag'])
    
    sim_path = os.path.join(KSfolder, modelTag, policyFlag, dataPath, 
                            modelTag + '_data.pkl')
    
    fil = open(sim_path,'rb')

    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData = pickle.loads(datas,encoding="bytes")   
    numSamples = len(simData)
    
    simMax = np.zeros([numSamples,len(r_vec)])
    simMin = np.zeros([numSamples,len(r_vec)])
    
    for i in range(numSamples):
        simSample = simData[i]
        if dropExit is True:
            simSample = np.delete(simSample,9,axis = 1)
        simMax[i,:] = np.percentile(simSample, 97.5, axis = 0)
        simMin[i,:] = np.percentile(simSample, 2.5, axis = 0)
        
    simMaxMean = np.percentile(simMax, 97.5, axis = 0)
    simMinMean = np.percentile(simMin,  2.5, axis = 0)
    
    overallMaxTab[j,:] = simMaxMean[None,:]
    overallMinTab[j,:] = simMinMean[None,:]
    
    print(modelTag)
    data_struct = mt.bin_quant(simData[0],lb,ub,r_vec,'')
    bounds = np.concatenate((simMinMean[:,None],
                             simMaxMean[:,None]),axis=1)
    
    print(bounds)
    
overallMax = np.max(overallMaxTab, axis = 0)
overallMin = np.min(overallMinTab, axis = 0)
