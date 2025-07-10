# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:25:52 2018

This script provides quantisation diagnostics for the MIC comparison of K+S
specifications

@author: Sylvain Barde, University of Kent
"""
import numpy as np
import zlib
import pickle
import mic.toolbox as mt

#       0   1   2   3   4  5   6  7   8   9   10
lb = [-10,-10,-60,-20,-12,-10,  0, 0, -10,  0]
ub = [ 10, 10, 60, 20, 12, 10, 50, 7,  10, 12]
r_vec = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

dropExit = True

# Load empirical data
path = 'empirical_dataset.txt'
data = np.loadtxt(path, delimiter="\t")
if dropExit is True:
    data = np.delete(data,9,axis = 1)

empMax = np.percentile(data, 97.5, axis = 0)
empMin = np.percentile(data, 2.5, axis = 0)

data_struct = mt.bin_quant(data,lb,ub,r_vec,'')

# Load simulation data
KSfolder = 'K+S'
dataPath = 'simData'

models = ['KS_sobol4000_full_mc',
          'KS_baseline_mc',
          'KS_baseline_us_mc']

overallMaxTab = np.zeros([len(models),len(r_vec)])
overallMinTab = np.zeros([len(models),len(r_vec)])

for j, modelTag in enumerate(models):

    fil = open(KSfolder + '/' + modelTag + '/' + dataPath + '/' + modelTag + '_data.pkl','rb')
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
