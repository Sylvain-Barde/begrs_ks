# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:26:41 2017

This script runs the parallel policy experiment replication carried out on the 
financial K+S model using the BEGRS-estimated parameters.

@author: Sylvain Barde, University of Kent
"""

import multiprocessing as mp
import numpy as np
import time
import os
import pickle
import zlib

from copy import deepcopy
from ks_functions import import_design, runKS, zipDirectory
#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # General run settings (folder locations/policy options/HPC & MC config)
    numCores = 36
    KSfolder = 'K+S'
    modelPath = 'models'
    setupPath = 'config'
    setupFiles = ['config.txt',
                  'flags.txt']
    empSettings = {0:{'name':'estimates_crisis',
                      'tag':'_crisis'},
                   1:{'name':'estimates_gm',
                      'tag':'_gm'}}
    policies = {'fiscal':{'flags':['flag_balancedbudget']*5,
                          'values':[0,1,2,3,4],
                          'tags':['norule', 'sgp_ec', 'sgp', 'fc_ec', 'fc']},
                'monetary':{'flags':['flagTAYLOR']*2 + ['flag_bonds'],
                              'values':[1,2,3],
                              'tags':['tr1', 'tr2', 'spread']}}
    dropExit = True      # Use to drop firm entry/exit variable
    numSamples = 1000       # Number of MC replications
    numObs = 300        # Number of observations per replication
    
    # Pick specific model and estimate to use
    expMode = 7  # Expectation choice - None or 0-5,7
    usCalib = True  # Only if expMode is None - Values:
                    # - True: US quarterly calibration
                    # - False: original calibration
    
    empMode = 1     # Empirical setting - 0 or 1 (ignored for baseline cases)
    mean = True     # Use to pick which BEGRS vector to simulate
    
    # Construct paths to design files and output folders
    baseline = False     # Baseline parametrisation flag
    if expMode is None: # Original baseline KS model - For replication
        baseline = True
        if usCalib is True:
            # designFile = 'KS_params_baseline_us_20.csv'
            # modelTagSave = 'KS_baseline_us_mc_20'
            designFile = 'KS_params_calib_us.csv'
            modelTagSave = 'KS_calib_us_mc'
        else:
            designFile = 'KS_params_calib_orig.csv'
            modelTagSave = 'KS_calib_orig_mc'
        
    elif expMode == 0:  # Baseline expectations model with estimated paramters
        designFile = 'KS_params_exp_base.csv'
        modelTagLoad = 'KS_sobol4000_base'
        modelTagSave = 'KS_sobol4000_base_mc'
        
    else:               # Alternate expectatons model with estimated parameters
        designFile = 'KS_params_exp{:d}.csv'.format(expMode)
        modelTagLoad = 'KS_sobol4000_exp{:d}'.format(expMode)
        modelTagSave = 'KS_sobol4000_exp{:d}_mc'.format(expMode)
    
    # Extract paths to empirical settings (Not used in baseline case)
    if baseline is False:
        estimateName = empSettings[empMode]['name'] 
        modelTagSave += empSettings[empMode]['tag']

    # Create overall path for base deisgn (will store all 15 policy runs)
    modPathBase = KSfolder + '//' + modelTagSave
    if not os.path.exists(modPathBase):
            os.makedirs(modPathBase,mode=0o777)
    print('Base path for design: ' + modPathBase)
        
    # Generate name to BEGRS models for loading parameter estimates
    numInducingPts = 250     # Number of inducing points
    batchSize = 20000        # Size of training minibatches (mainly a speed issue)
    numIter = 100            # Number of epoch iterations 
    learning_rate = 0.0005   # Learning rate
    saveName = '/begrs_batch_{:d}_ind_{:d}_lr_{:s}_ep_{:d}'.format(
                    batchSize,
                    numInducingPts,
                    str(learning_rate)[2:],
                    numIter) 
    if dropExit is True:
        namePad = '_cut'
    else:
        namePad = ''
    
    # Import design and set estimated parameter values (Not used for baseline)
    samples = {}
    base_design = import_design(designFile)
    if baseline is True:
        sample = np.asarray([]) # Empty parameter settings
    else:
        # Load empirical BEGRS NUTS samples to parameterise simulation
        fil = open(modelPath + '/' + modelTagLoad + namePad  + saveName + 
           '/' + estimateName + '.pkl','rb')        
        datas = fil.read()
        fil.close()
        results = pickle.loads(datas,encoding="bytes")

        if mean is True:
            flatSamples = results['samples']
            sample = np.mean(flatSamples, axis = 0)
        else:
            sample = results['mode']
            
    for i in range(numSamples):
        samples[i] = sample
    
    # Generate universe of all fiscal/monetary policy combinations
    policyFields = ['flags', 'values', 'tags']
    outList = []
    for field in policyFields:
        extractedPolicy = [policies['fiscal'][field],
                           policies['monetary'][field]]
        policyParams = np.array(np.meshgrid(*extractedPolicy)).T.reshape(
                        -1,len(extractedPolicy))
        outList.append(list(zip([field]*policyParams.shape[0], 
                            policyParams)))
    
    # Convert to list of dictionaries for ease of extraction in iterations
    policyList = []
    for item in zip(*outList):
        policyList.append(dict(item))    
    
    #--------------------------------------------------------------------------
    # FROM HERE - Iterate the base design over the 15 policy experiments
    for experiment in policyList:
        t_start = time.time()
        
        # Add policy flags to base design
        design = deepcopy(base_design)
        design['flags']['names'] += list(experiment['flags'])
        design['flags']['values'] = np.concatenate((design['flags']['values'],
                                                    experiment['values']))
        experimentTag = 'F_{:s}_M_{:s}'.format(*experiment['tags'])
        
        # Generate policy-specific folders for run
        # Create temporary model path:
        modPath = modPathBase + '/' + experimentTag
        if not os.path.exists(modPath):
                os.makedirs(modPath,mode=0o777)
        print('Created policy run path: ' + modPath)    
        
        # Create logging directory
        logPath = modPath + "//logs"
        print('Saving policy run logs to: ' + logPath)
        if not os.path.exists(logPath):
            os.makedirs(logPath,mode=0o777)
        
        # Create working directory
        simPath = modPath + "//simData"
        print('Saving simulated data to: ' + simPath)
        if not os.path.exists(simPath):
                os.makedirs(simPath,mode=0o777)
    
        # Populate settings for parallel job
        params = {'design' : design,
                  'samples' : samples}
        
        settings = {'numObs' : numObs,
                    'KSfolder' : KSfolder,
                    'logPath' : logPath,
                    'setupPath' : modPath + '//' + setupPath,
                    'setupFiles' : setupFiles,
                    'numEval' : 0}
        
        job_inputs = []
        for i in range(numSamples):
            job_inputs.append((settings, params, i))
    
        # -- Initialise Display
        print(' ')
        print('+------------------------------------------------------+')
        print('| Parallel run of C++ K+S model - MC policy evaluation |')
        print('+------------------------------------------------------+')
        print(' Model: ' + modelTagSave )
        print(' Number of cores: ' + str(numCores) + \
            ' - Number of tasks: ' + str(numSamples))
        print('+------------------------------------------------------+')
        
        # Create pool and run parallel job
        pool = mp.Pool(processes=numCores)
        res = pool.map(runKS,job_inputs)
    
        # Close pool when job is done
        pool.close()
            
        # Extract results for timer and process files
        sum_time = 0
        simDataFull = {}
        for i in range(numSamples):
    
            time_i, simData = res[i]
            sum_time = sum_time + time_i
            simDataFull[i] = simData
    
        fil = open(simPath + '/' + modelTagSave + '_data.pkl','wb') 
        fil.write(zlib.compress(pickle.dumps(simDataFull, protocol=2)))
        fil.close()
        
        # Clean up logs and configs into zip files (less space)
        zipDirectory(settings['setupPath'], remove = True)
        zipDirectory(settings['logPath'], remove = True)
    
        print('+------------------------------------------------------+')
        timer_1 = time.time() - t_start
        print(' Total running time:     {:10.4f} secs.'.format(timer_1))
        print(' Sum of iteration times: {:10.4f} secs.'.format(sum_time))
        print(' Mean iteration time:    {:10.4f} secs.'.format(
                sum_time/numSamples))
        print(' Speed up:               {:10.4f}'.format(sum_time/timer_1))
        print('+------------------------------------------------------+')
#------------------------------------------------------------------------------