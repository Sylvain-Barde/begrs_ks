# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:26:41 2017

This script produces the simulated K+S training data required to run the 
BEGRS analysis.

@author: Sylvain Barde, University of Kent
"""

import multiprocessing as mp
import time
import os
import pickle
import zlib

from ks_functions import import_design, get_sobol_samples, runKS, zipDirectory
#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Paths to the K+S model files
    KSfolder = 'K+S'
    setupPath = 'config'
    setupFiles = ['config.txt',
                  'flags.txt']
    
    # List of specifications simulated
    modelSpecs = [
                 {'tag':'KS_sobol4000_base',          # modelChoice == 0
                  'design':'KS_params_exp_base.csv'}, #   baseline expectations
                 {'tag':'KS_sobol4000_exp1',          # modelChoice == 1
                  'design':'KS_params_exp1.csv'},     #   AR4 exp (+2)
                 {'tag':'KS_sobol4000_exp2',          # modelChoice == 2
                  'design':'KS_params_exp2.csv'},     #   Accel. exp (+1)
                 {'tag':'KS_sobol4000_exp3',          # modelChoice == 3
                  'design':'KS_params_exp3.csv'},     #   Adapt. exp (+1)
                 {'tag':'KS_sobol4000_exp4',          # modelChoice == 4
                  'design':'KS_params_exp4.csv'},     #   Extrapol. exp (+2)
                 {'tag':'KS_sobol4000_exp5',          # modelChoice == 5
                  'design':'KS_params_exp5.csv'},     #   Trend exp (+1)
                 {'tag':'KS_sobol4000_exp7',          # modelChoice == 6
                  'design':'KS_params_exp7.csv'}      #   LAA exp (+1)
                 ]
    
    # Training and testing parameters configs
    paramConfig = {
                'train':{'ext':'',
                         'samples':4000,
                         'skip':500},
                'test':{'ext':'_sbc',
                        'samples':1000,
                        'skip':5000}
                   }
     
    # Set parameters for the run
    runMode = 'test'        # set to 'train' or 'test'
    modChoice = 6
    numCores = 36
    numEval = 0
    numObs = 300
    
    # Create model path:
    modelTag = modelSpecs[modChoice]['tag'] + paramConfig[runMode]['ext']
    modPath = KSfolder + '//' + modelTag
    if not os.path.exists(modPath):
            os.makedirs(modPath,mode=0o777)
    print('Created run path: ' + modPath)
    
    # Create logging directory
    logPath = modPath + "//logs"
    print('Saving logs to: ' + logPath)
    if not os.path.exists(logPath):
        os.makedirs(logPath,mode=0o777)
    
    # Create working directory
    simPath = modPath + "//simData"
    print('Saving simulated data to: ' + simPath)
    if not os.path.exists(simPath):
            os.makedirs(simPath,mode=0o777)
    
    # Import design
    designFile = modelSpecs[modChoice]['design']
    design = import_design(designFile)
    parameter_range = design['estimated']['range']
    
    # Create parametrisation
    print("Generating parameter samples for " + modelTag)
    numSamples = paramConfig[runMode]['samples']
    skip = paramConfig[runMode]['skip']
    param_dims = parameter_range.shape[0]
    samples = get_sobol_samples(numSamples, parameter_range, skip)
    params = {'design' : design,
              'samples' : samples,
              'parameter_range' : parameter_range}
        
    fil = open(simPath + '/parampool.pkl','wb')
    fil.write(pickle.dumps(params, protocol=2))
    fil.close()
    
    # Populate settings
    settings = {'numObs': numObs,
                'KSfolder' : KSfolder,
                'logPath' : logPath,
                'setupPath' : modPath + '//' + setupPath,
                'setupFiles' : setupFiles,
                'numEval' : 0}
    
    # ------------------------------------------------------------------------
    t_start = time.time()
    
    # Populate job
    job_inputs = []
    for i in range(numSamples):
        job_inputs.append((settings, params, i))

    # -- Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|    Parallel run of C++ K+S model - SOBOL sampling    |')
    print('+------------------------------------------------------+')
    print(' Model: ' + modelTag )
    print(' Number of cores: ' + str(numCores) + \
        ' - Number of tasks: ' + str(numSamples))
    print('+------------------------------------------------------+')
        
    # Create pool and run parallel job
    pool = mp.Pool(processes=numCores)
    res = pool.map(runKS, job_inputs)

    # Close pool when job is done
    pool.close()
        
    # Extract results for timer and process files
    sum_time = 0
    simDataFull = {}
    for i in range(numSamples):

        time_i, simData = res[i]
        sum_time = sum_time + time_i
        simDataFull[i] = simData
        
    fil = open(simPath + '/' + modelTag + '_data.pkl','wb') 
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