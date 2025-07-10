# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:03:46 2016

This script runs the MIC comparison across the different K+S specifications 
estimates via BEGRS, particularly the expectation mechanisms.

@author: Sylvain Barde, University of Kent
"""

import sys
import os
import time
import pickle
import zipfile
import zlib

import numpy as np
import multiprocessing as mp
import mic.toolbox as mt

#------------------------------------------------------------------------------
def wrapper(inputs):
    """ wrapper function"""

    tic = time.time()
    dropExit = True
    
    # Unpack inputs and parameters
    params = inputs[0]
    var_vec_base = inputs[1]
    task    = inputs[2]
    
    sim_dir    = params['sim_dir']
    log_path   = params['log_path']
    mod_name   = params['mod_name']
    empPath    = params['empPath']
    empSetting = params['empSetting']
    policyFlag = params['policyFlag']
    lb         = params['lb']
    ub         = params['ub']
    r_vec      = params['r_vec']
    hp_bit_vec = params['hp_bit_vec']
    mem        = params['mem']
    lags       = params['lags']
    d          = params['d']
    num_runs   = params['num_runs']
    
    # -- Declare task initialisation
    print (' Task number {:3d} initialised'.format(task))
    
    # Load simulated data
    sim_path = 'K+S//' + mod_name + '//' + policyFlag + '//' + sim_dir + '//' + mod_name + '_data.pkl'
    print (' Training load path:      ' + sim_path)
    fil = open(sim_path,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData = pickle.loads(datas,encoding="bytes")
    
    # Load empirical data, truncate as needed for empirical setting
    emp_data = np.loadtxt(empPath, delimiter="\t") 
    if empSetting == '_crisis':
        emp_data = emp_data[29:,:]
    elif empSetting == '_gm':
        emp_data = emp_data[0:93,:]
    
    emp_data_struct = mt.bin_quant(emp_data,lb,ub,r_vec,'notests')
    emp_data_bin = emp_data_struct['binary_data']

    scores = []     # Initialise score output

    # -- iterate over conditioned variables
    numVars = len(r_vec)
    for i in range(numVars):  
        var_vec = var_vec_base[i:numVars]
        var = var_vec[0]
            
        var_str = ''
        for var_i in var_vec:
            var_str = var_str + str(var_i)   
        
        # Redirect output to file and print task/process information
        file_name = 'task_' + str(task) + '_var_' +  var_str + '.out'
        sys.stdout = open(log_path + '//' + file_name, "w")
    
        print (' Task number :   {:3d}'.format(task))
        print (' Variables   :   ' + var_str)
        print (' Parent process: {:10d}'.format(os.getppid()))
        print (' Process id:     {:10d}'.format(os.getpid()))

        # -- Generate permutation from data
        perm = mt.corr_perm(emp_data, r_vec, hp_bit_vec, var_vec, lags, d)

        # - Stage 1 - train tree with training data
        tag = 'Model setting ' + mod_name
        for j in range(num_runs):
            
            if len(simData[j]) == 0:
                print('\n Training series {:} - no data'.format(j+1))
            else:
                dat = simData[j]
                if dropExit is True:
                    dat = np.delete(dat,9,axis = 1)
                    
                # Protect here (Ideally, fix MIC toolbox!)
                N = dat.shape[0]               # Number of observations
                N_crit = int(np.floor((N-lags)/10))  # Critical threshold (display)
                if np.remainder((N-lags),10) >= N_crit:
                    dat = np.delete(dat,-1,axis = 0)

                data_struct = mt.bin_quant(dat,lb,ub,r_vec,'notests') # Discretise
                data_bin = data_struct['binary_data']
                print('\n Training series {:}'.format(j+1))
                # -------------
                
                if j == 0:
                    output = mt.train(None, 
                                      data_bin, mem, lags, d, var, tag, perm)
                else:
                    T = output['T']
                    try:
                        output = mt.train(T, 
                                      data_bin, mem, lags, d, var,tag, perm)
                    except:
                        print(j)
                        data_string = data_bin.string
                        N = data_string.shape[0]               # Number of observations
                        N_crit = int(np.floor((N-T.lags)/10))  # Critical threshold (display)
                        
                        print(N)
                        print(N_crit)
                        # data_struct = mt.bin_quant(dat,lb,ub,r_vec) # Discretise
                        # print(data_struct['diagnostics'])

        T.desc()
        
        # - Stage 2 - Score empirical series with tree
        score_struct = mt.score(T, emp_data_bin)
        mic_vec = score_struct['score'] - score_struct['bound_corr']
        scores.append(mic_vec)
        
    # Redirect output to console and print completetion time
    sys.stdout = sys.__stdout__
    
    # Generate zip file and delete temporary logs to save space
    z = zipfile.ZipFile(log_path + "//log_task_" + str(task) + ".zip", "w",
                        zipfile.ZIP_DEFLATED)
    for i in range(numVars):
        var_vec = var_vec_base[i:numVars]           
        var_str = ''
        for var_i in var_vec:
            var_str = var_str + str(var_i)   
                
        file_name = 'task_' + str(task) + '_var_' +  var_str + '.out'
        file_name_full = log_path + '//' + file_name
        
        z.write(file_name_full,file_name)
        os.remove(file_name_full)

    z.close()
    
    # Print completetion time
    toc = time.time() - tic
    print(' Task number {:3d} complete - {:10.4f} secs.'.format(int(task),toc))

    # Return output (must be pickleable)
    return (toc,scores)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Load/Save directories
    sim_dir = 'simData'
    save_dir = 'scores'
    empDataPath = 'us_data/empirical_dataset_new.txt'
    originalCalib = False      # Flag for original calibration run

    # Pick empirical setting (fixes dataset + begrs estimates)
    empChoice = 0
    empSettings = ['_crisis',  # 0 - Crisis period
                   '_gm']      # 1 - Great moderation

    # Pick training method from list (for robustness checks)
    methodChoice = 2
    methods = ['high_L1',               # 0 - robustness
               'low_L2',                # 1 - robustness
               'high_L2']               # 2 - used in paper   NO!!
    
    # Simulated dataset from estimates
    if originalCalib:
        models = ['KS_calib_orig_mc',         # Original baseline model
                  'KS_calib_us_mc']     # baseline model - US calibrated
        # models = ['KS_baseline_us_mc']      # baseline model - US calibrated
    else:
        models = ['KS_sobol4000_base_mc',   # baseline expectations
                  'KS_sobol4000_exp1_mc',   # AR4 exp
                  'KS_sobol4000_exp2_mc',   # Accel. exp
                  'KS_sobol4000_exp3_mc',   # Adapt. exp
                  'KS_sobol4000_exp4_mc',   # Extrapol. exp
                  'KS_sobol4000_exp5_mc',   # Trend exp
                  'KS_sobol4000_exp7_mc']   # LAA exp
    policyFlag = 'F_norule_M_tr2'       # Policy flag (no fiscal, dual Taylor)
    
    var_vec_base = [[10,9,8,7,6,5,4,3,2,1],    # 0
                    [1,10,9,8,7,6,5,4,3,2],    # 1
                    [2,1,10,9,8,7,6,5,4,3],    # 2
                    [3,2,1,10,9,8,7,6,5,4],    # 3
                    [4,3,2,1,10,9,8,7,6,5],    # 4
                    [5,4,3,2,1,10,9,8,7,6],    # 5
                    [6,5,4,3,2,1,10,9,8,7],    # 6
                    [7,6,5,4,3,2,1,10,9,8],    # 7
                    [8,7,6,5,4,3,2,1,10,9],    # 8
                    [9,8,7,6,5,4,3,2,1,10],    # 9
                    
                    [1,2,3,4,5,6,7,8,9,10],    # 10
                    [10,1,2,3,4,5,6,7,8,9],    # 11
                    [9,10,1,2,3,4,5,6,7,8],    # 12
                    [8,9,10,1,2,3,4,5,6,7],    # 13
                    [7,8,9,10,1,2,3,4,5,6],    # 14
                    [6,7,8,9,10,1,2,3,4,5],    # 15
                    [5,6,7,8,9,10,1,2,3,4],    # 16
                    [4,5,6,7,8,9,10,1,2,3],    # 17
                    [3,4,5,6,7,8,9,10,1,2],    # 18
                    [2,3,4,5,6,7,8,9,10,1]]    # 19
    
    # Parametrise MIC methods
    if methodChoice == 0:     # High resolution, 1 memory lag
        res = 6
        lags = 1
    elif methodChoice == 1:   # Low resolution, 2 memory lags
        res = 3
        lags = 2
    elif methodChoice == 2:   # High resolution, 2 memory lags
        res = 6
        lags = 2
    
    numVars = len(var_vec_base[0])
    r_vec = numVars*[res]
    empSetting = empSettings[empChoice]
    method = methods[methodChoice]
    if originalCalib:
        empTag = ''
    else:
        empTag = empSetting
    
    for model in models:

        t_start = time.time()     
        
        # Create logging directory
        log_path = "logs//estimates"  + empSetting + '//' + method \
            + '//' + model + "//train_run_" + time.strftime("%d-%b-%Y_%H-%M-%S",
                    time.gmtime())
        print('Saving logs to: ' + log_path)
        os.makedirs(log_path,mode=0o777)
    
        # Create saving directory    
        save_path = save_dir + '//estimates'  + empSetting + '//' + \
        method + '//' + model
        if not os.path.exists(save_path):
            os.makedirs(save_path,mode=0o777)
    
        # Set parameters    
        params = dict(sim_dir = sim_dir,
                      log_path = log_path,
                      mod_name = model + empTag,
                      empPath = empDataPath,
                      empSetting = empSetting,
                      policyFlag = policyFlag,
                      lb = [-10,-10,-60,-20,-12,-10,  0, 0, -10,  0],
                      ub = [ 10, 10, 60, 20, 12, 10, 50, 7,  10, 12],
                      r_vec = r_vec,
                      hp_bit_vec = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                      mem  = 1000000,
                      d    = 32,
                      lags = lags,
                      num_runs = 1000)
        
        # Populate job
        num_tasks = len(var_vec_base)
        num_cores = num_tasks
        job_inputs = []
        for i in range(num_tasks):
            job_inputs.append((params, var_vec_base[i], i))
        
        # -- Initialise Display
        print(' ')
        print('+------------------------------------------------------+')
        print('|           Parallel MIC - Training and scoring        |')
        print('+------------------------------------------------------+')    
        print(' Number of cores: ' + str(num_cores) + \
            ' - Number of tasks: ' + str(num_tasks))
        print('+------------------------------------------------------+')
        
        # Create pool and run parallel job
        pool = mp.Pool(processes=num_cores)
        results = pool.map(wrapper,job_inputs)
    
        # Close pool when job is done
        pool.close()
     
        # Extract results and get timer
        sum_time = 0
        for(i, var_vec) in enumerate(var_vec_base):
            res_i = results[i]
            sum_time = sum_time + res_i[0]
    
            # -- Save results (name depends on variables)
            var_str = ''
            for var_i in var_vec:
                var_str = var_str + str(var_i)   
                
            fil = open(save_path + '//scores_var_' + var_str + '.pkl','wb')
            fil.write(pickle.dumps(res_i, protocol=2))
            fil.close()
            
        # Print timer diagnostics
        print('+------------------------------------------------------+')
        timer_1 = time.time() - t_start
        print(' Total running time:     {:10.4f} secs.'.format(timer_1))
        print(' Sum of iteration times: {:10.4f} secs.'.format(sum_time))
        print(' Mean iteration time:    {:10.4f} secs.'.format(
                sum_time/num_tasks))
        print(' Speed up:               {:10.4f}'.format(sum_time/timer_1))
        print('+------------------------------------------------------+')

#------------------------------------------------------------------------------
