# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:09:42 2022

This file contains the functions required to simulate the K+S model in 
parallel through python.

@author: Sylvain Barde, University of Kent
"""

import os
import time
import subprocess
import shlex
import shutil
import sobol
import zipfile

import numpy as np
import pandas as pd
from copy import copy

#------------------------------------------------------------------------------
def import_design(inPath):
    
    df = pd.read_csv(inPath,header=None)
    paramNameCol, settingCol = df.columns
    calibratedParams = {'names':[],
                        'values':[]}
    estimatedParams = {'names':[],
                       'range':np.empty([0, 2])}
    flags = {'names':[],
             'values':[]}
    
    for i, param in df.iterrows():
        paramName = param[paramNameCol]
        setting = param[settingCol]       
        
        # Estimated parameter if ':' is in the value range
        if ':' in setting:
            
            if '/' in setting:
                
                paramList = setting.split('/')
                for j, entry in enumerate(paramList):
                    estimatedParams['names'].append(
                        '{:s}~~{:d}'.format(paramName,j))
                    estimatedParams['range'] = np.append(
                        estimatedParams['range'],
                        np.asarray(entry.split(':')).astype(float)[None,:],
                        axis = 0)
                    
            else:
                
                estimatedParams['names'].append(paramName)
                estimatedParams['range'] = np.append(
                        estimatedParams['range'],
                        np.asarray(setting.split(':')).astype(float)[None,:],
                        axis = 0)
            
        else:
            
            # Split multiples
            if '/' in setting:
                
                nameList = paramName.split('/')
                paramList = setting.split('/')
                for (name, value) in zip(nameList,paramList):
                    if 'flag' in name:
                        flags['names'].append(name)
                        flags['values'] = np.append(
                                flags['values'],
                                np.asarray(value).astype(float))   
                    else:
                        calibratedParams['names'].append(name)
                        calibratedParams['values'] = np.append(
                                calibratedParams['values'],
                                np.asarray(value).astype(float))             
            
            else:
                if 'flag' in paramName:
                    flags['names'].append(paramName)
                    flags['values'] = np.append(
                                flags['values'],
                                np.asarray(setting).astype(float))
                else:
                    calibratedParams['names'].append(paramName)
                    calibratedParams['values'] = np.append(
                                calibratedParams['values'],
                                np.asarray(setting).astype(float))
            
    return {'calibrated':calibratedParams,
           'estimated':estimatedParams,
           'flags':flags}
#------------------------------------------------------------------------------
def expand_parameter_values(rawNames,rawValues):
    
    if len(rawNames) != len(rawValues):
        print('Warning: Length mismatch between inputs')
        
    names = []
    values = []
    
    # Look for and process special cases (usually AR processes)
    while len(rawNames) > 0:
        
        name = rawNames.pop(0)
        value = rawValues[0]
        rawValues = np.delete(rawValues,[0])
        
        # Special case 1: multiple allocation but single value
        if '/' in name and '~~' not in name:
            splitNames = name.split('/')
            
            for splitName in splitNames:
                names.append(splitName)
                values.append(value)
            
        # Special case 2: b_a2 & b_b2, where beta2 depends on beta1
        elif 'b_a2/b_b2' in name:
            rawNames.pop(0)
            names.append('b_a2')
            values.append(value)
            
            names.append('b_b2')
            value = rawValues[0]
            rawValues = np.delete(rawValues,[0])
            values.append(values[-1] + value)
        
        # Special case 3: beta1-beta4, follow and AR process with 2 params
        elif 'beta1/beta2/beta3/beta4' in name:
            rawNames.pop(0)
            value2 = rawValues[0]
            rawValues = np.delete(rawValues,[0])            
            
            splitNames = name.replace('~~0','').split('/')
            names.append(splitNames.pop(0))
            values.append(value)
      
            for splitName in splitNames:
                names.append(splitName)
                values.append(values[-1]*value2)

        # Special case 4: delta5-delta6, follow and AR process with 2 params
        elif 'delta5/delta6' in name:
            rawNames.pop(0)
            value2 = rawValues[0]
            rawValues = np.delete(rawValues,[0])
            
            names.append('delta5')
            values.append(value)
            
            names.append('delta6')
            values.append(values[-1]*value2)
        
        # General case, append single value to single name
        else:
            names.append(name)
            values.append(value)
                        
    return names, values


#------------------------------------------------------------------------------                   
def parametrise(params, sample, setupFiles, pathOutRoot):
    
    # pathOut = pathOutRoot + '/sample_{:d}'.format(sample + 1) 
    pathOutParams = pathOutRoot + '/' + setupFiles[0]
    pathOutFlags = pathOutRoot + '/' + setupFiles[1]
    os.makedirs(pathOutRoot,mode=0o777)
    configPaths = ['','']

    # Generate parameters configuration file. Seed is always provided
    paramList = []
    paramList.append("{:s} {:f}\n".format('seed', -(sample+1)))
    
    # 1 - Start with any parameters te be estimated
    if params['design']['estimated']['names']:
        estimatedParams, estimatedValues = expand_parameter_values(
            copy(params['design']['estimated']['names']),
            params['samples'][sample]
            )
        
        for name, value in zip(estimatedParams, estimatedValues):
            paramList.append("{:s} {:f}\n".format(name, value))
    
    # 2 - append any calibrated parameters
    if params['design']['calibrated']['names']:
        for name, value in zip(params['design']['calibrated']['names'], 
                               params['design']['calibrated']['values'].tolist()):
            paramList.append("{:s} {:f}\n".format(name, value))
    
    # Write the parameter confirguration file and report location
    configPaths[0] = pathOutParams
    with open(pathOutParams, 'w') as f_out:
        f_out.write("".join(paramList))
    f_out.close
        
    # Generate flags (if provided)
    if params['design']['flags']['names']:
        flagList = []
        for name, value in zip(params['design']['flags']['names'], 
                               params['design']['flags']['values'].tolist()):
            flagList.append("{:s} {:d}\n".format(name, int(value)))
    
        configPaths[1] = pathOutFlags
        with open(pathOutFlags, 'w') as f_out:
            f_out.write("".join(flagList))
        f_out.close
    
    return configPaths

#------------------------------------------------------------------------------
def get_sobol_samples(num_samples, parameter_support, skips):
    """
    

    Parameters
    ----------
    num_samples : TYPE
        DESCRIPTION.
    parameter_support : TYPE
        DESCRIPTION.
    skips : TYPE
        DESCRIPTION.

    Returns
    -------
    sobol_samples : TYPE
        DESCRIPTION.

    """
    params = np.transpose(parameter_support)
    sobol_samples = params[0,:] + sobol.sample(
                        dimension = parameter_support.shape[0], 
                        n_points = num_samples, 
                        skip = skips
                        )*(params[1,:]-params[0,:])
    
    return sobol_samples

#------------------------------------------------------------------------------
def zipDirectory(path, remove = False):
    
    zipDir = zipfile.ZipFile(path + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(path) + 1
    for base, dirs, files in os.walk(path):
        for file in files:
            fn = os.path.join(base, file)
            zipDir.write(fn, fn[rootlen:])
    
    zipDir.close()

    if remove:
        shutil.rmtree(path)
            
    return
    
#------------------------------------------------------------------------------
def processSimData(resultPath, numObs, 
                   # fileNames = ['/out_0_1_100.txt',
                   #              '/output_0_1_100.txt'],
                   fileNames = ['/output_0_1_100.txt',
                                '/variables_0.txt'],
                   F_init = [50,200]):
        
    variables = [
    'GDP',     # 0 - Real (in initial price terms) GDP
    'C/cpi',   # 1 - Real (in initial price terms) aggregated consumption
    'I/ppi',   # 2 - Aggregated real investment (in initial price terms)
    'CreditSupply_all',  # 3 - Total loans from financial sector
    'NW1',     # 4 - Total net worth of capital-good sector
    'NW2',     # 5 - Total net worth of consumption-good sector
    'w/p',     # 6 - Real wage
    'u',       # 7 - Unemployment rate (including discouraged workers)
    'r',       # 8 - Interest rate set by central bank defined by taylor rule
    'd_cpi',   # 9 - CPI Inflation (change) rate   
    'next1',   # 10 - Num of exiting firms in capital good sector
    'next2',   # 11 - Num of exiting firms in consumption good sector
    'BadDebt_all', # 12 - Total losses from bad debt in financial sector
    'cpi',     # 13 - Consumer price index
    'LS'       # 14 - Labor force size (supply)
    ]
    
    # Wait until file exists (IO issue, can take a while it to be written)
    maxAttempts = 200
    attempt = 0
    while not os.path.exists(resultPath + fileNames[0]):
        time.sleep(0.1)
        attempt+=1
        if attempt > maxAttempts:
            print('maxAttempts reached')
            break
        
    if os.path.exists(resultPath + fileNames[1]):
        varNameFile = resultPath + fileNames[1]
    else:
        print(' Variable file {:s} not produced'.format(
                                                resultPath + fileNames[1]))
        varNameFile = 'K+S' + fileNames[1]    # temporary, will cleanup
    
    # Open results files (names and values separate)
    df = pd.read_csv(resultPath + fileNames[0],delimiter='\s+',header=None)
    # names = np.loadtxt(resultPath + fileNames[1],dtype=str)
    names = np.loadtxt(varNameFile,dtype=str)
    df.rename(columns=dict(zip(np.arange(len(names)),
                               names)), 
              inplace=True)
    
    # Extract raw variables from variable files, remove burn-in period
    # Allow for 1 extra observation for taking lags
    numVars = len(variables)
    N = df.shape[0]
    burn = N - (numObs + 1)
    simDataBase = np.zeros([N,numVars])
    for i, varName in enumerate(variables):
        simDataBase[:,i] = df[varName].to_numpy().flatten()

    # Recover number of firms to calculate exit rate
    simDataBase[0,10:12] = F_init
    FBase = np.cumsum(simDataBase[:,10:12], axis = 0)
    simData = simDataBase[burn:N,:]
    F = FBase[burn:N,:]
    
    # Process variables by type to obtain observables
    logVars = copy(simData[:,0:5])              # Extract growth vars
    logVars[:,4] += simData[:,5]                # Combine net worth vars
    logVars[:,3:5] /= simData[:,-2,None]        # Deflate Loans & net worth
    logVars /= simData[:,-1,None]               # Convert to per-capita
    logVars = np.append(logVars,simData[:,6,None], axis = 1) # Add wages
    logVars[logVars == 0] = 1                   # Protect against zeros
    logDiffVars = np.diff(100*np.log(logVars),axis = 0) # Take log diff

    rateVars = 100*simData[1:,7:10]
    netEntry = 100*np.sum(simData[1:,10:12], axis = 1)/np.sum(F[:-1,:], axis = 1)
    debtLosses = 100*simData[1:,12]/(simData[1:,12]+simData[1:,3])

    # returned variables are:
    # dy	         #  0 - log difference of per-capita Real GDP
    # dc	         #  1 - log difference of per-capita Real Consumption
    # di             #  2 - log difference of per-capita Real Investment
    # dLoans	     #  3 - log difference of per-capita Loans
    # dNetWorth	     #  4 - log difference of per-capita Net worth
    # dw	         #  5 - log difference of real wages
    # u              #  6 - unemployment rate (in %)
    # r              #  7 - Policy rate (in %)
    # pi	         #  8 - CPI inflation rate (in %)
    # exitrate   	 #  9 - Exit rate (in %)
    # lossrate       # 10 - Percentage bad loans(in %)

    # Concatenate observables and return
    simDataProcessed = np.concatenate((logDiffVars,
                                    rateVars,
                                    netEntry[:,None],
                                    debtLosses[:,None]),
                                    axis = 1)
    
    return simDataProcessed

#------------------------------------------------------------------------------
def runKS(inputs):
    """ wrapper function for the K+S model"""

    tic = time.time()
    
    # Unpack inputs, extract run settings and model parameters
    settings = inputs[0]
    params = inputs[1]
    sample = inputs[2]
    
    # numObs = settings['numObs']
    numObs = settings['numObs']
    KSfolder = settings['KSfolder']
    logPath = settings['logPath']
    setupPath = settings['setupPath']
    setupFiles = settings['setupFiles']
    numEval = settings['numEval']
    
    # -- Declare task initialisation
    sampleID = int(numEval + sample + 1)
    print (' Sample {:3d} initialised\n'.format(sampleID))
    
    # -- Generate configuration files for the KS model run     
    pathOutRoot = setupPath + '/sample_{:d}'.format(sampleID) 
    configPaths = parametrise(params, sample, setupFiles, 
                              pathOutRoot = pathOutRoot)
    
    # Print task/process information to log and error files
    sampleStr = str(sampleID)
    logName = 'log_' + sampleStr + '.out'
    logNameFull = logPath + '//' + logName
    with open(logNameFull, 'w') as f_out:
        f_out.write(' Task number :   {:3d}\n'.format(sampleID))
        f_out.write(' Parent process: {:10d}\n'.format(os.getppid()))
        f_out.write(' Process id:     {:10d}\n'.format(os.getpid()))
    f_out.close
    
    logErrName = 'log_' + sampleStr + '_err.out'
    logErrNameFull = logPath + '//' + logErrName
    with open(logErrNameFull, 'w') as f_out:
        f_out.write(' Task number :   {:3d}\n'.format(sampleID))
        f_out.write(' Parent process: {:10d}\n'.format(os.getppid()))
        f_out.write(' Process id:     {:10d}\n'.format(os.getpid()))
    f_out.close
    
    # Launch KS simulation process via a shell call (10 min timeout)
    sh_str = './/{:s}//bin//KSmodel {:s} {:s}'.format(KSfolder,
                                                      configPaths[0],
                                                      configPaths[1])
    args = shlex.split(sh_str)       
    logFile = open(logNameFull,'a')
    errFile = open(logErrNameFull,'a')
    try: 
        subprocess.run(args,stdout=logFile,stderr=errFile,timeout=600)
        # subprocess.run(args,stdout=logFile,stderr=errFile,timeout=500)
    except subprocess.TimeoutExpired:
        print(' Sample number {:3d} error - timeout'.format(sampleID))
        
    # Check for pathologically large log/error files, zip them if they exist
    logStats = os.stat(logNameFull)
    errStats = os.stat(logErrNameFull)
    if (logStats.st_size / (1024 * 1024) +
        errStats.st_size / (1024 * 1024) > 5):
        
        rawFiles = [logName, logErrName]
        rawFilesFull = [logNameFull, logErrNameFull]
        
        zipPath = zipfile.ZipFile(logPath+'//log_'+sampleStr+'_packed.zip',
                                  'w', zipfile.ZIP_DEFLATED)
        for fileName, fileNameFull in zip(rawFiles,rawFilesFull):
            zipPath.write(fileNameFull,fileName)
            os.remove(fileNameFull)
        zipPath.close()
    
    # Process simulation data
    # time.sleep(4)     #short pause - ensure output file is written (I/O issue)
    try:
        simData = processSimData(pathOutRoot, numObs)
    except Exception as e:
        print(' Sample number {:3d} error - unable to process output'.format(sampleID))
        print(e)
        simData = []
    
    # Zip the sample folder to save space
    zipDirectory(pathOutRoot, remove = True)

    # Print completetion time
    toc = time.time() - tic
    print(' Sample number {:3d} complete - {:10.4f} secs.'.format(
            int(sampleID), toc))

    # Return time and simulated output
    return (toc, simData)
#------------------------------------------------------------------------------
