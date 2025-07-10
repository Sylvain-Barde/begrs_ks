# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:19:55 2024

This script produces the tables for the policy experiment replication carried
out on the financial K+S model using the BEGRS-estimated parameters (tables 6,
7 and 8 in the paper).

@author: Sylvain Barde, University of Kent
"""

import pickle
import zlib
import os

import numpy as np
import scipy.stats as stats
from statsmodels.iolib.table import SimpleTable

#------------------------------------------------------------------------------
def extractResults(model, empSetting):
    titleList = ['Mean GDP growth',
                 'Mean GDP growth volatility',
                 'Mean unemployment rate',
                 'Mean inflation rate']
    
    # Preallocate raw result tables per variable (as nans)
    gdpMeanRaw = np.empty([len(fiscalPolicies),
                     len(monetaryPolicies),
                     numMCIter])*np.nan
    gdpVolRaw = np.empty([len(fiscalPolicies),
                     len(monetaryPolicies),
                     numMCIter])*np.nan
    unempMeanRaw = np.empty([len(fiscalPolicies),
                     len(monetaryPolicies),
                     numMCIter])*np.nan
    inflMeanRaw = np.empty([len(fiscalPolicies),
                     len(monetaryPolicies),
                     numMCIter])*np.nan
    
    meanNans = np.zeros([len(fiscalPolicies),
                     len(monetaryPolicies)])
    
    # Iterate over all policy combinations to extract results
    for i, fiscalPolicy in enumerate(fiscalPolicies):
        for j, monetaryPolicy in enumerate (monetaryPolicies):
            
            loadPath = (KSfolder + '/' + model + empSetting + '/' + 
                        'F_{:s}_M_{:s}'.format(fiscalPolicy,monetaryPolicy) + 
                        '/simData/' + model + empSetting + '_data.pkl')
            
            fil = open(loadPath,'rb')
            datas = zlib.decompress(fil.read(),0)
            fil.close()
            simData = pickle.loads(datas,encoding="bytes")
            
            for n in range(len(simData)):
    
                if len(simData[n]) == 0:
                    meanNans[i,j] += 100/numMCIter
                else:
                    dat = simData[n]
                    if dropExit is True:
                        dat = np.delete(dat,9,axis = 1)
                        
                    # Gather statistics for tables
                    gdpMeanRaw[i,j,n] = np.mean(dat[:,0])
                    gdpVolRaw[i,j,n] = np.std(dat[:,0])
                    unempMeanRaw[i,j,n] = np.mean(dat[:,6])
                    inflMeanRaw[i,j,n] = np.mean(dat[:,8])
                    
    return ([gdpMeanRaw, gdpVolRaw, unempMeanRaw, inflMeanRaw], 
            titleList)

def baseTable(rawData):
    mean = np.nanmean(rawData, axis=-1)
    meanNorm = mean/mean[0,0]
    meanT = np.zeros(mean.shape)
    meanStars = np.empty(mean.shape,dtype='<U6')
    
    # Genarate statistics for table entries (T stats + significance stars)
    for i in range(rawData.shape[0]):
        for j in range(rawData.shape[1]):
            T_test = stats.ttest_ind(a = rawData[0,0,:], 
                                      b = rawData[i,j,:], 
                                      equal_var=False,
                                      nan_policy='omit')
            meanT[i,j] = np.abs(T_test[0])
            starStr = '^{'
            if T_test[1] < 0.1:
                starStr += '*'
            if T_test[1] < 0.05:
                starStr += '*'
            if T_test[1] < 0.01:
                starStr += '*'
            starStr+= '}'
            meanStars[i,j] = starStr
            
    # Format base table panel
    tableFormatted = []
    for meanRow, meanStarsRow, meanTRow in zip(meanNorm, 
                                               meanStars, 
                                               meanT):
        rowFormatted = []
        for cellValue, cellStars in zip (meanRow, meanStarsRow):
            rowMeanFormatted = '${:3.3f}{:s}$'.format(cellValue, cellStars)
            rowFormatted.append(rowMeanFormatted)
        tableFormatted.append(rowFormatted)
        
        rowFormatted = []
        for cellValue in meanTRow:
            rowTFormatted = '$({:3.3f})$'.format(cellValue)
            rowFormatted.append(rowTFormatted)
        tableFormatted.append(rowFormatted)
    
    tableLabels = []
    # for label in fiscalLabels:
    for i in range(rawData.shape[0]):
        tableLabels.append(fiscalLabels[i])
        tableLabels.append('\B')
            
    tableHeaders = []
    for i in range(rawData.shape[1]):
        tableHeaders.append(monetaryLabels[i])
        
    return (tableFormatted, tableLabels, tableHeaders)

#------------------------------------------------------------------------------
# Set Common parameters
KSfolder = 'K+S'
save_path_tables = 'tables'
dropExit = True
numMCIter = 1000
save = False


fiscalPolicies = ['norule', 'sgp', 'fc', 'sgp_ec', 'fc_ec'] # in table order
monetaryPolicies = ['tr1', 'tr2', 'spread']

fiscalLabels = ['\T $No \; rule$', '\T $SGP$', '\T $FC$', '\T $SGP_{ec}$', '\T $FC_{ec}$'] # pretty version
monetaryLabels = ['$TR_{\pi}$','$TR_{\pi,U}$','$spread$']

#------------------------------------------------------------------------------
# Calibrated model results
models = ['KS_calib_orig_mc',       # 0 - Original calibration
          'KS_calib_us_mc']         # 1 - US calibration

# Extract all data
rawDataModels = []
for model in models:
    rawDataList, titleList = extractResults(model,'')
    rawDataModels.append(rawDataList)

# Generate individual table panels and tile them as required
initFlagCol = True
for rawDataList in rawDataModels:
    
    initFlagRow = True
    labelFlagRow = True
    for rawData, title in zip(rawDataList, titleList):
    
        baseTblFormatted, baseTblLabels, baseTblHeaders = baseTable(rawData)        
        baseTblFormatted = np.asarray(baseTblFormatted)

        if initFlagRow:
            hpad = np.asarray(['   ']*baseTblFormatted.shape[1])[None,:]
            colFormatted = np.concatenate((hpad,
                                           baseTblFormatted),
                                           axis = 0)
            if labelFlagRow:
                tableLabels = [title, *baseTblLabels]
            initFlagRow = False
        else:
            baseTblFormatted = np.asarray(baseTblFormatted)
            
            colFormatted = np.concatenate((colFormatted, 
                                           hpad,
                                           baseTblFormatted),
                                           axis = 0)
            if labelFlagRow:
                extraLabels = [title, *baseTblLabels]
                tableLabels = tableLabels + extraLabels
    
    labelFlagRow = False
    if initFlagCol:
        tableFormatted = colFormatted
        vpad = np.asarray(['   ']*tableFormatted.shape[0])[:,None]
        tableHeaders = baseTblHeaders

        initFlagCol = False
    else:
        tableFormatted = np.concatenate((tableFormatted, 
                                       vpad,
                                       colFormatted),
                                       axis = 1)
        extraHeaders = ['   ', *baseTblHeaders]
        tableHeaders = tableHeaders + extraHeaders
    colFormatted = None
    
    
table = SimpleTable(
            tableFormatted,
            stubs = tableLabels,
            headers = tableHeaders,
            title = 'Policy experiments, calibrated models (left, original; right, US calibration)',
    )

print(table)

if save is True:
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables,mode=0o777)
    
    with open(save_path_tables + '/table_mc_calib.tex', 'w') as f_out:
        f_out.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                           replacements={"#": r"\#",
                                                          "$": r"$",
                                                          "%": r"\%",
                                                          "&": r"\&",
                                                          ">": r"$>$",
                                                          "_": r"_",
                                                          "|": r"$|$"}))

#------------------------------------------------------------------------------
# Full estimated model results
# Pick empirical setting (fixes dataset + begrs estimates)
empSettings = ['_gm',      # 0 - Great moderation
               '_crisis']  # 1 - Crisis period
                
# Simulated dataset from estimates
models = ['KS_sobol4000_base_mc',   # 0 - baseline expectations
          'KS_sobol4000_exp1_mc',   # 1 - AR4 exp
          'KS_sobol4000_exp2_mc',   # 2 - Accel. exp
          'KS_sobol4000_exp3_mc',   # 3 - Adapt. exp - Very volatile, high unemp
          # 'KS_sobol4000_exp4_mc',   # 4 - Extrapol. exp - gdp OK
          # 'KS_sobol4000_exp5_mc',   # 5 - Trend exp - seems OK
          'KS_sobol4000_exp7_mc']   # 6 - LAA exp

modelLabels = ['US est, baseline exp.',
               'US est, AR4 exp.',
               'US est, Accel. exp.',
               'US est, Adapt. exp.',
               # 'US est, Extrapol. exp.',
               # 'US est, Trend. exp.',
               'US est, LAA. exp.']

for model, modelLabel in zip(models, modelLabels):
    # Extract all data for the model (both empirical settings)
    rawDataModels = []
    for setting in empSettings:
        rawDataList, titleList = extractResults(model,setting)
        rawDataModels.append(rawDataList)

    # Generate individual table panels and tile them as required
    initFlagCol = True
    for rawDataList in rawDataModels:
        
        initFlagRow = True
        labelFlagRow = True
        for rawData, title in zip(rawDataList, titleList):
        
            baseTblFormatted, baseTblLabels, baseTblHeaders = baseTable(rawData)        
            baseTblFormatted = np.asarray(baseTblFormatted)
    
            if initFlagRow:
                hpad = np.asarray(['   ']*baseTblFormatted.shape[1])[None,:]
                colFormatted = np.concatenate((hpad,
                                               baseTblFormatted),
                                               axis = 0)
                if labelFlagRow:
                    tableLabels = [title, *baseTblLabels]
                initFlagRow = False
            else:
                # baseTblFormatted = np.asarray(baseTblFormatted)
                
                colFormatted = np.concatenate((colFormatted, 
                                               hpad,
                                               baseTblFormatted),
                                               axis = 0)
                if labelFlagRow:
                    extraLabels = [title, *baseTblLabels]
                    tableLabels = tableLabels + extraLabels
        
        labelFlagRow = False
        if initFlagCol:
            tableFormatted = colFormatted
            vpad = np.asarray(['   ']*tableFormatted.shape[0])[:,None]
            tableHeaders = baseTblHeaders
    
            initFlagCol = False
        else:
            tableFormatted = np.concatenate((tableFormatted, 
                                           vpad,
                                           colFormatted),
                                           axis = 1)
            extraHeaders = ['   ', *baseTblHeaders]
            tableHeaders = tableHeaders + extraHeaders
        colFormatted = None
        
    title = 'Policy experiments, {:s}, (left, {:s}; right, {:s})'.format(
                modelLabel,
                empSettings[0].replace('_',''),
                empSettings[1].replace('_',''))
    
    table = SimpleTable(
                tableFormatted,
                stubs = tableLabels,
                headers = tableHeaders,
                title = title,
        )
    
    print(table)
    
    if save is True:
        if not os.path.exists(save_path_tables):
            os.makedirs(save_path_tables,mode=0o777)
        
        with open(save_path_tables + '/table_mc_{:s}.tex'.format(model),
                  'w') as f_out:
            f_out.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                               replacements={"#": r"\#",
                                                             "$": r"$",
                                                             "%": r"\%",
                                                             "&": r"\&",
                                                             ">": r"$>$",
                                                             "_": r"_",
                                                             "|": r"$|$"}))    
