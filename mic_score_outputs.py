# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:16:07 2016

This script produces the MIC comparison for the US estimation of the K+S model
(table 5 in the paper).

@author: Sylvain Barde, University of Kent
"""

import pandas as pd
import numpy as np
import pickle
import os

#------------------------------------------------------------------------------
# Load/Save directories
empDataPath = 'us_data/empirical_dataset_'
load_dir = 'scores'
save_path_tables = 'tables'
save = True

# Empirical settings (fixes dataset + begrs estimates)
dataFreqs = {'annual':{'tag':'a',
                        'path':'annual'},
             'quarterly':{'tag':'q',
                       'path':'quarterly'}
             }

# Training methods (for robustness checks)
methods = ['high_L1',               # 0
           'low_L2',                # 1
           'high_L2']               # 2
lags = [1,2,2]
res = [6,3,6]

models = ['KS_calib_orig_mc',
          'KS_calib_us',         # US-calibrated
          'KS_sobol4000_base',   # baseline expectations
          'KS_sobol4000_exp1',   # AR4 exp
          'KS_sobol4000_exp2',   # Accel. exp
          'KS_sobol4000_exp3',   # Adapt. exp
          'KS_sobol4000_exp4',   # Extrapol. exp
          'KS_sobol4000_exp5',   # Trend exp
          'KS_sobol4000_exp7']   # LAA exp

rowNames = ['Original calibration',
            'US calibration',
            'US est, baseline exp.',
            'US est, AR4 exp.',
            'US est, Accel. exp.',
            'US est, Adapt. exp.',
            'US est, Extrapol. exp.',
            'US est, Trend. exp.',
            'US est, LAA. exp.']

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

num_vars = len(var_vec_base[0])

for key, dataFreq in dataFreqs.items():
    empPath = empDataPath + dataFreq['path'] + '.csv'
    emp_data = pd.read_csv(empPath, index_col=0).to_numpy()
    obs = emp_data.shape[0]
    
    for methodChoice, method in enumerate(methods):

        T = obs - lags[methodChoice] 
        tableRows = []
        
        for mod_ind, model in enumerate(models):
            
            load_path = load_dir + '//estimates_'  + dataFreq['path'] + '//' + \
                method + '//' + model
            
            var_scores = np.zeros([T,num_vars])
            scores_full = np.zeros([T,num_vars])
            tableRow = np.zeros(num_vars+2)
            
            for j, var_vec in enumerate(var_vec_base):
            
                num_vars = len(var_vec)
                var_str = ''
                for var_i in var_vec:
                    var_str = var_str + str(var_i)
                    
                fil = open(load_path + '//scores_var_' + var_str + '.pkl','rb')
                datas = fil.read()
                fil.close()
                
                results = pickle.loads(datas,encoding="bytes")
                setting_scores = results[1]
                
                # Extract scores from saved raw data
                scores = np.zeros([T, num_vars])
                for k in range(num_vars):
                    scores[:,k] = setting_scores[k]
            
                scores_full += scores/len(var_vec_base)
                
                # Get individual variable scores from the last variable in the first 
                # 10 measures
                if j < num_vars:
                    var_scores[:,j] += scores[:,-1]
                        
            tableRow[0:num_vars] = np.sum(var_scores,0)
            tableRow[-2] = sum(np.sum(scores_full,0))
            tableRows.append(tableRow)
        #----------------------------------------------------------------------
        tableRows = np.asarray(tableRows)
        tableRows[:,-1] = tableRows[:,-2] - tableRows[0,-2]
        
        if save is True:        
            if not os.path.exists(save_path_tables):
                os.makedirs(save_path_tables,mode=0o777)
        
        l1 = max([len(var) for var in rowNames])
        
        varName = '{:l1s} '.replace('l1','{:d}'.format(l1))
        cellStr = '& {:8.2f} '
        
        tableStr = []
        tableStr.append('\\begin{tabular}{lrrrrrrrrrrrr}')
        tableStr.append('\\hline')
        tableStr.append(r'\B \T & $\Delta y$ & $\Delta c$ & $\Delta i$ & $\Delta L$ & $\Delta nw$ & $\Delta w$ & u & r & $\pi$ & lr & Aggr. & Diff. \\')
        tableStr.append('\\hline')
        tableStr.append(r'\multicolumn{13}{l}{\emph{' + 
                        '{:s}'.format(dataFreq['path']) + 
                        r' data}, $L_0$ = ' + 
                        '{:d}'.format(res[methodChoice]*T) + 
                        r' \B} \\')
        for i, tableRow in enumerate(tableRows):
            if i == 0:
                pad = '\\T '
            elif i == len(tableRows)-1:
                pad = '\\B '
            else:
                pad = '   '
                    
            rowStr = pad + varName.format(rowNames[i])
            for cell in tableRow:
                rowStr += cellStr.format(cell)
                
            rowStr += '\\\\'
            tableStr.append(rowStr)
            
        tableStr.append('\\hline')
        tableStr.append('\\end{tabular}')
        print("\n".join(tableStr))
        print(res[methodChoice]*T)
        
        if save is True:
            if not os.path.exists(save_path_tables):
                os.makedirs(save_path_tables,mode=0o777)
            
            with open(save_path_tables + '/table_{:s}_{:s}.tex'.format(
                        dataFreq['tag'], method), 'w') as f_out:
                f_out.write("\n".join(tableStr))
            f_out.close
