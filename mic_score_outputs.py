# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:16:07 2016

This script produces the MIC comparison for the US estimation of the K+S model
(table 5 in the paper).

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import os

from statsmodels.iolib.table import SimpleTable

#------------------------------------------------------------------------------
# Load/Save directories
load_dir = 'scores'
save_path_tables = 'tables'
save = True

# Empirical settings (fixes dataset + begrs estimates)
empSettings = ['_crisis',  # 0 - Crisis period
               '_gm']      # 1 - Great moderation

# Training methods (for robustness checks)
methods = ['high_L1',               # 0
           'low_L2',                # 1
           'high_L2']               # 2

lags = [1,2,2]
res = [6,3,6]

models = ['KS_calib_orig_mc',
          'KS_calib_us_mc',
          'KS_sobol4000_base_mc',
          'KS_sobol4000_exp1_mc',
          'KS_sobol4000_exp2_mc',
          'KS_sobol4000_exp3_mc',
           # 'KS_sobol4000_exp4_mc',
          # 'KS_sobol4000_exp5_mc',
          'KS_sobol4000_exp7_mc']

rowNames = ['Original calibration',
            'US calibration',
            'US est, baseline exp.',
            'US est, AR4 exp.',
            'US est, Accel. exp.',
            'US est, Adapt. exp.',
            # 'US est, Extrapol. exp.',
            # 'US est, Trend. exp.',
            'US est, A-A. exp.']

colNames = ['$\Delta y$',
            '$\Delta c$',
            '$\Delta i$',
            '$\Delta L$',
            '$\Delta nw$',
            '$\Delta w$',
            'u',
            'r',
            '$\pi$',
            'lr',
            'Aggr',
            'Diff.' ]

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
obs = 141

for empSetting in empSettings:
    for methodChoice, method in enumerate(methods):

        if empSetting == '_crisis':
            T = obs - 29 - lags[methodChoice] 
        elif empSetting == '_gm':
            T = 93 - lags[methodChoice] 
        tableRows = []
        
        for model in models:
                            
            # - Generate load/save paths and allocate arrays
            load_path = load_dir + '//estimates'  + empSetting + '//' + \
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
                
                # Get individual variable scores from the last variable 
                # in the first 10 measures
                if j < num_vars:
                    var_scores[:,j] += scores[:,-1]
                        
            tableRow[0:num_vars] = np.sum(var_scores,0)
            tableRow[-2] = sum(np.sum(scores_full,0))
            tableRows.append(tableRow)
        
        tableRows = np.asarray(tableRows)
        tableRows[:,-1] = tableRows[:,-2] - tableRows[0,-2]
        #----------------------------------------------------------------------
        # Format array as a simpletable
        
        tableFormatted = []
        cellStr = '{:8.2f}'
        for row in tableRows:
            rowFormatted = []
            for cell in row:
                rowFormatted.append(cellStr.format(cell))
            tableFormatted.append(rowFormatted)
        
        title = 'MIC scores, {:s} period, {:s} setting, $L_0$ = {:d}'.format(
                        empSetting.replace('_',''),            
                        method,
                        res[methodChoice]*T)

        table = SimpleTable(
                    tableFormatted,
                    stubs = rowNames,
                    headers = colNames,
                    title = title
            )
        
        print(table)
        
        if save is True:
            if not os.path.exists(save_path_tables):
                os.makedirs(save_path_tables,mode=0o777)
            
            with open(save_path_tables + '/table{:s}_{:s}.tex'.format(
                        empSetting, method), 'w') as f_out:
                f_out.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                                   data_aligns='r',
                                                   header_align='r',
                                                   replacements={"#": r"\#",
                                                                  "$": r"$",
                                                                  "%": r"\%",
                                                                  "&": r"\&",
                                                                  ">": r"$>$",
                                                                  "_": r"_",
                                                                  "|": r"$|$"}))
