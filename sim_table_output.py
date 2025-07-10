# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:29:43 2023

This script produces the tables and figures presenting the empirical and 
simulated datasets (table 4 and figure 1 in the paper).

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import zlib
import pickle
from statsmodels.iolib.table import SimpleTable

import os
import pandas as pd
from matplotlib import pyplot as plt

save = False

# Setup latex output and load/save folders
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

sim_dir = 'simData'
empDataDir = 'us_data'
dataFile = 'empirical_dataset_new.txt'
datesFile = 'observation_dates.txt'
savePath = 'tables'
dropExit = True

models = ['KS_calib_orig_mc',
          'KS_calib_us_mc',
          'KS_sobol4000_base_mc_gm',
          'KS_sobol4000_base_mc_crisis']

policyFlag = 'F_norule_M_tr2'

# Load empirical data
emp_data = np.loadtxt(os.path.join(empDataDir, dataFile), delimiter="\t") 

#-----------------------------------------------------------------------------
# - PLOT empirical datasets
dates = pd.read_csv(os.path.join(empDataDir, datesFile),
                    header=None).to_numpy(dtype = 'datetime64')

fontSize = 40

inds = [7,6,9,2]
labels = ['Interest','Unemp.','Bank loss','Investment']
locs = ['upper right','upper left','upper left','lower left']

for ind, label, loc in zip(inds,labels,locs):

    series = emp_data[:,ind]
    sample1 = [0,93]
    sample2 = [29,140]
    
    y_max = max(series)*1.1
    y_min = min(series)-0.9
    y_range = y_max - y_min
    xlim_left = dates[0]
    xlim_right = dates[-1]
    
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1, 1, 1)
    ax.fill([dates[sample1[0]],dates[sample1[1]],dates[sample1[1]],dates[sample1[0]]], 
            [y_min,y_min,y_min+y_range*.667,y_min+y_range*.667], 
            color = 'r', alpha=0.15, label = r'Great moderation')
    ax.fill([dates[sample2[0]],dates[sample2[1]],dates[sample2[1]],dates[sample2[0]]], 
            [y_min,y_min,y_min+y_range*.333,y_min+y_range*.333], 
            color = 'b', alpha=0.15, label = r'Crisis')
    ax.plot(dates, series,'k', linewidth=2,label = label)
    ax.legend(loc=loc, frameon=False, prop={'size':fontSize})
    ax.set_ylim(top = y_max, bottom = y_min)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, y_min, ">k", ms=15, clip_on=False)
    ax.plot(xlim_left, y_max, "^k", ms=15, clip_on=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', pad=15, labelsize=fontSize)
    ax.tick_params(axis='y', pad=15, labelsize=fontSize)
        
    if save is True:
        plt.savefig("figures/color/plot_{:s}.pdf".format(label), 
                    format = 'pdf',bbox_inches="tight")

#-----------------------------------------------------------------------------
emp_data_gm = emp_data[0:93,:]
emp_data_crisis = emp_data[29:,:]

emp_data_mean = np.concatenate((np.mean(emp_data_gm,axis=0)[:,None],
                                np.mean(emp_data_crisis,axis=0)[:,None]),
                               axis=1)
emp_data_std = np.concatenate((np.std(emp_data_gm,axis=0)[:,None],
                                np.std(emp_data_crisis,axis=0)[:,None]),
                               axis=1)

sim_mean = np.zeros([emp_data.shape[1],len(models)])
sim_std = np.zeros([emp_data.shape[1],len(models)])

for index, mod_name in enumerate(models):

    # Load simulated data
    sim_path = os.path.join('K+S',mod_name, policyFlag, sim_dir, mod_name + '_data.pkl')
    
    # sim_path = 'K+S//' + mod_name + '//' + sim_dir + '//' + mod_name + '_data.pkl'
    print (' Training load path:      ' + sim_path)
    fil = open(sim_path,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData = pickle.loads(datas,encoding="bytes")

    for j in range (1000):
        dat = simData[j]
        
        if len(dat) > 0:
        
            if dropExit is True:
                dat = np.delete(dat,9,axis = 1)
                
            if j == 0:
                fulldat = np.copy(dat)
            else:
                fulldat = np.concatenate((fulldat,dat),axis = 0)
            
    sim_mean[:,index] = np.mean(fulldat,axis=0)
    sim_std[:,index] = np.std(fulldat,axis=0)

# Format diagnostic table
labels = [r'$\Delta y$',
          r' ',
          r'$\Delta c$',
          r' ',
          r'$\Delta i$',
          r' ',
          r'$\Delta L$',
          r' ',
          r'$\Delta nw$',
          r' ',
          r'$\Delta w$',
          r' ',
          r'u',
          r' ',
          r'r',
          r' ',
          r'$\pi$',
          r' ',
          r'lr',
          r' ']

header = ['Baseline Sim.', 'US Calib Sim.', 
          'GM Emp.', 'GM Sim.', 'Crisis Emp.', 'Crisis Sim.']

valuesFormatted = []
for empMeanRow, empStdRow, simMeanRow, simStdRow in zip(emp_data_mean,
                                                        emp_data_std,
                                                        sim_mean,
                                                        sim_std):
    rowMean = np.zeros(len(header))
    rowMean[0] = simMeanRow[0]
    rowMean[1] = simMeanRow[1]
    rowMean[2] = empMeanRow[0]
    rowMean[3] = simMeanRow[2]
    rowMean[4] = empMeanRow[1]
    rowMean[5] = simMeanRow[3]

    rowStd = np.zeros(len(header))
    rowStd[0] = simStdRow[0]
    rowStd[1] = simStdRow[1]
    rowStd[2] = empStdRow[0]
    rowStd[3] = simStdRow[2]
    rowStd[4] = empStdRow[1]
    rowStd[5] = simStdRow[3]

    rowMeanFormatted = []
    for cellValue in rowMean:
        cellValueFormatted = '{:3.3f}'.format(cellValue)   
        rowMeanFormatted.append(cellValueFormatted)
    valuesFormatted.append(rowMeanFormatted)
    
    rowStdFormatted = []
    for cellValue in rowStd:
        cellValueFormatted = '({:3.3f})'.format(cellValue)            
        rowStdFormatted.append(cellValueFormatted)
    valuesFormatted.append(rowStdFormatted)

table = SimpleTable(
        valuesFormatted,
        stubs=labels,
        headers=header,
        title='Comparison of simulated and empirical descriptive statistics',
    )

print(table)
if save is True:
    with open(savePath+'/table_descr_stats.tex','w') as f:
        f.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                       replacements={"#": r"\#",
                                                      "$": r"$",
                                                      "%": r"\%",
                                                      "&": r"\&",
                                                      ">": r"$>$",
                                                      "_": r"_",
                                                      "|": r"$|$"}))

#-----------------------------------------------------------------------------