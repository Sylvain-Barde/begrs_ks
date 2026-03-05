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

save = True

# Setup latex output and load/save folders
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

sim_dir = 'simData'
empDataDir = 'us_data'
dataFiles = ['empirical_dataset_annual.csv',
             'empirical_dataset_quarterly.csv',]
savePathFigures = 'figures/color'
savePath = 'tables'
dropExit = True

# Models for paper descriptive stats (short)
models = ['KS_calib_orig_mc',       
          'KS_calib_us_a_mc',
          'KS_calib_us_q_mc',
          'KS_sobol4000_exp7_a_mc', # LAA exp - annual
          'KS_sobol4000_exp7_q_mc'] # LAA exp - quarterly

# Models for paper descriptive stats (short)
models_full = ['KS_calib_orig_mc',
               'KS_calib_us_a_mc',
               'KS_calib_us_q_mc',
               'KS_sobol4000_exp1_a_mc',   # AR4 exp - annual
               'KS_sobol4000_exp1_q_mc',   # AR4 exp - quarterly
               'KS_sobol4000_exp2_a_mc',   # Accel. exp - annual
               'KS_sobol4000_exp2_q_mc',   # Accel. exp - quarterly
               'KS_sobol4000_exp3_a_mc',   # Adapt. exp - annual
               'KS_sobol4000_exp3_q_mc',   # Adapt. exp - quarterly
               'KS_sobol4000_exp7_a_mc', # LAA exp - annual
               'KS_sobol4000_exp7_q_mc'] # LAA exp - quarterly

policyFlag = 'F_norule_M_tr2'

# Load empirical dataframes
emp_data_dfs = []
for dataFile in dataFiles:
    emp_data_dfs.append(pd.read_csv(os.path.join(empDataDir, dataFile), 
                                   index_col=0))
#-----------------------------------------------------------------------------
# Empirical Plots
fontSize = 40

inds = ['r','u','lr','di', 'dL',
        'dy','dc','dnw','dw','dpi']
labels = ['Central bank policy rate',
          'Unemployment rate',
          'Share of net charge offs to total loans \& leases',
          'Log diff. of p.c. real investment',
          'Log diff. of p.c. real loans',
          
          'Log diff. of p.c. real GDP',
          'Log diff. of p.c. real consumption',
          'Log diff. of p.c. real net worth',
          'Log diff. of p.c. real compensation',
          'Log diff. of GDP deflator']

colors = ['tab:blue', 'tab:red']
flags = [0,1]

for ind, label in zip(inds, labels):
    
    # Find X and Y limits (for plot sizing)
    xlim_left = min(pd.to_datetime(emp_data_dfs[0].index))
    xlim_left = min(xlim_left,
                    min(pd.to_datetime(emp_data_dfs[1].index)))
    xlim_right = max(pd.to_datetime(emp_data_dfs[0].index))
    xlim_right = max(xlim_right, 
                  max(pd.to_datetime(emp_data_dfs[1].index)))
    
    y1_max = max(emp_data_dfs[0][ind])*1.2
    y1_min = min(emp_data_dfs[0][ind])-0.9
    y2_max = max(emp_data_dfs[1][ind])*1.2
    y2_min = min(emp_data_dfs[1][ind])-0.9
    
    # Generate plot on first axis (left)
    fig, ax1 = plt.subplots(figsize=(16,12))
    ax1.plot(pd.to_datetime(emp_data_dfs[0].index), 
             emp_data_dfs[0][ind], colors[0], linewidth=2)

    ax1.set_ylabel('\% annual', color=colors[0], fontsize=fontSize)
    ax1.tick_params(axis='x', pad=15, labelsize=fontSize)
    ax1.tick_params(axis='y', pad=15, labelsize=fontSize, 
                    labelcolor=colors[0])
    
    # Clone axis for second plot (on the right)
    ax2 = ax1.twinx()  
    ax2.plot(pd.to_datetime(emp_data_dfs[1].index), 
             emp_data_dfs[1][ind], colors[1], linewidth=2)

    ax2.set_ylabel('\% quarterly', color=colors[1], fontsize=fontSize) 
    ax2.tick_params(axis='y', pad=15, labelsize=fontSize,  
                    labelcolor=colors[1])

    # Tidy up axes and figure options once plots are done
    ax1.set_xlim(left = xlim_left,right = xlim_right)
    ax1.set_ylim(top = y1_max, bottom = y1_min)
    ax2.set_ylim(top = y2_max, bottom = y2_min)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_visible(False)

    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_visible(False)

    ax1.plot(xlim_left, y1_max, "^k", ms=15, clip_on=False)
    ax2.plot(xlim_right, y2_max, "^k", ms=15, clip_on=False)
    
    fig.suptitle(label, fontsize=fontSize)
    
    if save is True:
        if not os.path.exists(savePathFigures):
            os.makedirs(savePathFigures,mode=0o777)
        
        plt.savefig(savePathFigures + "/plot_{:s}.pdf".format(ind), 
                    format = 'pdf',bbox_inches="tight")
#------------------------------------------------------------------------------
# Descriptive stats table - paper
emp_data_a = emp_data_dfs[0].to_numpy()
emp_data_q = emp_data_dfs[1].to_numpy()

emp_data_mean = np.concatenate((np.mean(emp_data_a,axis=0)[:,None],
                                np.mean(emp_data_q,axis=0)[:,None]),
                                axis=1)
emp_data_std = np.concatenate((np.std(emp_data_a,axis=0)[:,None],
                                np.std(emp_data_q,axis=0)[:,None]),
                                axis=1)

sim_mean = np.zeros([emp_data_a.shape[1],len(models)])
sim_std = np.zeros([emp_data_a.shape[1],len(models)])

simDataList = []

for index, mod_name in enumerate(models):

    # Load simulated data
    sim_path = os.path.join('K+S', mod_name, policyFlag, sim_dir, 
                            mod_name + '_data.pkl')
    print (' Training load path:      ' + sim_path)
    fil = open(sim_path,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData = pickle.loads(datas,encoding="bytes")
    simDataList.append(simData)

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

header = ['Baseline Sim.', 'US A Calib Sim.', 'US Q Calib Sim.',
          'A Emp.', 'A Sim.', 'Q Emp.', 'Q Sim.']

valuesFormatted = []
for empMeanRow, empStdRow, simMeanRow, simStdRow in zip(emp_data_mean,
                                                        emp_data_std,
                                                        sim_mean,
                                                        sim_std):
    rowMean = np.zeros(len(header))
    rowMean[0] = simMeanRow[0]
    rowMean[1] = simMeanRow[1]
    rowMean[2] = simMeanRow[2]
    rowMean[3] = empMeanRow[0]
    rowMean[4] = simMeanRow[3]
    rowMean[5] = empMeanRow[1]
    rowMean[6] = simMeanRow[4] 

    rowStd = np.zeros(len(header))
    rowStd[0] = simStdRow[0]
    rowStd[1] = simStdRow[1]
    rowStd[2] = simStdRow[2]
    rowStd[3] = empStdRow[0]
    rowStd[4] = simStdRow[3]
    rowStd[5] = empStdRow[1]
    rowStd[6] = simStdRow[4]

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
    if not os.path.exists(savePath):
        os.makedirs(savePath,mode=0o777)
    
    with open(savePath+'/table_descr_stats.tex','w') as f:
        f.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                        replacements={"#": r"\#",
                                                      "$": r"$",
                                                      "%": r"\%",
                                                      "&": r"\&",
                                                      ">": r"$>$",
                                                      "_": r"_",
                                                      "|": r"$|$"}))
        
#------------------------------------------------------------------------------
# Descriptive stats table - Full simulations
sim_mean = np.zeros([emp_data_a.shape[1],len(models_full)])
sim_std = np.zeros([emp_data_a.shape[1],len(models_full)])

simDataList = []

for index, mod_name in enumerate(models_full):

    # Load simulated data
    sim_path = os.path.join('K+S', mod_name, policyFlag, sim_dir, 
                            mod_name + '_data.pkl')
    print (' Training load path:      ' + sim_path)
    fil = open(sim_path,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData = pickle.loads(datas,encoding="bytes")
    simDataList.append(simData)

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

header = ['Baseline Sim.', 'US A Calib Sim.', 'US Q Calib Sim.',
          'A AR4', 'Q AR4', 
          'A Accel.', 'Q Accel.',
          'A Adapt.', 'Q Adapt.', 
          'A A-A.', 'Q A-A.']

valuesFormatted = []
for simMeanRow, simStdRow in zip(sim_mean,
                                 sim_std):
    rowMean = np.zeros(len(header))
    rowMean[0] = simMeanRow[0]
    rowMean[1] = simMeanRow[1]
    rowMean[2] = simMeanRow[2]
    rowMean[3] = simMeanRow[3]
    rowMean[4] = simMeanRow[4]
    rowMean[5] = simMeanRow[5]
    rowMean[6] = simMeanRow[6] 
    rowMean[7] = simMeanRow[7] 
    rowMean[8] = simMeanRow[8] 
    rowMean[9] = simMeanRow[9] 
    rowMean[10] = simMeanRow[10] 

    rowStd = np.zeros(len(header))
    rowStd[0] = simStdRow[0]
    rowStd[1] = simStdRow[1]
    rowStd[2] = simStdRow[2]
    rowStd[3] = simStdRow[3]
    rowStd[4] = simStdRow[4]
    rowStd[5] = simStdRow[5]
    rowStd[6] = simStdRow[6]
    rowStd[7] = simStdRow[7]
    rowStd[8] = simStdRow[8]
    rowStd[9] = simStdRow[9]
    rowStd[10] = simStdRow[10]

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
        title='Comparison of simulated descriptive statistics - all models',
    )

print(table)
if save is True:
    if not os.path.exists(savePath):
        os.makedirs(savePath,mode=0o777)
    
    with open(savePath+'/table_descr_stats_full.tex','w') as f:
        f.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                        replacements={"#": r"\#",
                                                      "$": r"$",
                                                      "%": r"\%",
                                                      "&": r"\&",
                                                      ">": r"$>$",
                                                      "_": r"_",
                                                      "|": r"$|$"}))
#-----------------------------------------------------------------------------
# Simulated Plots

pick = 500                              # Choose 1 out of 1000
xlim_left = 0                           # Start point
xlim_right = 150                        # Stop point
sim_series_a = simDataList[-2][pick]
sim_series_q = simDataList[-1][pick]

labels = [r'$\Delta y$',
          r'$\Delta c$',
          r'$\Delta i$',
          r'$\Delta L$',
          r'$\Delta nw$',
          r'$\Delta w$',
          r'u',
          r'r',
          r'$\pi$',
          r'lr']


for ind, label in enumerate(labels):


    y_max = max(max(sim_series_a[xlim_left:xlim_right,ind]),
                 max(sim_series_q[xlim_left:xlim_right,ind]))
    y_min = min(min(sim_series_a[xlim_left:xlim_right,ind]),
                 min(sim_series_q[xlim_left:xlim_right,ind]))   
    y_range = (y_max - y_min)*1.3
    y_max*=1.2
    y_min = y_max - y_range
        
    # Generate plot on first axis (left)
    fig, ax = plt.subplots(figsize=(16,12))
    ax.plot(sim_series_a[xlim_left:xlim_right,ind], 
             colors[0], linewidth=2, label = 'annual')
    ax.plot(sim_series_q[xlim_left:xlim_right,ind],
             colors[1], linewidth=2, label = 'quarterly')


    ax.set_ylabel(label, fontsize=fontSize)
    ax.tick_params(axis='x', pad=15, labelsize=fontSize)
    ax.tick_params(axis='y', pad=15, labelsize=fontSize)


    # Tidy up axes and figure options once plots are done
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.set_ylim(top = y_max, bottom = y_min)
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)    
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_visible(False)

    ax.plot(xlim_left, y_max, "^k", ms=15, clip_on=False)
    ax.plot(xlim_right, y_min, ">k", ms=15, clip_on=False)

    leg = fig.legend(*ax.get_legend_handles_labels(), 
                     loc='upper right', ncol= 2,
                     frameon=False, prop={'size':fontSize})

    if save is True:
        plt.savefig(savePathFigures + "/plot_sim_{:d}.pdf".format(ind), 
                    format = 'pdf',bbox_inches="tight")

#------------------------------------------------------------------------------