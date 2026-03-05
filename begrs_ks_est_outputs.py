# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:24:25 2021

This script produces the plots and tables for the posterior parameters samples
obtained with NUTS on the BEGRS surrogate estimation on US macroeconomic data.
(figures 4 and 5, table 3 in the paper).

@author: Sylvain Barde, University of Kent
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

import os
from corner import corner
from begrs import begrs

# Select empirical dataset
diagnosticPlots = True         # Show diagnostic plots on main estimate
save = True
color = True
fontSize = 20

# Paths 
save_path_figs = 'figures'
save_path_tables = 'tables'
KSfolder = 'K+S'
dataPath = 'simData'
modelPath = 'models'

modelTags = ['KS_sobol4000_base',   # modelChoice == 0
             'KS_sobol4000_exp1',   # modelChoice == 1
             'KS_sobol4000_exp2',   # modelChoice == 2
             'KS_sobol4000_exp3',   # modelChoice == 3
             'KS_sobol4000_exp7']   # modelChoice == 6

rowNames = ['US mean est, basline exp.',
            'US est, AR(4) exp.',
            'US est, Accel. exp.',
            'US est, Adapt. exp.',
            'US est, A-A exp.']

modNames = ['Baseline',
            'AR(4)',
            'Accel.',
            'Adapt.',
            'A-A']

dataFreqs = {'annual':{'tag':'a',
                        'path':'annual'},
             'quarterly':{'tag':'q',
                       'path':'quarterly'}
            }
dropExit = True     # Drop the firm exit variable (not used in new data)

cvecDens = ['black','red','blue','green','cyan','magenta','gold']

numInducingPts = 250     # Number of inducing points
batchSize = 20000        # Size of training minibatches (mainly a speed issue)
numIter = 100            # Number of epoch iterations 
learning_rate = 0.0005   # Learning rate
#-----------------------------------------------------------------------------
# Setup latex output and saving folder
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Setup colors & adapt save folder
if color is True:
    cvec = 'r'
    save_path_figs += '/color'
else:
    cvec = 'k'
    save_path_figs += '/bw'

# Create save folder if required
if save is True:
    if not os.path.exists(save_path_figs):
        os.makedirs(save_path_figs,mode=0o777)
        
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables,mode=0o777)

# Labels
variables = [r'o_1, o_2',
             r'\alpha_1,\beta_1',
             r'\alpha_2',
             r'\beta_2^+',
             r'\delta_{\mu}',
             r'\chi',
             r'\omega_1',
             r'\omega_2',
             r'\psi_1',
             r'\psi_2',
             r'\psi_3',
             r'\iota',
             r'u',
             r'\phi_b',
             r'repay',
             r'\gamma_{\pi}',
             r'\gamma_U',
             r'\gamma_B']

params = np.array([
                0.3,
                3,
                2,
                2,
                0.01,
                -1,
                1,
                1,
                0.05,
                1,
                0.05,
                0.1,
                0.75,
                0.5,
                0.33,
                1.1,
                1.1,
                1])

# Variable description (for table)
variableDescr = ['Elasticity of innov./imit. w.r.t R\\&D',
                'Beta dist. parameters for innov. draws',
                'Beta dist. parameters for imit. draws',
                'Beta dist. parameters for imit. draws',
                'Sensitivity of mark-up adjustment',
                'Replicator dynamics coefficient',
                'Competitiveness weight of price',
                'Competitiveness weight of unfilled demand',
                'Share of inflation passed to wages',
                'Elasticity of wages to productivity',
                'Elasticity of wages to unemployment',
                'Share of exp. demand held in inventory',
                'Planned utilization of machinery',
                'Lending sens. to net worth vs. turnover',
                'Desired share of debt to pay back',
                'Taylor rule sensitivity to target infl.',
                'Taylor rule sensitivity to target unemp.',
                'Bank sensitivity to financial fragility']

#-----------------------------------------------------------------------------
# Load parameter ranges & produce diagnostic plots if required.
# Create a begrs estimation object, load existing model
saveName = '/begrs_batch_{:d}_ind_{:d}_lr_{:s}_ep_{:d}'.format(
                batchSize,
                numInducingPts,
                str(learning_rate)[2:],
                numIter) 

if dropExit is True:
    namePad = '_cut'
else:
    namePad = ''

modelTag = modelTags[0] +'_{:s}'.format(dataFreqs['annual']['tag'])
begrsEst = begrs()
begrsEst.load( modelPath + '/{:s}/'.format(dataFreqs['annual']['path']) + 
               modelTag + namePad + saveName)
parameter_range = begrsEst.parameter_range

# Generate diagnostic plots (if requested)
if diagnosticPlots:

    for freq in dataFreqs:
    
        # Load NUTS Samples
        modelTag = modelTags[0] +'_{:s}'.format(dataFreqs[freq]['tag'])
        fil = open(modelPath + '/{:s}/'.format(dataFreqs[freq]['path']) + 
                   modelTag + namePad  + saveName + '/' + 
                   'estimates.pkl','rb')
        datas = fil.read()
        fil.close()
        results = pickle.loads(datas,encoding="bytes")
        flatSamples = results['samples']
        
        # Sample plot - visual check for autocorrelation in MCMC chains
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(flatSamples)
        
        # Corner plot - visual check for any correlation in parameter estimates
        figure = corner(flatSamples)

#-----------------------------------------------------------------------------
# Plots for common parameter estimates across expectation specifications
print('Generating common parameter plots, both samples, by expectation')
for freq in dataFreqs:
    
    fig = plt.figure(figsize=(16,12),constrained_layout=True)
    subplotSpec = gridspec.GridSpec(ncols=4, nrows=5, figure = fig)
    ndim = len(variables)
    for i in range(ndim):
        
        x_range = parameter_range[i,1] - parameter_range[i,0]
        xlim_left = parameter_range[i,0] - x_range*0.025
        xlim_right = parameter_range[i,1] + x_range*0.025
        y_max = 0
    
        ax = fig.add_subplot(subplotSpec[i])
        for j, model in enumerate(modelTags):
    
            modelTag = model +'_{:s}'.format(dataFreqs[freq]['tag'])
            fil = open(modelPath + '/{:s}/'.format(dataFreqs[freq]['path']) + 
                       modelTag + namePad  + saveName + '/' + 
                       'estimates.pkl','rb')
            datas = fil.read()
            fil.close()
            results = pickle.loads(datas,encoding="bytes")
            mode = results['mode']
            flatSamples = results['samples']
    
            d = flatSamples[:,i]  
            res = ax.hist(x=d, bins='fd', density = True, edgecolor = 'None', 
                          color = cvecDens[j], alpha=0.33, label = modNames[j])
        
            y_max_curr = 1.25*max(res[0])
            y_max = max(y_max, y_max_curr)
        
        ax.plot([params[i],params[i]], [0, y_max], linewidth=1, color = cvec, 
                alpha=0.6, label = r'Calibr.')
        
        # Format axes
        ax.set_ylabel('dens.',
                      fontdict = {'fontsize': fontSize})
        ax.set_xlabel(r'${:s}$'.format(variables[i]),
                      fontdict = {'fontsize': fontSize})
        ax.axes.yaxis.set_ticks([])
        
        ax.set_ylim(top = y_max, bottom = 0)
        ax.set_xlim(left = xlim_left,right = xlim_right)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, y_max, "^k", clip_on=False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params(axis='x', labelsize=fontSize)
        ax.tick_params(axis='y', labelsize=fontSize)
        
    leg = fig.legend(*ax.get_legend_handles_labels(), 
                     loc='lower right', ncol= 3,
                     frameon=False, prop={'size':fontSize})
    
    if save is True:
        plt.savefig(save_path_figs + "/dens_by_exp_{:s}.pdf".format(freq), 
                    format = 'pdf',bbox_inches="tight")

#-----------------------------------------------------------------------------
# Plots for common parameter estimates across empirical samples
print('Generating common parameter plots, LAA expectations, by sample')

model_thetas = []

for modelTag_base, modName in zip(modelTags, modNames):

    fig = plt.figure(figsize=(16,12),constrained_layout=True)
    subplotSpec = gridspec.GridSpec(ncols=4, nrows=5, figure = fig)
    theta_mean = np.zeros([len(variables),len(dataFreqs)])
    
    # Density plots per parameter
    for i in range(ndim):
        x_range = parameter_range[i,1] - parameter_range[i,0]
        xlim_left = parameter_range[i,0] - x_range*0.025
        xlim_right = parameter_range[i,1] + x_range*0.025
        
        y_max = 0
        ax = fig.add_subplot(subplotSpec[i])
    
        for j, freq in enumerate(dataFreqs):
    
            # modelTag = modelTags[-1] +'_{:s}'.format(dataFreqs[freq]['tag'])
            modelTag = modelTag_base +'_{:s}'.format(dataFreqs[freq]['tag'])
            fil = open(modelPath + '/{:s}/'.format(dataFreqs[freq]['path']) + 
                       modelTag + namePad  + saveName + '/' + 
                       'estimates.pkl','rb')
            
            datas = fil.read()
            fil.close()
            results = pickle.loads(datas,encoding="bytes")
    
            flatSamples = results['samples']
            theta_mean[:,j] = np.mean(flatSamples[:,0:len(variables)], axis = 0)
                
            d = flatSamples[:,i]  
            res = ax.hist(x=d, bins='fd', density = True, edgecolor = 'None', 
                          color = cvecDens[j], alpha=0.33,
                          label = freq)
        
            y_max_curr = 1.25*max(res[0])
            y_max = max(y_max, y_max_curr)
        
        model_thetas.append(theta_mean)
        ax.plot([params[i],params[i]], [0, y_max], linewidth=1, color = cvec, 
                alpha=0.6, label = r'Calibr.')
        
        # Format axes
        ax.set_ylabel('dens.',
                      fontdict = {'fontsize': fontSize})
        ax.set_xlabel(r'${:s}$'.format(variables[i]),
                      fontdict = {'fontsize': fontSize})
        ax.axes.yaxis.set_ticks([])
        
        ax.set_ylim(top = y_max, bottom = 0)
        ax.set_xlim(left = xlim_left,right = xlim_right)
        ax.plot(xlim_right, 0, ">k", clip_on=False)
        ax.plot(xlim_left, y_max, "^k", clip_on=False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params(axis='x', labelsize=fontSize)
        ax.tick_params(axis='y', labelsize=fontSize)
        
    leg = fig.legend(*ax.get_legend_handles_labels(), 
                     loc='lower right', ncol= 3,
                     frameon=False, prop={'size':fontSize})
        
    if save is True:
        plt.savefig(save_path_figs + "/dens_by_sample_{:s}.pdf".format(
                    modName), format = 'pdf', 
                    bbox_inches="tight")

#-----------------------------------------------------------------------------
# Latex table of estimates - Uses AA for the table
l1 = max([len(var) for var in variableDescr])
l2 = max([len(var) for var in variables])
l3 = 6

rowStr = ('{:s} {:l1s} & ${:l2s}$ & {:l3g} & {:l3g} - {:l3g} & {:l3.3f} & {:l3.3f}  \\\\')
rowStr = rowStr.replace('l1','{:d}'.format(l1))
rowStr = rowStr.replace('l2','{:d}'.format(l2))
rowStr = rowStr.replace('l3','{:d}'.format(l3))

tableStr = []
tableStr.append('\\begin{tabular}{lcrrrr}')
tableStr.append('\\hline')
tableStr.append('\\T  & & & Prior & \\multicolumn{2}{c}{Posterior Mean} \\\\')
tableStr.append('\\multicolumn{2}{c}{Parameter} \\B & Baseline & Range & Annual & Quartely \\\\')
tableStr.append('\\hline')
for i in range(ndim):
    if i == 0:
        pad = '\\T '
    elif i == ndim-1:
        pad = '\\B '
    else:
        pad = '   '
    
    tableStr.append(rowStr.format(pad,
                                  variableDescr[i],
                                  variables[i],
                                  params[i],
                                  parameter_range[i,0],
                                  parameter_range[i,1],
                                  model_thetas[-1][i,0],  # last model, so AA
                                  model_thetas[-1][i,1])) # last model, so AA
                                  # theta_mean[i,0],
                                  # theta_mean[i,1]))
    
tableStr.append('\\hline')
tableStr.append('\\end{tabular}')

print("\n".join(tableStr))
if save is True:
    with open(save_path_tables + '/table_mean_estimates.tex', 'w') as f_out:
        f_out.write("\n".join(tableStr))
    f_out.close
#-----------------------------------------------------------------------------