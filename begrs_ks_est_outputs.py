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
diagnosticPlots = False         # Show diagnostic plots on main estiamte
expectationPlots = False        # Show expectation parameter plots
save = True
color = True
fontSize = 20

# Paths 
save_path_figs = 'figures'
save_path_tables = 'tables'
KSfolder = 'K+S'
dataPath = 'simData'
modelPath = 'models'

models = ['KS_sobol4000_base_cut',
          'KS_sobol4000_exp1_cut',
          'KS_sobol4000_exp2_cut',
          'KS_sobol4000_exp3_cut',
          # 'KS_sobol4000_exp4_cut',
          # 'KS_sobol4000_exp5_cut',
          'KS_sobol4000_exp7_cut']

rowNames = ['US mean est, basline exp.',
            'US est, AR(4) exp.',
            'US est, Accel. exp.',
            'US est, Adapt. exp.',
            # 'US est, Extrapol. exp.',
            # 'US est, Trend. exp.',
            'US est, A-A exp.']

modNames = ['Baseline',
            'AR(4)',
            'Accel.',
            'Adapt.',
            # 'Extrapol.',
            # 'Trend',
            'A-A']

estimates = ['estimates_full', 'estimates_gm', 'estimates_crisis']
estimateNames = ['Full sample', 'Great moderation', 'Crisis']

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

begrsEst = begrs()
begrsEst.load( modelPath + '/' + models[0]  + saveName)
parameter_range = begrsEst.parameter_range

# Generate diagnostic plots (if requested)
if diagnosticPlots:
    
    # Load NUTS Samples
    fil = open(modelPath + '/' + models[0] + saveName + '/' 
               + estimates[0] +'.pkl','rb')
    datas = fil.read()
    fil.close()
    results = pickle.loads(datas,encoding="bytes")
    flatSamples = results['samples']
    
    # Sample plot - visual check for autocorrelation in MCMC chains
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(flatSamples)
    
    # Corner plot  - visual check for any correlation in parameter estimates
    figure = corner(flatSamples)

#-----------------------------------------------------------------------------
# Plots for common parameter estimates across expectation specifications
print('Generating common parameter plots, full sample, by expectation')

subplotSpec = gridspec.GridSpec(ncols=4, nrows=5, hspace=0.25, wspace=0.2)
fig = plt.figure(figsize=(16,12))
ndim = len(variables)
for i in range(ndim):
    
    x_range = parameter_range[i,1] - parameter_range[i,0]
    xlim_left = parameter_range[i,0] - x_range*0.025
    xlim_right = parameter_range[i,1] + x_range*0.025
    y_max = 0

    ax = fig.add_subplot(subplotSpec[i])
    for j, model in enumerate(models):

        fil = open(modelPath + '/' + model + saveName + '/' 
                   + estimates[0] + '.pkl','rb')
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
    
    # Annotate plot to add parameter name
    plt.text(0.9,0.9, r'${:s}$'.format(variables[i]), 
             fontsize = fontSize,
             transform=ax.transAxes)
    
    # Format axes
    ax.set_ylabel(''),
    ax.set_xlabel('')
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
                 loc='lower center', ncol= 3,
                 frameon=False, prop={'size':fontSize})

if save is True:
    plt.savefig(save_path_figs + "/dens_by_exp.pdf", 
                format = 'pdf',bbox_inches="tight")

#-----------------------------------------------------------------------------
# Plots for common parameter estimates across empirical samples
print('Generating common parameter plots, baseline expectations, by sample')

subplotSpec = gridspec.GridSpec(ncols=4, nrows=5, hspace=0.25, wspace=0.2)
fig = plt.figure(figsize=(16,12))
theta_mean = np.zeros([len(variables),len(estimates)])

# Density plots per parameter
for i in range(ndim):
    x_range = parameter_range[i,1] - parameter_range[i,0]
    xlim_left = parameter_range[i,0] - x_range*0.025
    xlim_right = parameter_range[i,1] + x_range*0.025
    
    y_max = 0
    ax = fig.add_subplot(subplotSpec[i])

    for j, estimate in enumerate(estimates):

        fil = open(modelPath + '/' + models[0] + saveName + '/' 
                   + estimate + '.pkl','rb')
        datas = fil.read()
        fil.close()
        results = pickle.loads(datas,encoding="bytes")

        flatSamples = results['samples']
        theta_mean[:,j] = np.mean(flatSamples[:,0:len(variables)], axis = 0)
            
        d = flatSamples[:,i]  
        res = ax.hist(x=d, bins='fd', density = True, edgecolor = 'None', 
                      color = cvecDens[j], alpha=0.33,label = estimateNames[j])
    
        y_max_curr = 1.25*max(res[0])
        y_max = max(y_max, y_max_curr)
    
    ax.plot([params[i],params[i]], [0, y_max], linewidth=1, color = cvec, 
            alpha=0.6, label = r'Calibr.')
    
    # Annotate plot to add parameter name
    plt.text(0.9,0.9, r'${:s}$'.format(variables[i]), 
             fontsize = fontSize,
             transform=ax.transAxes)
    
    # Format axes
    ax.set_ylabel(''),
    ax.set_xlabel('')
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
                 loc='lower center', ncol= 4,
                 frameon=False, prop={'size':fontSize})
    
if save is True:
    plt.savefig(save_path_figs + "/dens_by_sample.pdf", format = 'pdf', 
                bbox_inches="tight")
    
#-----------------------------------------------------------------------------
# Plots for estimates of additional expectation parameters
if expectationPlots:

    models.pop(0)
    modNames.pop(0)
    
    print('Generating additional expectation plots on full estimates')
    
    for j, model in enumerate(models):
        
        # Load models for specific parameter range
        begrsEst = begrs()
        begrsEst.load( modelPath + '/' + model  + saveName)
        parameter_range = begrsEst.parameter_range
        
        # Load empirical Samples
        fil = open(modelPath + '/' + model + saveName + '/' 
                   + estimates[0] +'.pkl','rb')
        datas = fil.read()
        fil.close()
        results = pickle.loads(datas,encoding="bytes")
        
        flatSamples = results['samples']
        
        for i in range(ndim,flatSamples.shape[1]):
            d = flatSamples[:,i]
            x_range = parameter_range[i,1] - parameter_range[i,0]
            xlim_left = parameter_range[i,0] - x_range*0.025
            xlim_right = parameter_range[i,1] + x_range*0.025
            
            fig = plt.figure(figsize=(16,12))
            ax = fig.add_subplot(1, 1, 1)
            res = ax.hist(x=d, bins='fd', density = True, edgecolor = 'None', 
                          color = 'black', alpha=0.4, label = modNames[j])
            y_max = 1.25*max(res[0])
    
            ax.set_xlabel(r'$\theta_{:d}$'.format(i+1-ndim), 
                          fontdict = {'fontsize': fontSize})
            ax.set_ylabel(r'$p(\theta_{:d})$'.format(i+1-ndim), rotation=0, 
                          labelpad=100, fontdict = {'fontsize': fontSize})
            ax.xaxis.set_label_coords(.925, -.06)
            ax.yaxis.set_label_coords(-.10, .925)
            ax.axes.yaxis.set_ticks([])
            ax.legend(loc='best', frameon=False, prop={'size':fontSize})
        
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
            
            if save is True:
                plt.savefig(save_path_figs + "/dens_{:s}_ex{:d}.pdf".format(
                    model,i+1-ndim), format = 'pdf',
                    bbox_inches="tight")
#-----------------------------------------------------------------------------
# Latex table of estimates
l1 = max([len(var) for var in variableDescr])
l2 = max([len(var) for var in variables])
l3 = 7

rowStr = ('{:s} {:l1s} & ${:l2s}$ & {:l3g} & {:l3g} - {:l3g} & {:l3.3f} & {:l3.3f} & {:l3.3f}  \\\\')
rowStr = rowStr.replace('l1','{:d}'.format(l1))
rowStr = rowStr.replace('l2','{:d}'.format(l2))
rowStr = rowStr.replace('l3','{:d}'.format(l3))

tableStr = []
tableStr.append('\\begin{tabular}{lcrrrrr}')
tableStr.append('\\hline')
tableStr.append('\\B \\T  & & & Prior & \\multicolumn{3}{c}{Posterior Mean} \\\\')
tableStr.append('\\multicolumn{2}{c}{Parameter} \\B \\T & Baseline & Range & Full & GM & Crisis \\\\')
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
                                  theta_mean[i,0],
                                  theta_mean[i,1],
                                  theta_mean[i,2]))
    
tableStr.append('\\hline')
tableStr.append('\\end{tabular}')

print("\n".join(tableStr))
if save is True:
    with open(save_path_tables + '/table_mean_estimates.tex', 'w') as f_out:
        f_out.write("\n".join(tableStr))
    f_out.close
#-----------------------------------------------------------------------------