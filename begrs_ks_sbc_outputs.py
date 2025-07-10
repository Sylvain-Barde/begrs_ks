# -*- coding: utf-8 -*-
"""
Created on Wed Oct 2 16:48:44 2024

This script produces the plots for the Simulation-Based Calibration 
diagnostics on the convergence of the posterior obtained via BEGRS in 
estimation of the K+S model on US macroeconomic data (figures 2, 3, 6 and 7 in
the paper).

@author: Sylvain Barde, University of Kent

"""

import os
import pickle
import zlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import binom

# SBC run configurations
save = True
color = True

save_path_figs = 'figures'
dataPath = 'simData'
sbcPath = 'sbc'
fontSize = 20

modelTags = ['KS_sobol4000_base',   # modelChoice == 0
             'KS_sobol4000_exp1',   # modelChoice == 1
             'KS_sobol4000_exp2',   # modelChoice == 2
             'KS_sobol4000_exp3',   # modelChoice == 3
             # 'KS_sobol4000_exp4',   # modelChoice == 4
             # 'KS_sobol4000_exp5',   # modelChoice == 5
             'KS_sobol4000_exp7']   # modelChoice == 6

modelNames = ['Baseline',
              'AR(4)',
              'Accel.',
              'Adapt.',
              # 'Extrapol.',
              # 'Trend',
              'A-A']

cvecDens = ['black','red','blue','green','cyan','magenta','gold']

# Select training data parameters (to identify trained model)
numInducingPts = 250     # Number of inducing points
batchSize = 20000        # Size of training minibatches (mainly a speed issue)
numIter = 100            # Number of epoch iterations 
learning_rate = 0.0005   # Learning rate

# Labels
labels = [r'o_1, o_2',
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

#-----------------------------------------------------------------------------
# Setup latex output and saving folder
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Setup colors & adapt save folder
if color is True:
    cvec = 'b'
    save_path_figs += '/color'
else:
    cvec = 'k'
    save_path_figs += '/bw'

# Generate BEGRS config string
lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
begrsTag = 'ind_{:d}_lr_{:s}_ep_{:d}'.format(numInducingPts,
                                            lrnStr,
                                            numIter)
savePathSBC = save_path_figs + '/sbc_{:s}'.format(begrsTag)

# Create save folder if required
if save is True:
    if not os.path.exists(savePathSBC):
        os.makedirs(savePathSBC,mode=0o777)

#------------------------------------------------------------------------------
# ESS plots
y_max=0
# subplotSpec = gridspec.GridSpec(ncols=3, nrows=3, hspace=0.25, wspace=0.2)
subplotSpec = gridspec.GridSpec(ncols=3, nrows=2, hspace=0.25, wspace=0.2)

fig = plt.figure(figsize=(16,9))
ndim = len(modelTags)
axList = []
for i in range(ndim):

    fil = open(sbcPath + 
                '/{:s}_{:s}.pkl'.format(modelTags[i],begrsTag),
                'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    sbcData = pickle.loads(datas,encoding="bytes")
    hist = sbcData['hist']
    
    distESS = sbcData['posteriorESS']
    ESS_max = max(distESS)
    ESSbins = np.arange(0,ESS_max,step = 10)
    ax = fig.add_subplot(subplotSpec[i])
    res = ax.hist(distESS,bins = ESSbins, color = cvecDens[0],
                  alpha=0.33)
    
    # Update maximum (must be common across plots)
    y_max_curr = 1.25*max(res[0])
    y_max = max(y_max, y_max_curr)
    
    # Format axes - stuff that doesn't depend on y_max
    ax.axes.yaxis.set_ticks([])
    ax.set_xlim(left = 0,right = ESS_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=fontSize)    
    axList.append(ax)

# Formatting all the stuff that depended on the maximum y-axis in particular)
for i, ax in enumerate(axList):
    ax.plot([50,50], [0, y_max], linewidth=1, color = 'r', 
                alpha=0.6, label = r'Calibr.')

    # Annotate plot to add parameter name
    plt.text(0.5,0.85, '{:s}'.format(modelNames[i]), 
             fontsize = fontSize,
             transform=ax.transAxes)

    ax.set_ylim(top = y_max, bottom = 0)
    ax.plot(ESS_max, 0, ">k", clip_on=False)
    ax.plot(0, y_max, "^k", clip_on=False)

if save is True:
    plt.savefig(savePathSBC +  
                '/sbc_ess_plot.pdf', format = 'pdf',bbox_inches='tight')

#------------------------------------------------------------------------------
# Generate SBC rank plots by model
for modelTag, modelName in zip(modelTags,modelNames):
    fil = open(sbcPath + 
                '/{:s}_{:s}.pkl'.format(modelTag,begrsTag),
                'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    sbcData = pickle.loads(datas,encoding="bytes")
    hist = sbcData['hist']
    
    # Generate confidence intervals based on binomial counts.
    bins = np.arange(hist.shape[0])
    numObs = np.sum(hist,axis = 0)[0]
    numBins = hist.shape[0]
    
    pad = 4;
    confidenceBoundsX = [-0.5-pad,
                         -0.5-pad/2,
                         -0.5-pad,
                         numBins+0.5+pad,
                         numBins+0.5+pad/2,
                         numBins+0.5+pad]
    
    confidenceBoundsY = [binom.ppf(0.005, numObs, 1/numBins),
                         binom.ppf(0.5, numObs, 1/numBins),
                         binom.ppf(0.995, numObs, 1/numBins),
                         binom.ppf(0.995, numObs, 1/numBins),
                         binom.ppf(0.5, numObs, 1/numBins),
                         binom.ppf(0.005, numObs, 1/numBins)]
    
    x_range = max(confidenceBoundsX) - min(confidenceBoundsX)
    xlim_left = min(confidenceBoundsX) - x_range*0.025
    xlim_right = max(confidenceBoundsX) + x_range*0.025
    
    # Generate Rank Plot for each parameter
    subplotSpec = gridspec.GridSpec(ncols=4, nrows=5)
    fig = plt.figure(figsize=(16,12))
    
    for i in range(len(labels)):
        
        ax = fig.add_subplot(subplotSpec[i])
    
        confidenceBounds = ax.fill(confidenceBoundsX,
                                   confidenceBoundsY,
                                   'silver', label = '$95\%$ conf.')
        sbcHist = ax.bar(bins-0.5, 
                     hist[:,i],
                     width=1, 
                     align="edge",
                     edgecolor = None,
                     color = cvec, 
                     alpha = 0.4, 
                     label = '{:s}'.format(modelName))
            
        # Set y axis limits
        yMax = 3*max(confidenceBoundsY)
    
        # Annotate
        plt.text(0.9,0.9, r'${:s}$'.format(labels[i]), 
                 fontsize = fontSize,
                 transform=ax.transAxes)
    
        ax.set_ylabel(''),
        ax.set_xlabel('')
        ax.axes.yaxis.set_ticks([])
    
        ax.set_ylim(top = yMax, bottom = 0)
        ax.set_xlim(left = xlim_left,right = xlim_right)
        ax.plot(xlim_right, 0, ">k", ms=8, clip_on=False)
        ax.plot(xlim_left, yMax, "^k", ms=8, clip_on=False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params(axis='x', labelsize=fontSize)    
        ax.tick_params(axis='y', labelsize=fontSize) 
            
    leg = fig.legend(*ax.get_legend_handles_labels(), 
                     loc='lower center', ncol= 2,
                     frameon=False, prop={'size':fontSize})
            
    if save is True:
        plt.savefig(savePathSBC +  
                    '/{:s}_sbc.pdf'.format(modelTag), 
                    format = 'pdf',bbox_inches='tight')