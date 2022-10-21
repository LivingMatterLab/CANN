#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:42:09 2022

@author: kevinlinka
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import r2_score
ColorS = [0.5, 0.00, 0.0]
ColorE = [0.8, 0.00, 0.0]
ColorI = [1.0, 0.65, 0.0]
ColorR = [0.0, 0.00, 0.7]   
    

#%% Uts
def r2_score_own(Truth, Prediction):
    try:
        R2 = r2_score(Truth,Prediction)
        R2 = max(R2,0.0)
    except ValueError:
        R2 = np.nan
    return R2



def GetZeroList(model_weights):
    model_zeros = []
    for i in range(len(model_weights)):
        model_zeros.append(np.zeros_like(model_weights[i]))
    return model_zeros



def color_map(ax2, gamma_ss, model_SS, model_weights, Psi_model, cmaplist, terms):
    predictions = np.zeros([gamma_ss.shape[0],terms])
    
    for i in range(len(model_weights)-1):
        
        model_plot = GetZeroList(model_weights)
        model_plot[i] =  model_weights[i]
        model_plot[-1][i] = model_weights[-1][i]
        Psi_model.set_weights(model_plot)
        
        lower = np.sum(predictions,axis=1)
        upper = lower + model_SS.predict(gamma_ss)[:].flatten()
        predictions[:,i] = model_SS.predict(gamma_ss)[:].flatten()
        ax2.fill_between(gamma_ss[:], lower.flatten(), upper.flatten(), zorder=i+1, alpha=1.0, color=cmaplist[i])



#%% Plotting Loss
 



def plotLoss(axe, history):   
    axe.plot(history)
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    
    
        
#%% Plotting color map

    
    
def plotMap(ax2, lam, P, model_plot, terms, Psi_model,  model_weights, label):
    
    cmap = plt.cm.get_cmap('jet_r',terms)   # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
        
    lam_pre = np.linspace(1,np.amax(lam))
    prediction = model_plot.predict(lam_pre)
    prediction_raw = model_plot.predict(lam)
    
    ax2.scatter(lam, P,s=70, zorder=50,lw=2.5, facecolors='w', edgecolors='k',alpha=1.0,label=label+' exp. data')
    ax2.plot(lam_pre, prediction, label='model '+label,zorder=5, lw=2,color='k');
    
    color_map(ax2, lam_pre, model_plot, model_weights, Psi_model, cmaplist, terms)
    
    ax2.set_ylabel(r'nominal stress $P$ [kPa]',fontsize='x-large')
    ax2.set_xlabel(r'stretch $\lambda$ [-]',fontsize='x-large')
    
    ax2.set_yticks([0.0, 30.0, 60.00])
    ax2.set_xticks([1.00, 2.0, 4.0, 6, 8 ])
    
    ax2.set_ylim(-0.3, 70)
    ax2.set_xlim(0.5, 8.0)
    
    ax2.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=12)



    
#%% Plot Rubber stress strain


    
def PlotTre(ax2, lam_UT, lam_ET, lam_PS, P_UT, P_ET, P_PS, model_UT, model_PS, model_ET, FittingMode):
    
    lam_UT_pre = np.linspace(1,np.amax(lam_UT))
    lam_ET_pre = np.linspace(1,np.amax(lam_ET))
    lam_PS_pre = np.linspace(1,np.amax(lam_PS))
    
    Pre_UT = model_UT.predict(lam_UT_pre)
    Pre_ET = model_PS.predict(lam_PS_pre)
    Pre_PS = model_ET.predict(lam_ET_pre)
    
    Pre_raw_UT = model_UT.predict(lam_UT)
    Pre_raw_ET = model_PS.predict(lam_PS)
    Pre_raw_PS = model_ET.predict(lam_ET)
    
    ax2.scatter(lam_UT, P_UT,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='k',alpha=0.7,label='Uniaxial tension exp. data')
    ax2.scatter(lam_ET, P_ET,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors=ColorS,alpha=0.7,label='Biaxial tension exp. data')
    ax2.scatter(lam_PS, P_PS,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='b',alpha=0.7,label='Pure shear tension exp. data')
    ax2.set_ylabel(r'nominal stress $P$ [kPa]',fontsize='x-large')
    ax2.set_xlabel(r'stretch $\lambda$ [-]',fontsize='x-large')
    ax2.plot(lam_UT_pre, Pre_UT, label='model UT',zorder=5, lw=2,color=ColorI);
    ax2.plot(lam_ET_pre, Pre_ET, label='model ET',zorder=5, lw=2,color=ColorS);
    ax2.plot(lam_PS_pre, Pre_PS, label='model PS',zorder=5, lw=2,color=ColorR);
    
        
    ax2.set_yticks([0.0, 30.0, 60.00])
    ax2.set_xticks([1.00, 2.0, 4.0, 6, 8 ])
    
    ax2.set_ylim(-0.3, 70)
    ax2.set_xlim(0.5, 8.0)
    ax2.set_title('fitting mode: '+FittingMode, fontsize='x-large')
    
    R2_UT = r2_score_own(P_UT, Pre_raw_UT)
    R2_PS = r2_score_own(P_PS, Pre_raw_PS)
    R2_ET = r2_score_own(P_ET, Pre_raw_ET)
    
    ax2.text(0.85,0.25,r'$R_{ut}^2$: '+f"{R2_UT:.2f}",transform=ax2.transAxes,fontsize=14, horizontalalignment='left',color='k')
    ax2.text(0.85,0.15,r'$R_{et}^2$: '+f"{R2_ET:.2f}",transform=ax2.transAxes,fontsize=14, horizontalalignment='left',color='k')
    ax2.text(0.85,0.05,r'$R_{ps}^2$: '+f"{R2_PS:.2f}",transform=ax2.transAxes,fontsize=14, horizontalalignment='left',color='k')

    plt.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=12)
   
   
   
   
    