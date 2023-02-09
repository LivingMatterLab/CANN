#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import pickle
import json
import pandas as pd
import os

from sklearn.metrics import r2_score
from ModelsSkin_p import*
ColorI = [1.0, 0.65, 0.0]
ColorS = [0.5, 0.00, 0.0]


#%% Uts

def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)

def flatten(l):
    return [item for sublist in l for item in sublist]

def GetZeroList(model_weights):
    model_zeros = []
    for i in range(len(model_weights)):
        model_zeros.append(np.zeros_like(model_weights[i]))
    return model_zeros



def color_map(ax2, lamx, lamy,lamplot, model_BT, model_weights, Psi_model, cmaplist, terms, label):
    predictions = np.zeros([lamx.shape[0],terms])
    cmap_r = list(reversed(cmaplist))
    for i in range(len(model_weights)-1):
        
        model_plot = GetZeroList(model_weights)
        model_plot[i] =  model_weights[i]
        model_plot[-1][i] = model_weights[-1][i]
        # print(model_plot)
        Psi_model.set_weights(model_plot)
        
        lower = np.sum(predictions,axis=1)
        if label == 'x':
            upper = lower +  model_BT.predict([lamx, lamy])[0][:].flatten()
            predictions[:,i] = model_BT.predict([lamx, lamy])[0][:].flatten()
        else:
            upper = lower +  model_BT.predict([lamx, lamy])[1][:].flatten()
            predictions[:,i] = model_BT.predict([lamx, lamy])[1][:].flatten()
            
        im = ax2.fill_between(lamplot[:], lower.flatten(), upper.flatten(), zorder=i+1, alpha=1.0, color=cmap_r[i])
        


def color_map_Fung(ax2,lamplot, model_BT, model_weights, Psi_model, cmaplist, terms):
    
    predictions = np.zeros([lamplot.shape[0],terms])
    cmap_r = list(reversed(cmaplist))
    
    for i in range(len(model_weights)-1):
        
        model_plot = GetZeroList(model_weights)
        model_plot[i] =  model_weights[i]
        model_plot[-1][i] = model_weights[-1][i]
        # print(model_plot)
        Psi_model.set_weights(model_plot)
        
        lower = np.sum(predictions,axis=1)

        upper = lower +  model_BT.predict(lamplot)[:].flatten()
        predictions[:,i] = model_BT.predict(lamplot)[:].flatten()
            
        im = ax2.fill_between(lamplot[:], lower.flatten(), upper.flatten(), zorder=i+1, alpha=1.0, color=cmap_r[i])
        

#%% Plotting


c_lis = ['b','g','r','k','m']

def plotLoss(axe, history):   
    axe.plot(history)
    # plt.plot(history.history['val_loss'])
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    
def plotLossVal(axe, history, val_history):   
    axe.plot(history)
    axe.plot(val_history)
    # plt.plot(history.history['val_loss'])
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    
    
def PlotCycles(id1, ax1, ax2, ax3, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted):       

    delta = 0
    R2x_all = []
    R2y_all = []
    for k in range(len(all_lam_x)):
        if k == (id1-1):
            lsStyle = 'dashed'
            # print(id1-1)
        else:
            lsStyle = 'solid'
            
        ax1.plot(all_lam_x[k], Stress_predicted[k][0], zorder=5,lw=2.5, ls=lsStyle, color=c_lis[k], alpha=1.0)
        ax1.scatter(all_lam_x[k][::2], all_Sigma_xx[k][::2],s=70, zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.6)
        R2x = r2_score_own(all_Sigma_xx[k], Stress_predicted[k][0])
        ax1.text(0.02,0.83-delta,r'$R^2$: '+f"{R2x:.3f}",transform=ax1.transAxes,fontsize=14, horizontalalignment='left',color=c_lis[k])
        R2x_all.append(R2x)
        
        ax2.plot(all_lam_y[k], Stress_predicted[k][1], zorder=5,lw=2.5, ls=lsStyle, color=c_lis[k], alpha=1.0)
        ax2.scatter(all_lam_y[k][::2], all_Sigma_yy[k][::2],s=70, zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.6)
        R2y = r2_score_own(all_Sigma_yy[k], Stress_predicted[k][1])
        ax2.text(0.02,0.83-delta,r'$R^2$: '+f"{R2y:.3f}",transform=ax2.transAxes,fontsize=14, horizontalalignment='left',color=c_lis[k])
        R2y_all.append(R2y)
        
        delta = delta+0.08
        ax3.plot(all_lam_x[k], all_lam_y[k],zorder=5, lw=2,color=c_lis[k], label='y-data')
        
    R2xall = np.mean(np.array(R2x_all))
    ax1.text(0.02,0.83-0.45,r'$R^2_{all}$: '+f"{R2xall:.3f}",transform=ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    ax1.plot(np.nan, np.nan, zorder=5,lw=2.5, ls='solid', color='k', alpha=1.0,label='model x')
    ax1.scatter(np.nan, np.nan,s=70, zorder=4,lw=1.0, facecolors='none', edgecolors='k', alpha=0.7,label='data x')
    ax1.legend(loc='upper left',ncol=2, fancybox=True, framealpha=0.,fontsize=16)
    ax1.set_ylabel(r'Cauchy stress $\sigma$ [MPa]',fontsize='x-large')
    ax1.set_xlabel(r'stretch $\lambda_x$  [-]',fontsize='x-large')
    ax1.set_ylim(-0.05,0.65)
    ax1.set_xlim(0.99,1.25)
    ax1.set_yticks([0, 0.30, 0.60])
    ax1.set_xticks([1.0, 1.1, 1.20])
        
    R2yall = np.mean(np.array(R2y_all))
    ax2.text(0.02,0.83-0.45,r'$R^2_{all}$: '+f"{R2yall:.3f}",transform=ax2.transAxes,fontsize=16, horizontalalignment='left',color='k')
    ax2.plot(np.nan, np.nan, zorder=5,lw=2.5, ls='solid', color='k', alpha=1.0,label='model y')
    ax2.scatter(np.nan, np.nan,s=70, zorder=4,lw=1.0, facecolors='none', edgecolors='k', alpha=0.7,label='data z')
    ax2.legend(loc='upper left',ncol=2, fancybox=True, framealpha=0.,fontsize=16)
    ax2.set_ylabel(r'Cauchy stress $\sigma$ [MPa]',fontsize='x-large')
    ax2.set_xlabel(r'stretch $\lambda_y$  [-]',fontsize='x-large')
    ax2.set_ylim(-0.05,0.65)
    ax2.set_xlim(0.99,1.25)
    ax2.set_yticks([0, 0.30, 0.60])
    ax2.set_xticks([1.0, 1.1, 1.20])
    
    ax3.set_ylabel(r'stretch $\lambda_y$  [-]',fontsize='x-large')
    ax3.set_xlabel(r'stretch $\lambda_x$  [-]',fontsize='x-large')
    
    return R2x_all, R2y_all
    
    



from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
        
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)



def plotMapTenCom16(fig, fig_ax1, Psi_model, model_weights, model_BT, terms, lamx, lamy, lamplot, P_ut_all, Stress_predict_UT, label, cy):
    
    cmap = plt.cm.get_cmap('jet_r',terms)   # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    
     
    color_map(fig_ax1, lamx, lamy,lamplot, model_BT, model_weights, Psi_model, cmaplist, terms, label)
    fig_ax1.scatter(lamplot[::2], P_ut_all[::2],s=70, zorder=100,lw=1.5, facecolors='none', edgecolors='k',alpha=0.7,label='data '+label)
    
    fig_ax1.plot(lamplot, Stress_predict_UT, color='k', label='model '+label,zorder=25, lw=2);
    
    fig_ax1.text(0.02,0.83-0.15,'load path: '+f"{cy:.0f}",transform=fig_ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    
    fig_ax1.set_ylim(-0.05,0.65)
    fig_ax1.set_xlim(0.99,1.25)
    fig_ax1.set_yticks([0, 0.30, 0.60])
    fig_ax1.set_xticks([1.0, 1.1, 1.20])
    
    # plt.tight_layout()
    fig_ax1.set_ylabel(r'Cauchy stress $\sigma$ [MPa]',fontsize='x-large')
    fig_ax1.set_xlabel(r'stretch $\lambda_'+label+'$ [-]',fontsize='x-large')
    fig_ax1.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=14)
    
    divider = make_axes_locatable(fig_ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # cbar=plt.colorbar(ticks=range(terms))

    # fig.subplots_adjust(bottom=0.5)
    
    # cmap = mpl.cm.viridis
    # bounds = np.arange(1,terms+1)
    # norm = mpl.colors.BoundaryNorm(bounds, terms, extend='both')
    norm = mpl.colors.Normalize(vmin=0, vmax=terms)
    
    tick_arr = list(np.flip(np.arange(terms))+0.5)
    tick_label = [r'$I_1$',r'$\exp(I_1)$',r'$I_1^2$',r'$\exp(I_1^2)$',
                  r'$I_2$',r'$\exp(I_2)$',r'$I_2^2$',r'$\exp(I_2^2)$',
                  r'$I_4$',r'$\exp(I_4)$',r'$I_4^2$',r'$\exp(I_4^2)$',
                  r'$I_5$',r'$\exp(I_5)$',r'$I_5^2$',r'$\exp(I_5^2)$']
                  # r'$I_2$',r'$I_4^{\ast}$']
    

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax,ticks=tick_arr, orientation='vertical',
                 label="", ax=fig_ax1)
    cax.set_yticklabels(tick_label,fontsize=10)
  

#%% additional
    


def PlotSolo_arr(id1, ax1, ax2, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted):       

    delta = 0

    # for k in range(len(all_lam_x)):
    #     if k == (id1):

    k = id1
    ax1.plot(all_lam_x[k], Stress_predicted[k][0], zorder=5,lw=2.5, ls='solid', color=c_lis[k], alpha=1.0)
    ax1.scatter(all_lam_x[k][::2], all_Sigma_xx[k][::2],s=70, zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.6)
    R2x = r2_score_own(all_Sigma_xx[k], Stress_predicted[k][0])
    ax1.text(0.02,0.73-delta,r'$R^2_x$: '+f"{R2x:.3f}   |",transform=ax1.transAxes,fontsize=14, horizontalalignment='left',color=c_lis[k])
    # R2x_all.append(R2x)
    
    ax1.plot(all_lam_x[k], Stress_predicted[k][1], zorder=5,lw=2.5, ls='dashed', color=c_lis[k], alpha=1.0)
    ax1.scatter(all_lam_x[k][::2], all_Sigma_yy[k][::2],s=70,marker='s', zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.6)
    R2y = r2_score_own(all_Sigma_yy[k], Stress_predicted[k][1])
    ax1.text(0.3,0.73-delta,r'$R^2_y$: '+f"{R2y:.3f}",transform=ax1.transAxes,fontsize=14, horizontalalignment='left',color=c_lis[k])
    # R2y_all.append(R2y)
    
    delta = delta+0.08

    # R2xall = np.mean(np.array(R2x_all))
    # ax1.text(0.02,0.83-0.45,r'$R^2_{all}$: '+f"{R2xall:.3f}   |",transform=ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    ax1.plot(np.nan, np.nan, zorder=5,lw=2.5, ls='solid', color=c_lis[k], alpha=1.0,label='model x')
    ax1.scatter(np.nan, np.nan,s=70, zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.7,label='data x')
    
    ax1.plot(np.nan, np.nan, zorder=5,lw=2.5, ls='dashed', color=c_lis[k], alpha=1.0,label='model y')
    ax1.scatter(np.nan, np.nan,s=70, zorder=4,lw=1.0, marker='s', facecolors='none', edgecolors=c_lis[k], alpha=0.7,label='data y')
    
    ax1.legend(loc='upper left',ncol=2, fancybox=True, framealpha=0.,fontsize=16)
    ax1.set_ylabel(r'nominal stress $P$ [MPa]',fontsize='x-large')
    ax1.set_xlabel(r'stretch $\lambda_x$  [-]',fontsize='x-large')
    
    ax1.set_ylim(-0.05,0.65)
    ax1.set_xlim(0.99,1.25)
    ax1.set_yticks([0, 0.30, 0.60])
    ax1.set_xticks([1.0, 1.1, 1.20])
    # R2yall = np.mean(np.array(R2y_all))
    # ax1.text(0.15,0.83-0.45,r'$R^2_{y,all}$: '+f"{R2yall:.3f}",transform=ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')

    
    return R2x, R2y

def PlotSolo_FungBiax(id1, ax1, ax2, ax3, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted):       

    delta = 0

    # for k in range(len(all_lam_x)):
    #     if k == (id1):

    k = id1
    ax1.plot(all_lam_x[k], Stress_predicted[k][0], zorder=5,lw=2.5, ls='solid', color=c_lis[k], alpha=1.0)
    ax1.scatter(all_lam_x[k][::2], all_Sigma_xx[k][::2],s=70, zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.6)
    R2x = r2_score_own(all_Sigma_xx[k], Stress_predicted[k][0])
    ax1.text(0.02,0.73-delta,r'$R^2$: '+f"{R2x:.3f}   |",transform=ax1.transAxes,fontsize=14, horizontalalignment='left',color=c_lis[k])
    # R2x_all.append(R2x)
    
    ax1.plot(all_lam_x[k], Stress_predicted[k][1], zorder=5,lw=2.5, ls='dashed', color=c_lis[k], alpha=1.0)
    ax1.scatter(all_lam_x[k][::2], all_Sigma_yy[k][::2],s=70,marker='s', zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.6)
    R2y = r2_score_own(all_Sigma_yy[k], Stress_predicted[k][1])
    ax1.text(0.15,0.73-delta,r'$R^2_y$: '+f"{R2y:.3f}",transform=ax1.transAxes,fontsize=14, horizontalalignment='left',color=c_lis[k])
    # R2y_all.append(R2y)
    
    delta = delta+0.08
    ax3.plot(all_lam_x[k], all_lam_y[k],zorder=5, lw=2,color=c_lis[k], label='y-data')

    # R2xall = np.mean(np.array(R2x_all))
    # ax1.text(0.02,0.83-0.45,r'$R^2_{all}$: '+f"{R2xall:.3f}   |",transform=ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    ax1.plot(np.nan, np.nan, zorder=5,lw=2.5, ls='solid', color=c_lis[k], alpha=1.0,label='model x')
    ax1.scatter(np.nan, np.nan,s=70, zorder=4,lw=1.0, facecolors='none', edgecolors=c_lis[k], alpha=0.7,label='data x')
    
    ax1.plot(np.nan, np.nan, zorder=5,lw=2.5, ls='dashed', color=c_lis[k], alpha=1.0,label='model y')
    ax1.scatter(np.nan, np.nan,s=70, zorder=4,lw=1.0, marker='s', facecolors='none', edgecolors=c_lis[k], alpha=0.7,label='data y')
    
    ax1.legend(loc='upper left',ncol=2, fancybox=True, framealpha=0.,fontsize=16)
    ax1.set_ylabel(r'Cauchy stress $\sigma$ [kPa]',fontsize='x-large')
    ax1.set_xlabel(r'stretch $\lambda_x$  [-]',fontsize='x-large')
    ax1.set_ylim(-0.05,18.5)
    ax1.set_xlim(0.99,2.05)
    ax1.set_yticks([0, 6.0, 9.0, 12.00])
    ax1.set_xticks([1.0, 1.4, 1.60])
    
    # R2yall = np.mean(np.array(R2y_all))
    # ax1.text(0.15,0.83-0.45,r'$R^2_{y,all}$: '+f"{R2yall:.3f}",transform=ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')

    
    ax3.set_ylabel(r'stretch $\lambda_y$  [-]',fontsize='x-large')
    ax3.set_xlabel(r'stretch $\lambda_x$  [-]',fontsize='x-large')
    
    return R2x, R2y



#%% grid plots

def AllPlots_solo(fig2, path2saveResults_0, spec, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy):
    count = 1
    modelFit_mode_all = ['1', '2', '3', "4", "5"]
    for kk, modelFit_mode in enumerate(modelFit_mode_all):
        
        jj = kk  
    
        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
            
       
        Psi_model, terms = StrainEnergy_i5()
        model_BT = modelArchitecture_I5(Psi_model, True, np.pi)
    
         
        #%%  Model training
        model_given = model_BT    
        # jj = 2
        if modelFit_mode == 'all':
    
            input_train = [np.array(flatten(all_lam_x)), np.array(flatten(all_lam_y))]
            output_train = [np.array(flatten(all_Sigma_xx)), np.array(flatten(all_Sigma_yy))]        
        else:
            input_train = [all_lam_x[jj], all_lam_y[jj]]
            output_train = [all_Sigma_xx[jj], all_Sigma_yy[jj]]
            
        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        path_checkpoint = path2saveResults_check + '/model_checkpoint_.h5' 
    
        model_given.load_weights(Save_weights, by_name=False, skip_mismatch=False)
        
        
        # PI-CANN  get model response
        Stress_predicted = []
        for j in range(len(all_lam_x)):
            Stress_pre = model_BT.predict([all_lam_x[j], all_lam_y[j]])
            Stress_predicted.append(Stress_pre)
         
            
    
        ax1 = fig2.add_subplot(spec[0,kk])
        ax2 = fig2.add_subplot(spec[1,kk])
        ax3 = fig2.add_subplot(spec[2,kk])
        
        
        _, _ = PlotSolo_arr(jj, ax1, 0, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted)
        
        model_weights_0 = Psi_model.get_weights()
        
        plotMapTenCom16(fig2, ax2, Psi_model, model_weights_0, model_BT, terms,
                        all_lam_x[kk], all_lam_y[kk], all_lam_x[kk], all_Sigma_xx[kk], Stress_predicted[kk][0], 'x',kk+1)
        plotMapTenCom16(fig2, ax3, Psi_model, model_weights_0, model_BT,
                        terms, all_lam_x[kk], all_lam_y[kk], all_lam_y[kk], all_Sigma_yy[kk], Stress_predicted[kk][1], 'y',kk+1)
        


def AllPlots_combine(fig2, path2saveResults_0, spec, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy):
    
    count = 1
    modelFit_mode_all = ['1', '2', '3', "4",'5']
    for kk, modelFit_mode in enumerate(modelFit_mode_all):
        
        jj = kk  
        modelFit_mode = 'all'
        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
            
       
        Psi_model, terms = StrainEnergy_i5()
        model_BT = modelArchitecture_I5(Psi_model, True, np.pi)

         
        #%%  Model training
        model_given = model_BT    
        # jj = 2
        if modelFit_mode == 'all':

            input_train = [np.array(flatten(all_lam_x)), np.array(flatten(all_lam_y))]
            output_train = [np.array(flatten(all_Sigma_xx)), np.array(flatten(all_Sigma_yy))]        
        else:
            input_train = [all_lam_x[jj], all_lam_y[jj]]
            output_train = [all_Sigma_xx[jj], all_Sigma_yy[jj]]
            
        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        path_checkpoint = path2saveResults_check + '/model_checkpoint_.h5' 

        model_given.load_weights(Save_weights, by_name=False, skip_mismatch=False)
        
        
        # PI-CANN  get model response
        Stress_predicted = []
        for j in range(len(all_lam_x)):
            Stress_pre = model_BT.predict([all_lam_x[j], all_lam_y[j]])
            Stress_predicted.append(Stress_pre)
         
        ax1 = fig2.add_subplot(spec[0,kk])
        ax2 = fig2.add_subplot(spec[1,kk])
        ax3 = fig2.add_subplot(spec[2,kk])
        
        
        _, _ = PlotSolo_arr(jj, ax1, 0, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted)
        
        model_weights_0 = Psi_model.get_weights()
        
        plotMapTenCom16(fig2, ax2, Psi_model, model_weights_0, model_BT, terms,
                        all_lam_x[kk], all_lam_y[kk], all_lam_x[kk], all_Sigma_xx[kk], Stress_predicted[kk][0], 'x',kk+1)
        plotMapTenCom16(fig2, ax3, Psi_model, model_weights_0, model_BT,
                        terms, all_lam_x[kk], all_lam_y[kk], all_lam_y[kk], all_Sigma_yy[kk], Stress_predicted[kk][1], 'y',kk+1)
        
    fig2.tight_layout()
    
    
#%% Bar Plot
import seaborn as sns

def barPlot(path2saveResults_0, model_type, R2_all, R2_pick, f, axes ):
    colum_name = ['x1','x2','x3','x4','x5','y1','y2','y3','y4','y5']

    modelFit_mode_all_table = ['all', '1', '2', '3', "4", "5"]
    R2_mean = np.expand_dims(np.mean(R2_all,axis=0), axis=0)
    R2_sd = np.expand_dims(np.std(R2_all,axis=0), axis=0)
    R2_all_mean = np.concatenate((R2_all,R2_mean,R2_sd), axis=0)


    R2_df = pd.DataFrame(R2_all_mean, index=colum_name + ['mean', 'SD'], columns= modelFit_mode_all_table)
    R2_df.to_latex(path2saveResults_0+'/R2_table.tex',index=True)
    R2_df.to_csv(path2saveResults_0+'/R2_table.csv',index=True)

    R2p_df = pd.DataFrame(R2_pick, index=['x', 'y'], columns= modelFit_mode_all_table)
    R2p_df.to_latex(path2saveResults_0+'/R2p_table.tex',index=True)
    R2p_df.to_csv(path2saveResults_0+'/R2p_table.csv',index=True)


    sns.set_theme(style="whitegrid")
    colors = [(1.0, 0.0, 0.0), (0.0, 0.5, 1.0)]
    df_R2_sel3 = pd.melt(R2p_df, ignore_index=False)
    
    ax1 = axes
    c = sns.barplot(
        data=df_R2_sel3,
        x="variable", y="value", hue=df_R2_sel3.index,
        errorbar=('sd'), estimator=np.mean,
        palette=colors, alpha=1.0, width=0.7,
        order=modelFit_mode_all_table, ax=ax1, saturation=1
    )
    
    c.set_xticklabels(c.get_xticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    
    c.set_xlabel("")
    c.set_ylabel(f"$R^2$  [-]",fontsize='large')
    ax1.legend(loc='upper right',frameon=True,ncol=3, facecolor='white', framealpha=1,fontsize=14)
    ax1.set_title(model_type, fontsize='large') 
    c.set_ylim(0,1.3)
    c.set_yticks([0, 0.5, 1.0])
    
    f.tight_layout()


