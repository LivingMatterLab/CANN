#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
ColorI = [1.0, 0.65, 0.0]
ColorS = [0.5, 0.00, 0.0]


#%% Uts
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)


#%% Plotting

def plotLoss(axe, history):   
    axe.plot(history)
    # plt.plot(history.history['val_loss'])
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')


def plotActi(axe, CANN_i1, CANN_i2, R2):   
    axe.plot(CANN_i1, label='I1', color='k', ls='solid', lw=2)
    axe.plot(CANN_i2, label='I2', zorder=5, color='grey', ls='dashed', lw=2)
    # plt.plot(history.history['val_loss'])
    # axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('Actication')
    plt.xlabel('epoch')
    
    # twin object for two different y-axis on the sample plot
    ax2=axe.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(R2, color="blue",lw=2)
    ax2.set_ylabel(r"$R^2$ [-]",color="blue",fontsize=14)
        
    ax2.set_ylim(-0.05,1.05)
    
    
    # plt.legend(['train', 'val'], loc='upper left')




    
def plotTenCom(fig_ax1, lam_ut_all, P_ut_all, Stress_predict_UT, Region):
    fig_ax1.scatter(lam_ut_all, P_ut_all,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='k',alpha=0.7,label=Region+ ' compression/tension data')
    fig_ax1.set_ylabel(r'nominal stress $P$ [kPa]',fontsize='x-large')
    fig_ax1.set_xlabel(r'stretch $\lambda$ [-]',fontsize='x-large')
    fig_ax1.plot(lam_ut_all, Stress_predict_UT, label='model',zorder=5, lw=2,color=ColorI);
    
    R2_c = r2_score_own(P_ut_all[:17], Stress_predict_UT[:17])
    R2_t = r2_score_own(P_ut_all[16:], Stress_predict_UT[16:])
    R2 = r2_score_own(P_ut_all, Stress_predict_UT)
    
    fig_ax1.text(0.85,0.25,r'$R^2$: '+f"{R2:.3f}",transform=fig_ax1.transAxes,fontsize=14, horizontalalignment='left',color='k')
    fig_ax1.text(0.85,0.15,r'$R_{t}^2$: '+f"{R2_t:.3f}",transform=fig_ax1.transAxes,fontsize=14, horizontalalignment='left',color='k')
    fig_ax1.text(0.85,0.05,r'$R_{c}^2$: '+f"{R2_c:.3f}",transform=fig_ax1.transAxes,fontsize=14, horizontalalignment='left',color='k')
    
    
    # fig_ax1.set_yticks([-1.20, -0.60, -0.30, 0.00, 0.30])
    # fig_ax1.set_xticks([0.90, 0.95, 1.00, 1.05, 1.10 ])
    # fig_ax1.set_ylim(-1.2,0.6)
    # fig_ax1.set_xlim(0.895,1.105)
    
    # plt.tight_layout()
    fig_ax1.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=14)
    
    return R2, R2_c, R2_t



def plotShear(ax2, gamma_ss, P_ss, Stress_predict_SS, Region):
    
    ax2.scatter(gamma_ss, P_ss,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='k',alpha=0.7,label=Region+' simple shear data')
    ax2.set_ylabel(r'shear stress [kPa]',fontsize='x-large')
    ax2.set_xlabel(r'amount of shear $gamma$ [-]',fontsize='x-large')
    ax2.plot(gamma_ss,Stress_predict_SS, label='model',zorder=5, lw=2,color=ColorS);
    R2ss = r2_score_own(P_ss,Stress_predict_SS)
    ax2.text(0.85,0.05,r'$R^2$: '+f"{R2ss:.3f}",transform=ax2.transAxes,fontsize=14, horizontalalignment='left',color='k')
    
    # ax2.set_yticks([-0.40, -0.20, 0.0, 0.20, 0.40])
    # ax2.set_xticks([-0.2, -0.1, 0.00, 0.1, 0.20 ])
    # ax2.set_ylim(-0.5,0.5)
    # ax2.set_xlim(-0.25,0.25)
    
    
    plt.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=14)
    
    
    return R2ss

