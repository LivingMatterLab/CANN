#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from math import gamma
from numpy import dtype, ndarray
from numpy._typing._shape import _AnyShape
from typing import Any
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
    plt.title('model loss', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    axe.tick_params(labelsize=16)
    # plt.legend(['train', 'val'], loc='upper left')


def plotActi(axe, CANN_i1, CANN_i2, R2):   
    axe.plot(CANN_i1, label='I1', color='k', ls='solid', lw=2)
    axe.plot(CANN_i2, label='I2', zorder=5, color='grey', ls='dashed', lw=2)
    # plt.plot(history.history['val_loss'])
    # axe.set_yscale('log')
    plt.title('model loss', fontsize=16)
    plt.ylabel('Actication', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    axe.tick_params(labelsize=16)
    
    # twin object for two different y-axis on the sample plot
    ax2=axe.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(R2, color="blue",lw=2)
    ax2.set_ylabel(r"$R^2$ [-]",color="blue",fontsize=16)
    ax2.tick_params(labelsize=16)
        
    ax2.set_ylim(-0.05,1.05)
    
    
    # plt.legend(['train', 'val'], loc='upper left')




    
def plotTenCom(fig_ax1, lam_ut_all, P_ut_all, Stress_predict_UT, Region):
    fig_ax1.scatter(lam_ut_all, P_ut_all,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='k',alpha=0.7,label=Region+ ' compression/tension data')
    fig_ax1.set_ylabel(r'nominal stress $P$ [kPa]',fontsize=16)
    fig_ax1.set_xlabel(r'stretch $\lambda$ [-]',fontsize=16)
    fig_ax1.plot(lam_ut_all, Stress_predict_UT, label='model',zorder=5, lw=2,color=ColorI);
    
    R2_c = r2_score_own(P_ut_all[:17], Stress_predict_UT[:17])
    R2_t = r2_score_own(P_ut_all[16:], Stress_predict_UT[16:])
    R2 = r2_score_own(P_ut_all, Stress_predict_UT)
    
    fig_ax1.text(0.85,0.25,r'$R^2$: '+f"{R2:.3f}",transform=fig_ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    fig_ax1.text(0.85,0.15,r'$R_{t}^2$: '+f"{R2_t:.3f}",transform=fig_ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    fig_ax1.text(0.85,0.05,r'$R_{c}^2$: '+f"{R2_c:.3f}",transform=fig_ax1.transAxes,fontsize=16, horizontalalignment='left',color='k')
    fig_ax1.tick_params(labelsize=16)
    
    
    # fig_ax1.set_yticks([-1.20, -0.60, -0.30, 0.00, 0.30])
    # fig_ax1.set_xticks([0.90, 0.95, 1.00, 1.05, 1.10 ])
    # fig_ax1.set_ylim(-1.2,0.6)
    # fig_ax1.set_xlim(0.895,1.105)
    
    # plt.tight_layout()
    fig_ax1.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=16)
    
    return R2, R2_c, R2_t




def plotShear(ax2, gamma_ss, P_ss, Stress_predict_SS, Region):
    
    ax2.scatter(gamma_ss, P_ss,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='k',alpha=0.7,label=Region+' simple shear data')
    ax2.set_ylabel(r'shear stress [kPa]',fontsize=16)
    ax2.set_xlabel(r'amount of shear $gamma$ [-]',fontsize=16)
    ax2.plot(gamma_ss,Stress_predict_SS, label='model',zorder=5, lw=2,color=ColorS);
    R2ss = r2_score_own(P_ss,Stress_predict_SS)
    ax2.text(0.85,0.05,r'$R^2$: '+f"{R2ss:.3f}",transform=ax2.transAxes,fontsize=16, horizontalalignment='left',color='k')
    ax2.tick_params(labelsize=16)
    
    # ax2.set_yticks([-0.40, -0.20, 0.0, 0.20, 0.40])
    # ax2.set_xticks([-0.2, -0.1, 0.00, 0.1, 0.20 ])
    # ax2.set_ylim(-0.5,0.5)
    # ax2.set_xlim(-0.25,0.25)
    
    
    plt.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=16)
    
    
    return R2ss

def plotTrans(ax3, lam_ut, P_ut, Stress_predict_trans, Region):
    """Plot transverse stress vs stretch"""
    
    n_plot_pts = 16
    lam_plot = np.linspace(np.amin(lam_ut), np.amax(lam_ut), n_plot_pts)
    P_plot = np.interp(lam_plot, lam_ut, P_ut)
    
    ax3.scatter(lam_plot, P_plot, s=70, zorder=5, lw=2.5, facecolors='none', edgecolors='k', alpha=0.7, label=Region+' transverse data')
    ax3.set_ylabel(r'transverse stress [kPa]', fontsize=16)
    ax3.set_xlabel(r'stretch $\lambda$ [-]', fontsize=16)
    ax3.plot(lam_ut, Stress_predict_trans, label='model', zorder=5, lw=2, color='blue')
    # R2_trans = r2_score_own(P_ut, Stress_predict_trans)
    # ax3.text(0.85, 0.05, r'$R^2$: '+f"{R2_trans:.3f}", transform=ax3.transAxes, fontsize=16, horizontalalignment='left', color='k')
    ax3.tick_params(labelsize=16)
    
    ax3.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=16)
    
    # return R2_trans

def plotTensionWithContributions(lam_ut, P_ut, Stress_predict_axial, stress_contributions, term_names, Region):
    """Plot tension data with stacked term contributions"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get tension data (second half)
    midpoint = int(len(lam_ut) / 2)
    lam_tension = lam_ut[midpoint:]
    P_tension = P_ut[midpoint:]
    stress_tension = Stress_predict_axial[midpoint:]

    n_plot_pts = 16
    lam_plot = np.linspace(np.amin(lam_tension), np.amax(lam_tension), n_plot_pts)
    P_plot = np.interp(lam_plot, lam_tension, P_tension)
    # stress_plot = stress_tension
    # Get tension contributions
    tension_contributions = [contrib[midpoint:] for contrib in stress_contributions]
    

    
    # Plot experimental data
    ax.scatter(lam_plot, P_plot, s=70, zorder=10, lw=2.5, facecolors='none', 
               edgecolors='k', alpha=0.7, label=Region+' tension data')
    
    # Plot model prediction
    ax.plot(lam_tension, stress_tension, label='Model prediction', zorder=9, lw=3, color='red')
    
    # Create stacked area plot for contributions
    # Define colors using jet_r colormap - dynamically sized to number of terms
    num_terms = len(term_names)
    cmap = plt.cm.get_cmap('jet_r', num_terms)
    colors = [cmap(i) for i in range(num_terms)]
    
    # Stack the contributions
    bottom = np.zeros_like(lam_tension)
    for i, (contrib, name) in enumerate(zip(tension_contributions, term_names)):
        # Ensure contrib is 1-dimensional
        contrib_flat = np.array(contrib).flatten()
        if np.any(np.abs(contrib_flat) > 1e-6):  # Only plot significant contributions
            ax.fill_between(lam_tension, bottom, bottom + contrib_flat, 
                           alpha=0.6, color=colors[i], 
                           label=name, zorder=5)
            bottom += contrib_flat
    
    # Calculate R² for tension
    R2_tension = r2_score_own(P_tension, stress_tension)
    
    # Add R² text
    ax.text(0.5, 0.95, r'$R^2$: '+f"{R2_tension:.3f}", transform=ax.transAxes, 
            fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(r'Nominal stress $P$ [kPa]', fontsize=16)
    ax.set_xlabel(r'Stretch $\lambda$ [-]', fontsize=16)
    ax.set_title(f'Tension - {Region}', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=16)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    ax.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=16)

    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, R2_tension

def plotCompressionWithContributions(lam_ut, P_ut, Stress_predict_axial, stress_contributions, term_names, Region):
    """Plot compression data with stacked term contributions"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get compression data (first half)
    midpoint = int(len(lam_ut) / 2)
    lam_compression = lam_ut[:(midpoint + 1)]
    P_compression = P_ut[:(midpoint + 1)]
    stress_compression = Stress_predict_axial[:(midpoint + 1)]

    n_plot_pts = 16
    lam_plot = np.linspace(np.amin(lam_compression), np.amax(lam_compression), n_plot_pts)
    P_plot = np.interp(lam_plot, lam_compression, P_compression)
    # Get compression contributions
    compression_contributions = [contrib[:(midpoint + 1)] for contrib in stress_contributions]
    
    # Plot experimental data (flipped axes: negative stress up, stretch decreasing left to right)
    ax.scatter(lam_plot, P_plot, s=70, zorder=10, lw=2.5, facecolors='none', 
               edgecolors='k', alpha=0.7, label=Region+' compression data')
    
    # Plot model prediction
    ax.plot(lam_compression, stress_compression, label='Model prediction', zorder=9, lw=3, color='red')
    
    # Create stacked area plot for contributions
    # Define colors using jet_r colormap - dynamically sized to number of terms
    num_terms = len(term_names)
    cmap = plt.cm.get_cmap('jet_r', num_terms)
    colors = [cmap(i) for i in range(num_terms)]
    
    # Stack the contributions
    bottom = np.zeros_like(lam_compression)
    for i, (contrib, name) in enumerate[tuple[ndarray[_AnyShape, dtype[Any]] | ndarray[_AnyShape, dtype], Any]](zip(compression_contributions, term_names)):
        # Ensure contrib is 1-dimensional
        contrib_flat = np.array(contrib).flatten()
        if np.any(np.abs(contrib_flat) > 1e-6):  # Only plot significant contributions
            ax.fill_between(lam_compression, bottom, bottom + contrib_flat, 
                           alpha=0.6, color=colors[i], 
                           label=name, zorder=5)
            bottom += contrib_flat
    
    # Flip both axes so stretch decreases from left to right and negative stress is up
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Calculate R² for compression
    R2_compression = r2_score_own(P_compression, stress_compression)
    
    # Add R² text
    ax.text(0.5, 0.95, r'$R^2$: '+f"{R2_compression:.3f}", transform=ax.transAxes, 
            fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel(r'Nominal stress $P$ [kPa]', fontsize=16)
    ax.set_xlabel(r'Stretch $\lambda$ [-]', fontsize=16)
    ax.set_title(f'Compression - {Region}', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=16)
    ax.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, R2_compression

def plotShearWithContributions(gamma_ss, P_ss, Stress_predict_shear, stress_contributions, term_names, Region):
    """Plot shear data with stacked term contributions (positive half only)"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get only positive half of the data (since shear is symmetrical)
    midpoint = len(gamma_ss) // 2
    gamma_positive = gamma_ss[midpoint:]
    P_positive = P_ss[midpoint:]
    stress_positive = Stress_predict_shear[midpoint:]

    n_plot_pts = 16
    gamma_plot = np.linspace(np.amin(gamma_positive), np.amax(gamma_positive), n_plot_pts)
    P_plot = np.interp(gamma_plot, gamma_positive, P_positive)
    # Get positive contributions
    positive_contributions = [contrib[midpoint:] for contrib in stress_contributions]
    
    # Plot experimental data (positive half only)
    ax.scatter(gamma_plot, P_plot, s=70, zorder=10, lw=2.5, facecolors='none', 
               edgecolors='k', alpha=0.7, label=Region+' shear data')
    
    # Plot model prediction
    ax.plot(gamma_positive, stress_positive, label='Model prediction', zorder=9, lw=3, color='red')
    
    # Create stacked area plot for contributions
    # Define colors using jet_r colormap - dynamically sized to number of terms
    num_terms = len(term_names)
    cmap = plt.cm.get_cmap('jet_r', num_terms)
    colors = [cmap(i) for i in range(num_terms)]
    
    # Stack the contributions
    bottom = np.zeros_like(gamma_positive)
    for i, (contrib, name) in enumerate(zip(positive_contributions, term_names)):
        # Ensure contrib is 1-dimensional
        contrib_flat = np.array(contrib).flatten()
        if np.any(np.abs(contrib_flat) > 1e-6):  # Only plot significant contributions
            ax.fill_between(gamma_positive, bottom, bottom + contrib_flat, 
                           alpha=0.6, color=colors[i], 
                           label=name, zorder=5)
            bottom += contrib_flat
    
    # Calculate R² for shear (using positive half only)
    R2_shear = r2_score_own(P_positive, stress_positive)
    
    # Add R² text
    ax.text(0.5, 0.95, r'$R^2$: '+f"{R2_shear:.3f}", transform=ax.transAxes, 
            fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(r'Shear stress [kPa]', fontsize=16)
    ax.set_xlabel(r'Amount of shear $\gamma$ [-]', fontsize=16)
    ax.set_title(f'Shear - {Region}', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=16)
    ax.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, R2_shear

