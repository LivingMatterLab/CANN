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

def plotTrans(fig_ax1, lam_ut_all, P_ut_all, Stress_predict_UT, Region):
    fig_ax1.scatter(lam_ut_all, P_ut_all,s=70, zorder=5,lw=2.5, facecolors='none', edgecolors='k',alpha=0.7,label=Region+ ' compression/tension data')
    fig_ax1.set_ylabel(r'nominal stress $P$ [kPa]',fontsize='x-large')
    fig_ax1.set_xlabel(r'stretch $\lambda$ [-]',fontsize='x-large')
    fig_ax1.plot(lam_ut_all, Stress_predict_UT, label='model',zorder=5, lw=2,color=ColorI);
    
    R2_trans = r2_score_own(P_ut_all, Stress_predict_UT)
    fig_ax1.text(0.85, 0.05, r'$R^2$: '+f"{R2_trans:.3f}", transform=fig_ax1.transAxes, fontsize=14, horizontalalignment='left', color='k')
    
    # plt.tight_layout()
    fig_ax1.legend(loc='upper left', fancybox=True, framealpha=0.,fontsize=14)
    
    return R2_trans



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

def plotTrans(ax3, lam_ut, P_ut, Stress_predict_trans, Region):
    """Plot transverse stress vs stretch"""
    
    ax3.scatter(lam_ut, P_ut, s=70, zorder=5, lw=2.5, facecolors='none', edgecolors='k', alpha=0.7, label=Region+' transverse data')
    ax3.set_ylabel(r'transverse stress [kPa]', fontsize='x-large')
    ax3.set_xlabel(r'stretch $\lambda$ [-]', fontsize='x-large')
    ax3.plot(lam_ut, Stress_predict_trans, label='model', zorder=5, lw=2, color='blue')
    R2_trans = r2_score_own(P_ut, Stress_predict_trans)
    ax3.text(0.85, 0.05, r'$R^2$: '+f"{R2_trans:.3f}", transform=ax3.transAxes, fontsize=14, horizontalalignment='left', color='k')
    
    ax3.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=14)
    
    return R2_trans

def plotTensionWithContributions(lam_ut, P_ut, Stress_predict_axial, stress_contributions, term_names, Region, lam_curve=None, stress_curve=None, contribs_curve=None):
    """Plot tension data with stacked term contributions"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get tension data (second half)
    midpoint = int(len(lam_ut) / 2)
    lam_tension = lam_ut[midpoint:]
    P_tension = P_ut[midpoint:]
    stress_tension = Stress_predict_axial[midpoint:]

    # If dense curve inputs are provided, use them for plotting the smooth model and contributions
    use_dense = lam_curve is not None and stress_curve is not None and contribs_curve is not None
    if use_dense:
        lam_plot = np.asarray(lam_curve).reshape(-1)
        stress_plot = np.asarray(stress_curve).reshape(-1)
        Craw = np.asarray(contribs_curve)
        # Normalize to shape (num_terms, n_points)
        if Craw.ndim == 3:
            # likely (num_terms, n_points, 1) -> squeeze
            Cmat = np.squeeze(Craw, axis=-1)
        elif Craw.ndim == 2:
            # could be (num_terms, n_points) or (n_points, num_terms)
            if Craw.shape[0] == len(term_names) and Craw.shape[1] == lam_plot.shape[0]:
                Cmat = Craw
            elif Craw.shape[0] == lam_plot.shape[0] and Craw.shape[1] == len(term_names):
                Cmat = Craw.T
            else:
                # fallback: treat rows as terms
                Cmat = Craw
        else:
            # list of vectors
            Cmat = np.vstack([np.array(c).reshape(-1) for c in contribs_curve])
        tension_contributions = [Cmat[i, :] for i in range(min(Cmat.shape[0], len(term_names)))]
    else:
        lam_plot = lam_tension
        stress_plot = stress_tension
        # Get tension contributions
        tension_contributions = [contrib[midpoint:] for contrib in stress_contributions]
    

    
    # Plot experimental data
    ax.scatter(lam_tension, P_tension, s=70, zorder=10, lw=2.5, facecolors='none', 
               edgecolors='k', alpha=0.7, label=Region+' tension data')
    
    # Plot model prediction
    ax.plot(lam_plot, stress_plot, label='Model prediction', zorder=9, lw=3, color='red')
    
    # Create stacked area plot for contributions
    # Define colors using jet_r colormap - dynamically sized to number of terms
    num_terms = len(term_names)
    cmap = plt.cm.get_cmap('jet_r', num_terms)
    colors = [cmap(i) for i in range(num_terms)]
    
    # Stack the contributions
    bottom = np.zeros_like(lam_plot)
    for i, (contrib, name) in enumerate(zip(tension_contributions, term_names)):
        # Ensure contrib is 1-dimensional
        contrib_flat = np.array(contrib).flatten()
        if np.any(np.abs(contrib_flat) > 1e-6):  # Only plot significant contributions
            ax.fill_between(lam_plot, bottom, bottom + contrib_flat, 
                           alpha=0.6, color=colors[i], 
                           label=name, zorder=5)
            bottom += contrib_flat
    
    # Calculate R² for tension
    R2_tension = r2_score_own(P_tension, stress_tension)
    
    # Add R² text
    ax.text(0.05, 0.95, r'$R^2$: '+f"{R2_tension:.3f}", transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(r'Nominal stress $P$ [kPa]', fontsize='x-large')
    ax.set_xlabel(r'Stretch $\lambda$ [-]', fontsize='x-large')
    ax.set_title(f'Tension - {Region}', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, R2_tension

def plotCompressionWithContributions(lam_ut, P_ut, Stress_predict_axial, stress_contributions, term_names, Region, lam_curve=None, stress_curve=None, contribs_curve=None):
    """Plot compression data with stacked term contributions"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get compression data (first half)
    midpoint = int(len(lam_ut) / 2)
    lam_compression = lam_ut[:(midpoint + 1)]
    P_compression = P_ut[:(midpoint + 1)]
    stress_compression = Stress_predict_axial[:(midpoint + 1)]

    # If dense curve inputs are provided, use them for plotting the smooth model and contributions
    use_dense = lam_curve is not None and stress_curve is not None and contribs_curve is not None
    if use_dense:
        lam_plot = np.asarray(lam_curve).reshape(-1)
        stress_plot = np.asarray(stress_curve).reshape(-1)
        Craw = np.asarray(contribs_curve)
        if Craw.ndim == 3:
            Cmat = np.squeeze(Craw, axis=-1)
        elif Craw.ndim == 2:
            if Craw.shape[0] == len(term_names) and Craw.shape[1] == lam_plot.shape[0]:
                Cmat = Craw
            elif Craw.shape[0] == lam_plot.shape[0] and Craw.shape[1] == len(term_names):
                Cmat = Craw.T
            else:
                Cmat = Craw
        else:
            Cmat = np.vstack([np.array(c).reshape(-1) for c in contribs_curve])
        compression_contributions = [Cmat[i, :] for i in range(min(Cmat.shape[0], len(term_names)))]
    else:
        lam_plot = lam_compression
        stress_plot = stress_compression
        # Get compression contributions
        compression_contributions = [contrib[:(midpoint + 1)] for contrib in stress_contributions]
    
    # Plot experimental data (flipped axes: negative stress up, stretch decreasing left to right)
    ax.scatter(lam_compression, P_compression, s=70, zorder=10, lw=2.5, facecolors='none', 
               edgecolors='k', alpha=0.7, label=Region+' compression data')
    
    # Plot model prediction
    ax.plot(lam_plot, stress_plot, label='Model prediction', zorder=9, lw=3, color='red')
    
    # Create stacked area plot for contributions
    # Define colors using jet_r colormap - dynamically sized to number of terms
    num_terms = len(term_names)
    cmap = plt.cm.get_cmap('jet_r', num_terms)
    colors = [cmap(i) for i in range(num_terms)]
    
    # Stack the contributions
    bottom = np.zeros_like(lam_plot)
    for i, (contrib, name) in enumerate(zip(compression_contributions, term_names)):
        # Ensure contrib is 1-dimensional
        contrib_flat = np.array(contrib).flatten()
        if np.any(np.abs(contrib_flat) > 1e-6):  # Only plot significant contributions
            ax.fill_between(lam_plot, bottom, bottom + contrib_flat, 
                           alpha=0.6, color=colors[i], 
                           label=name, zorder=5)
            bottom += contrib_flat
    
    # Flip both axes so stretch decreases from left to right and negative stress is up
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Calculate R² for compression
    R2_compression = r2_score_own(P_compression, stress_compression)
    
    # Add R² text
    ax.text(0.05, 0.95, r'$R^2$: '+f"{R2_compression:.3f}", transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(r'Nominal stress $P$ [kPa]', fontsize='x-large')
    ax.set_xlabel(r'Stretch $\lambda$ [-]', fontsize='x-large')
    ax.set_title(f'Compression - {Region}', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, R2_compression

def plotShearWithContributions(gamma_ss, P_ss, Stress_predict_shear, stress_contributions, term_names, Region, gamma_curve=None, stress_curve=None, contribs_curve=None):
    """Plot shear data with stacked term contributions (positive half only)"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get only positive half of the data (since shear is symmetrical)
    midpoint = len(gamma_ss) // 2
    gamma_positive = gamma_ss[midpoint:]
    P_positive = P_ss[midpoint:]
    stress_positive = Stress_predict_shear[midpoint:]

    # If dense curve inputs are provided, use them for plotting the smooth model and contributions
    use_dense = gamma_curve is not None and stress_curve is not None and contribs_curve is not None
    if use_dense:
        gamma_plot = np.asarray(gamma_curve).reshape(-1)
        stress_plot = np.asarray(stress_curve).reshape(-1)
        Craw = np.asarray(contribs_curve)
        if Craw.ndim == 3:
            Cmat = np.squeeze(Craw, axis=-1)
        elif Craw.ndim == 2:
            if Craw.shape[0] == len(term_names) and Craw.shape[1] == gamma_plot.shape[0]:
                Cmat = Craw
            elif Craw.shape[0] == gamma_plot.shape[0] and Craw.shape[1] == len(term_names):
                Cmat = Craw.T
            else:
                Cmat = Craw
        else:
            Cmat = np.vstack([np.array(c).reshape(-1) for c in contribs_curve])
        positive_contributions = [Cmat[i, :] for i in range(min(Cmat.shape[0], len(term_names)))]
    else:
        gamma_plot = gamma_positive
        stress_plot = stress_positive
        # Get positive contributions
        positive_contributions = [contrib[midpoint:] for contrib in stress_contributions]
    
    # Plot experimental data (positive half only)
    ax.scatter(gamma_positive, P_positive, s=70, zorder=10, lw=2.5, facecolors='none', 
               edgecolors='k', alpha=0.7, label=Region+' shear data')
    
    # Plot model prediction
    ax.plot(gamma_plot, stress_plot, label='Model prediction', zorder=9, lw=3, color='red')
    
    # Create stacked area plot for contributions
    # Define colors using jet_r colormap - dynamically sized to number of terms
    num_terms = len(term_names)
    cmap = plt.cm.get_cmap('jet_r', num_terms)
    colors = [cmap(i) for i in range(num_terms)]
    
    # Stack the contributions
    bottom = np.zeros_like(gamma_plot)
    for i, (contrib, name) in enumerate(zip(positive_contributions, term_names)):
        # Ensure contrib is 1-dimensional
        contrib_flat = np.array(contrib).flatten()
        if np.any(np.abs(contrib_flat) > 1e-6):  # Only plot significant contributions
            ax.fill_between(gamma_plot, bottom, bottom + contrib_flat, 
                           alpha=0.6, color=colors[i], 
                           label=name, zorder=5)
            bottom += contrib_flat
    
    # Calculate R² for shear (using positive half only)
    R2_shear = r2_score_own(P_positive, stress_positive)
    
    # Add R² text
    ax.text(0.05, 0.95, r'$R^2$: '+f"{R2_shear:.3f}", transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(r'Shear stress [kPa]', fontsize='x-large')
    ax.set_xlabel(r'Amount of shear $\gamma$ [-]', fontsize='x-large')
    ax.set_title(f'Shear - {Region}', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, R2_shear

