"""
Last modified August 2023 - Corrected Version

@author: Kevin Linka, Skyler St. Pierre
"""

import matplotlib.pyplot as plt
import numpy as np
import copy

from sklearn.metrics import r2_score

# Set Plotting Parameters
# plt.rcParams["font.family"] = "Source Sans 3"
# plt.rcParams['xtick.labelsize'] = 55
# plt.rcParams['ytick.labelsize'] = 55
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 55
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 14
plt.rcParams['ytick.major.pad'] = 14
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.axisbelow'] = True

# ColorI = [1.0, 0.65, 0.0]


# R2 calculation
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth, Prediction)
    return max(R2, 0.0)


def color_map_zero(ax, stretch, model, model_weights, Psi_model, cmaplist, terms, model_type):
    predictions = np.zeros([stretch.shape[0], terms])
    model_plot = copy.deepcopy(model_weights)  # deep copy model weights

    for i in range(terms):
        if model_type == 'Stretch':
            model_plot = np.zeros_like(model_weights)  # wx1 all set to zero
            model_plot[i] = model_weights[i]  # wx1[i] set to trained value
        else:  # for architectures with multiple layers (VL, invariant)
            model_plot[-1] = np.zeros_like(model_weights[-1])  # wx2 all set to zero
            model_plot[-1][i] = model_weights[-1][i]  # wx2[i] set to trained value

        Psi_model.set_weights(model_plot)
        lower = np.sum(predictions, axis=1)
        upper = lower + model.predict(stretch)[:].flatten()
        predictions[:, i] = model.predict(stretch)[:].flatten()
        ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[i],
                         label=i + 1)
        # if i == 0:  # one or two term models, get the correct color
        #     ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[0],
        #                      label=i + 1)
        # else:
        #     ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[7],
        #                      label=i + 1)
        ax.plot(stretch, upper, lw=0.4, zorder=34, color='k')
        # plt.savefig('color' + str(i) + '.pdf')


def plotLoss(axe, history, epochs, path2saveResults, Region, modelFit_mode):
    axe.plot(history.history['loss'], lw=4)
    axe.set_yscale('log')
    plt.ylabel('log loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.tight_layout()
    plt.savefig(path2saveResults + '/Plot_loss_' + Region + '_' + modelFit_mode + '.pdf')


def plotTen(fig_ax1, lam_ut_all, P_ut_all, Stress_predict_UT, Region, path2saveResults, modelFit_mode):
    fig_ax1.set_xticks([1, 1.05, 1.1])
    fig_ax1.set_xticklabels(['', '', ''])
    fig_ax1.set_xlim(1, 1.1)
    if Region == 'AC':
        fig_ax1.set_yticks([0, 5, 10, 15, 20])
        fig_ax1.set_yticklabels(['', '', '', '', ''])
        fig_ax1.set_ylim(0., 20.999)
    elif Region == 'TF':
        fig_ax1.set_yticks([0, 7, 14, 21, 28])
        fig_ax1.set_yticklabels(['', '', '', '', ''])
        fig_ax1.set_ylim(0, 28.543)
    elif Region == 'RC':
        fig_ax1.set_yticks([0, 7, 21])
        fig_ax1.set_yticklabels(['', '', ''])
        fig_ax1.set_ylim(0., 21.803)
    fig_ax1.scatter(lam_ut_all, P_ut_all, s=200, zorder=31, lw=4, facecolors='w', edgecolors='k', clip_on=False)
    fig_ax1.plot(lam_ut_all, Stress_predict_UT, 'k', zorder=31, lw=6, color='darkorange')
    plt.tight_layout()
    plt.savefig(path2saveResults + '/TenPlot_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')
    return


def plotCom(fig_ax1, lam_ut_all, P_ut_all, Stress_predict_UT, Region, path2saveResults, modelFit_mode):
    fig_ax1.set_xticks([1, 0.95, 0.9])
    fig_ax1.set_xticklabels(['', '', ''])
    fig_ax1.set_xlim(1, 0.9)
    if Region == 'AC':
        fig_ax1.set_yticks([0, -2, -4, -6])
        fig_ax1.set_yticklabels(['', '', '', ''])
        fig_ax1.set_ylim(0, -7.231)
    elif Region == 'TF':
        fig_ax1.set_yticks([0, -15, -30])
        fig_ax1.set_yticklabels(['', '', ''])
        fig_ax1.set_ylim(0, -38.297)
    elif Region == 'RC':
        fig_ax1.set_yticks([0, -5, -10, -15])
        fig_ax1.set_yticklabels(['', '', '', ''])
        fig_ax1.set_ylim(0., -16.446)
    fig_ax1.scatter(lam_ut_all, P_ut_all, s=200, zorder=31, lw=4, facecolors='w', edgecolors='k', clip_on=False)
    fig_ax1.plot(lam_ut_all, Stress_predict_UT, 'k', zorder=31, lw=6, color='darkorange')
    plt.tight_layout()
    plt.savefig(path2saveResults + '/CompPlot_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')
    return


def plotShear(ax2, gamma_ss, P_ss, Stress_predict_SS, Region, path2saveResults, modelFit_mode):
    ax2.set_xticks([0, 0.05, 0.1])
    ax2.set_xticklabels(['', '', ''])
    ax2.set_xlim(0, 0.1)
    if Region == 'AC':
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['', '', ''])
        ax2.set_ylim(0., 2.395)
    elif Region == 'TF':
        ax2.set_yticks([0, 2, 4])
        ax2.set_yticklabels(['', '', ''])
        ax2.set_ylim(0., 5.464)
    elif Region == 'RC':
        ax2.set_yticks([0, .4, .8])
        ax2.set_yticklabels(['', '', ''])
        ax2.set_ylim(0., 0.887)
    ax2.scatter(gamma_ss, P_ss, s=200, zorder=31, lw=4, facecolors='w', edgecolors='k', clip_on=False)
    ax2.plot(gamma_ss, Stress_predict_SS, 'k', zorder=31, lw=6, color='darkorange')
    plt.tight_layout()
    plt.savefig(path2saveResults + '/ShearPlot_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')
    return


def plotMapTen(ax, Psi_model, model_weights, model_UT, terms, lam_ut_all, P_ut_all, Region, path2saveResults, modelFit_mode, model_type):
    # define number of terms for colormap plotting; purpose is to keep term colors the same for 1 or 2-term models
    if model_type == 'VL':
        numTerms = 14
    if model_type == 'Invariant':
        numTerms = 12
    if model_type == 'Stretch':
        numTerms = 20  # change if change range
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([1, 1.05, 1.1])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(1, 1.1)
    if Region == 'AC':
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_yticklabels(['', '', '', '', ''])
        ax.set_ylim(0., 20.999)
    elif Region == 'TF':
        ax.set_yticks([0, 7, 14, 21, 28])
        ax.set_yticklabels(['', '', '', '', ''])
        ax.set_ylim(0, 28.543)
    elif Region == 'RC':
        ax.set_yticks([0, 7, 21])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 21.803)
    color_map_zero(ax, lam_ut_all, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut_all, P_ut_all, s=800, zorder=35, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/TenCmap_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')


def plotMapCom(ax, Psi_model, model_weights, model_UT, terms, lam_ut_all, P_ut_all, Region, path2saveResults, modelFit_mode, model_type):
    # define number of terms for colormap plotting; purpose is to keep term colors the same for 1 or 2-term models
    if model_type == 'VL':
        numTerms = 14
    if model_type == 'Invariant':
        numTerms = 12
    if model_type == 'Stretch':
        numTerms = 20  # change if change range
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([1, 0.95, 0.9])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(1, 0.9)
    if Region == 'AC':
        ax.set_yticks([0, -2, -4, -6])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0, -7.231)
    elif Region == 'TF':
        ax.set_yticks([0, -15, -30])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0, -38.297)
    elif Region == 'RC':
        ax.set_yticks([0, -5, -10, -15])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., -16.446)
    color_map_zero(ax, lam_ut_all, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut_all, P_ut_all, s=800, zorder=35, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/CompCmap_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')


def plotMapShear(ax, Psi_model, model_weights, model_SS, terms, gamma_ss, P_ss, Region, path2saveResults, modelFit_mode, model_type):
    # define number of terms for colormap plotting; purpose is to keep term colors the same for 1 or 2-term models
    if model_type == 'VL':
        numTerms = 14
    if model_type == 'Invariant':
        numTerms = 12
    if model_type == 'Stretch':
        numTerms = 20  # change if change range
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([0, 0.05, 0.1])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(0, 0.1)
    if Region == 'AC':
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 2.395)
    elif Region == 'TF':
        ax.set_yticks([0, 2, 4])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 5.464)
    elif Region == 'RC':
        ax.set_yticks([0, .4, .8])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 0.887)
    color_map_zero(ax, gamma_ss, model_SS, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(gamma_ss, P_ss, s=800, zorder=35, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    # ax.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=14)  # if want to see colormap
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/ShearCmap_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')

