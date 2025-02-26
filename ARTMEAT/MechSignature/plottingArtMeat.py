"""
Last modified April 2024

@author: Kevin Linka, Skyler St. Pierre
"""

import matplotlib.pyplot as plt
import numpy as np
import copy

from sklearn.metrics import r2_score

# Set Plotting Parameters
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


# R2 calculation
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth, Prediction)
    return max(R2, 0.0)


def color_map_zero(ax, stretch, model, model_weights, Psi_model, cmaplist, terms, model_type):
    predictions = np.zeros([stretch.shape[0], terms])
    model_plot = copy.deepcopy(model_weights)  # deep copy model weights

    for i in range(terms):
        model_plot[-1] = np.zeros_like(model_weights[-1])  # wx2 all set to zero
        model_plot[-1][i] = model_weights[-1][i]  # wx2[i] set to trained value

        Psi_model.set_weights(model_plot)
        lower = np.sum(predictions, axis=1)
        upper = lower + model.predict(stretch)[:].flatten()
        predictions[:, i] = model.predict(stretch)[:].flatten()
        ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[i],
                        label=i + 1)
        # if i == 0:  # one or two term models, get the correct color
        #     ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[1],
        #                     label=i + 1)
        # else:
        #     ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[6],
        #                     label=i + 1)
        ax.plot(stretch, upper, lw=0.4, zorder=34, color='k')


def plotLoss(axe, history, epochs, path2saveResults, Region, modelFit_mode):
    axe.plot(history.history['loss'], lw=4)
    axe.set_yscale('log')
    plt.ylabel('log loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.tight_layout()
    plt.savefig(path2saveResults + '/Plot_loss_' + Region + '_' + modelFit_mode + '.pdf')


def plotTen(ax, lam_ut_all, P_ut_all, Stress_predict_UT, Region, path2saveResults, modelFit_mode):
    if Region == 'PB_SAUS':  # 1.15
        ax.set_yticks([0, 2.5, 5, 7.5, 10])
        ax.set_yticklabels(['', '', '', '', ''])
        ax.set_ylim(0., 10.479)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    elif Region == 'TOFURKY':  # 1.1
        ax.set_yticks([0, 5, 10, 15])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0, 15.366)
        ax.set_xticks([1, 1.05, 1.1])
        ax.set_xticklabels(['', '', ''])
        ax.set_xlim(1, 1.1)
    elif Region == 'FIRM_TF':  # 1.15
        ax.set_yticks([0, 1.5, 3])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 3.545)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    elif Region == 'XFIRM_TF':  # 1.1
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 3.167)
        ax.set_xticks([1, 1.05, 1.1])
        ax.set_xticklabels(['', '', ''])
        ax.set_xlim(1, 1.1)
    elif Region == 'PB_HOTDOG':  # 1.2
        ax.set_yticks([0, 3.5, 7])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 7.5)
        ax.set_xticks([1, 1.1, 1.2])
        ax.set_xticklabels(['', '', ''])
        ax.set_xlim(1, 1.2)
    elif Region == 'RL_HOTDOG':  # 1.35
        ax.set_yticks([0, 10, 20])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 20.771)
        ax.set_xticks([1, 1.1, 1.2, 1.3])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.35)
    elif Region == 'SPAM_TK':  # 1.15
        ax.set_yticks([0, 6.5, 13])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 13.836)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    elif Region == 'RL_SAUS':  # 1.15
        ax.set_yticks([0, 5, 10, 15])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 17.584)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    ax.scatter(lam_ut_all, P_ut_all, s=200, zorder=31, lw=4, facecolors='w', edgecolors='k', clip_on=False)
    ax.plot(lam_ut_all, Stress_predict_UT, 'k', zorder=31, lw=6, color='darkorange')
    plt.tight_layout()
    plt.savefig(path2saveResults + '/TenPlot_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')
    return


def plotCom(ax, lam_ut_all, P_ut_all, Stress_predict_UT, Region, path2saveResults, modelFit_mode):
    ax.set_xticks([1, 0.95, 0.9])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(1, 0.9)
    if Region == 'PB_SAUS':
        ax.set_yticks([0, -2.5, -5, -7.5, -10])
        ax.set_yticklabels(['', '', '', '', ''])
        ax.set_ylim(0., -10.433)
    elif Region == 'TOFURKY':
        ax.set_yticks([0, -10, -20])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0, -22.081)
    elif Region == 'FIRM_TF':
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -2.876)
    elif Region == 'XFIRM_TF':
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -2.469)
    elif Region == 'PB_HOTDOG':
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -4.098)
    elif Region == 'RL_HOTDOG':
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -4.159)
    elif Region == 'SPAM_TK':
        ax.set_yticks([0, -3.5, -7])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -7.389)
    elif Region == 'RL_SAUS':
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., -4.745)
    ax.scatter(lam_ut_all, P_ut_all, s=200, zorder=31, lw=4, facecolors='w', edgecolors='k', clip_on=False)
    ax.plot(lam_ut_all, Stress_predict_UT, 'k', zorder=31, lw=6, color='darkorange')
    plt.tight_layout()
    plt.savefig(path2saveResults + '/CompPlot_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')
    return


def plotShear(ax, gamma_ss, P_ss, Stress_predict_SS, Region, path2saveResults, modelFit_mode):
    ax.set_xticks([0, 0.05, 0.1])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(0, 0.1)
    if Region == 'PB_SAUS':
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 3.156)
    elif Region == 'TOFURKY':
        ax.set_yticks([0, 3, 6])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0, 6.966)
    elif Region == 'FIRM_TF':
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 0.693)
    elif Region == 'XFIRM_TF':
        ax.set_yticks([0, 0.35, 0.7])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 0.745)
    elif Region == 'PB_HOTDOG':
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 1.090)
    elif Region == 'RL_HOTDOG':
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 1.027)
    elif Region == 'SPAM_TK':
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 2.152)
    elif Region == 'RL_SAUS':
        ax.set_yticks([0, 0.45, 0.9])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 0.939)
    ax.scatter(gamma_ss, P_ss, s=200, zorder=31, lw=4, facecolors='w', edgecolors='k', clip_on=False)
    ax.plot(gamma_ss, Stress_predict_SS, 'k', zorder=31, lw=6, color='darkorange')
    plt.tight_layout()
    plt.savefig(path2saveResults + '/ShearPlot_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')
    return


def plotMapTen(ax, Psi_model, model_weights, model_UT, terms, lam_ut_all, P_ut_all, Region, path2saveResults, modelFit_mode, model_type):
    # define number of terms for colormap plotting; purpose is to keep term colors the same for 1 or 2-term models
    if model_type == 'Invariant':
        numTerms = 8
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if Region == 'PB_SAUS':  # 1.15
        ax.set_yticks([0, 2.5, 5, 7.5, 10])
        ax.set_yticklabels(['', '', '', '', ''])
        ax.set_ylim(0., 10.479)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    elif Region == 'TOFURKY':  # 1.1
        ax.set_yticks([0, 5, 10, 15])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0, 15.366)
        ax.set_xticks([1, 1.05, 1.1])
        ax.set_xticklabels(['', '', ''])
        ax.set_xlim(1, 1.1)
    elif Region == 'FIRM_TF':  # 1.15
        ax.set_yticks([0, 1.5, 3])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 3.545)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    elif Region == 'XFIRM_TF':  # 1.1
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 3.167)
        ax.set_xticks([1, 1.05, 1.1])
        ax.set_xticklabels(['', '', ''])
        ax.set_xlim(1, 1.1)
    elif Region == 'PB_HOTDOG':  # 1.2
        ax.set_yticks([0, 3.5, 7])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 7.5)
        ax.set_xticks([1, 1.1, 1.2])
        ax.set_xticklabels(['', '', ''])
        ax.set_xlim(1, 1.2)
    elif Region == 'RL_HOTDOG':  # 1.35
        ax.set_yticks([0, 10, 20])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 20.771)
        ax.set_xticks([1, 1.1, 1.2, 1.3])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.35)
    elif Region == 'SPAM_TK':  # 1.15
        ax.set_yticks([0, 6.5, 13])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 13.836)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    elif Region == 'RL_SAUS':  # 1.15
        ax.set_yticks([0, 5, 10, 15])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 17.584)
        ax.set_xticks([1, 1.05, 1.1, 1.15])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_xlim(1, 1.15)
    color_map_zero(ax, lam_ut_all, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut_all, P_ut_all, s=800, zorder=35, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/TenCmap_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')


def plotMapCom(ax, Psi_model, model_weights, model_UT, terms, lam_ut_all, P_ut_all, Region, path2saveResults, modelFit_mode, model_type):
    # define number of terms for colormap plotting; purpose is to keep term colors the same for 1 or 2-term models
    if model_type == 'Invariant':
        numTerms = 8
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([1, 0.95, 0.9])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(1, 0.9)
    if Region == 'PB_SAUS':
        ax.set_yticks([0, -2.5, -5, -7.5, -10])
        ax.set_yticklabels(['', '', '', '', ''])
        ax.set_ylim(0., -10.433)
    elif Region == 'TOFURKY':
        ax.set_yticks([0, -10, -20])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0, -22.081)
    elif Region == 'FIRM_TF':
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -2.876)
    elif Region == 'XFIRM_TF':
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -2.469)
    elif Region == 'PB_HOTDOG':
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -4.098)
    elif Region == 'RL_HOTDOG':
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -4.159)
    elif Region == 'SPAM_TK':
        ax.set_yticks([0, -3.5, -7])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., -7.389)
    elif Region == 'RL_SAUS':
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., -4.745)
    color_map_zero(ax, lam_ut_all, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut_all, P_ut_all, s=800, zorder=35, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/CompCmap_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')


def plotMapShear(ax, Psi_model, model_weights, model_SS, terms, gamma_ss, P_ss, Region, path2saveResults, modelFit_mode, model_type):
    # define number of terms for colormap plotting; purpose is to keep term colors the same for 1 or 2-term models
    if model_type == 'Invariant':
        numTerms = 8
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([0, 0.05, 0.1])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(0, 0.1)
    if Region == 'PB_SAUS':
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 3.156)
    elif Region == 'TOFURKY':
        ax.set_yticks([0, 3, 6])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0, 6.966)
    elif Region == 'FIRM_TF':
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(['', '', '', ''])
        ax.set_ylim(0., 0.693)
    elif Region == 'XFIRM_TF':
        ax.set_yticks([0, 0.35, 0.7])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 0.745)
    elif Region == 'PB_HOTDOG':
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 1.090)
    elif Region == 'RL_HOTDOG':
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 1.027)
    elif Region == 'SPAM_TK':
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 2.152)
    elif Region == 'RL_SAUS':
        ax.set_yticks([0, 0.45, 0.9])
        ax.set_yticklabels(['', '', ''])
        ax.set_ylim(0., 0.939)
    color_map_zero(ax, gamma_ss, model_SS, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(gamma_ss, P_ss, s=800, zorder=35, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    # ax.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=14)  # if want to see colormap
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/ShearCmap_' + 'Train' + modelFit_mode + '_' + 'Region' + Region + '.pdf')

