#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:42:09 2022

@author: kevinlinka
"""

# Functions for creating plots and figures
from pdf2image import convert_from_path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

import os
from cont_mech import modelArchitecture
from matplotlib import gridspec
from util_functions import *
from cont_mech import *
import seaborn as sns
import pandas as pd
import matplotlib
from models import *


## Override default matplotlib setting so plots look better
# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.family"] = "Source Sans 3"

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['xtick.minor.size'] = 7
plt.rcParams['ytick.minor.size'] = 7
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.labelsize'] = 40

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 14
plt.rcParams['ytick.major.pad'] = 14
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.titlesize'] = 40
plt.rcParams["axes.titleweight"] = "bold"

plt.rcParams["figure.titlesize"] = 40
plt.rcParams["figure.titleweight"] = "extra bold"
matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.handleheight'] = 1
# matplotlib.rcParams['legend.fontsize'] = 60


ColorI = [1.0, 0.65, 0.0]
ColorS = [0.5, 0.00, 0.0]

def GetZeroList(model_weights):
    """
    :param model_weights: list of (potentially nonzero) np arrays
    :return: list of np zero arrays with same shape as model_weights
    """
    model_zeros = []
    for i in range(len(model_weights)):
        model_zeros.append(np.zeros_like(model_weights[i]))
    return model_zeros

def color_map(ax2, gamma_ss, model, model_weights, Psi_model, cmaplist, terms):
    """
    Create color coded graph of different terms in a tuned model
    :param ax2: matplotlib axis to use for plotting
    :param gamma_ss: Input to model (stretch, shear strain, etc)
    :param model:complete model
    :param model_weights: model weights to use for plotting
    :param Psi_model: Strain energy model
    :param cmaplist: list of length terms that specifies the color to use for each term
    :param terms: number of terms in model
    :return:
    """
    terms = min(terms, len(model_weights))
    predictions = np.zeros([gamma_ss.shape[0], terms])
    all_plots = []
    for i in range(terms): # Iterate through all terms
        # Get list the same size as model weights with only one weight nonzero, and set model weights
        model_plot = GetZeroList(model_weights)
        model_plot[i] = model_weights[i]
        Psi_model.set_weights(model_plot)
        # Lower bound of terms is sum of predictions so far
        lower = np.sum(predictions, axis=1)
        # Upper bound of terms is the lower bound plus the contribution from the current term
        upper = lower + model.predict(gamma_ss)[:].flatten()
        predictions[:, i] = model.predict(gamma_ss)[:].flatten() # update list of contributions from each terms
        # Create plot that fills between the lower and upper bound
        all_plots.append(ax2.fill_between(gamma_ss[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[i],
                         label=i + 1))
        # Create plot that draws a line on the upper bound
        all_plots.append(ax2.plot(gamma_ss, upper, lw=0.4, zorder=24, color='k'))
    # Return list of all the plots so we can update them as the weights change throughout training
    return all_plots

# Helper function for sorting weights from largest to smallest outer weight/gain (note order is inner weight, outer weight, inner weight, outer weight, ...)
def sort_weights(weights):
    # reshape weights as tuples of (inner weight, outer weight)
    weights_reshaped = [[weights[i], weights[i + 1]] for i in range(0, len(weights), 2)]
    def get_gain(e):
        return e[1]
    # sort tuples by second term (outer weight)
    weights_reshaped.sort(key=get_gain, reverse=True)
    return weights_reshaped

# Plot gains and exponents in a box and whisker plot in order to show uniqueness of solution
def plot_box_whisker(path2saveResults, l1_flag, show_zeros=False):
    """
    Create box and whisker plot of model weights over multiple iterations of training the same model with the same data to show the uniqueness of the solution
    :param path2saveResults: Directory to save resulting plot
    :param l1_flag: true if L1 regularization is used, false if L0.5 regularization is used
    :param show_zeros: True if should plot all terms, false if should only plot nonzero terms
    """
    # Format of sorted weights is M x N x 2, where M is # of iterations, N is # of terms, and 2 is for exponent, gain
    plt.rcParams['figure.figsize'] = [16, 10]
    sorted_weights = np.load(f'{path2saveResults}/box_whisker.npy')
    # Get which gains are nonzero for any iteration
    nonzero_weights = sorted_weights[:, :, 1].mean(axis=0) > 0
    if not show_zeros: # if show_zeros is false, then filter out gains that are always zero
        sorted_weights = sorted_weights[:, nonzero_weights, :]
    if not l1_flag: # if not l1, square weights
        sorted_weights[:, :, 1] = sorted_weights[:, :, 1] ** 2
    fig, (ax_exponents, ax_gains) = plt.subplots(2, 1)
    # Create box plot of exponents
    ax_exponents.boxplot(sorted_weights[:, :, 0])
    ax_exponents.set_ylabel("Exponents [-]", fontsize=30)
    ax_exponents.set_xticklabels([])
    # Create box plot of gains
    ax_gains.boxplot(sorted_weights[:, :, 1])
    ax_gains.set_ylabel("Gains [kPa]", fontsize=30)
    ax_gains.set_xticklabels([])
    # Save plot as png
    if len(path2saveResults) > 0:
        plt.savefig(f'{path2saveResults}/box_whisker.png', bbox_inches='tight')
    plt.close('all')

def plot_r2_v_num_terms_all_mode(path2saveResults, cann_name):
    """
    Function to plot the r squared of the fit vs the number of terms in the model for all modes at once
    :param path2saveResults: Path to save
    :param cann_name: name of model (specifies the name of the subdirectory to get results from)
    """
    plt.rcParams['figure.figsize'] = [20, 10]
    fig, ax = plt.subplots(1, 1)

    r2_all_mode = []
    weight_hist_all_mode = []
    modelFit_mode_all = []
    for subdir in os.listdir(path2saveResults):
        path = os.path.join(path2saveResults, subdir)
        if not os.path.isfile(path):
            path = os.path.join(path, cann_name, "0", "training.pickle")
            with open(path, 'rb') as handle:
                input_data = pickle.load(handle)
            full_weight_hist = input_data["weight_hist"]
            r2s_ten = input_data["r2"][0]
            r2s_com = input_data["r2"][1]
            r2s_ss = input_data["r2"][2]
            terms = input_data["terms"]
            r2_all_mode.append([r2s_ten, r2s_com, r2s_ss])
            weight_hist_all_mode.append(full_weight_hist)
            modelFit_mode_all.append(subdir)

    # Extract just the gains
    if len(weight_hist_all_mode[0][0]) > terms:  # If number of weights is > number of terms, then every other term is a gain
        full_gain_hist_all_mode = [[[[x[i] for i in range(1, len(x), 2)] for x in y] for y in z] for z in weight_hist_all_mode]
    else:  # IF number of weights = number of terms, then every weight is a gain
        full_gain_hist_all_mode = weight_hist_all_mode
    # Count number of nonzero gains
    nonzero_term_counts = [[sum([int(gain > 0) for gain in gain_hist[-1]]) for gain_hist in full_gain_hist] for full_gain_hist in full_gain_hist_all_mode]
    # Iterate through all modes
    for i in range(len(weight_hist_all_mode)):
        mode = modelFit_mode_all[i] # get mode as string
        # Get length 3 array of whether each loading configuration is train or test
        # Mark is o for training set and + for test set, label_str is Train for training set and Test for test set
        mark = ["o" if (char in mode) else "+" for char in ["T", "C", "SS"]]
        label_str = [" Train" if (char in mode) else " Test" for char in ["T", "C", "SS"]]
        # Create scatter plots for each loading configuration, colors represent loading config and markers represent train vs test
        ax.scatter(nonzero_term_counts[i], r2_all_mode[i][0], color="red", label="Tension"+label_str[0], marker=mark[0])
        ax.scatter(nonzero_term_counts[i], r2_all_mode[i][1], color="green", label="Compression"+label_str[1], marker=mark[1])
        ax.scatter(nonzero_term_counts[i], r2_all_mode[i][2], color="blue", label="Shear"+label_str[2], marker=mark[2])

    # Add title, axis labels, legend
    fig.suptitle('R Squared vs Number of Nonzero Terms', fontsize=30, weight="bold")
    ax.set_xlabel('\\# of Nonzero Terms')
    ax.set_ylabel('R-Squared')
    ax.legend(fontsize=30)

    # Save resulting figure to png
    if len(path2saveResults) > 0:
        plt.savefig(f'{path2saveResults}/{cann_name}_r2_v_num_terms.png', bbox_inches='tight')
    plt.close()

# Function to plot the r squared of the fit vs the number of terms in the model for a single training mode
def plot_r2_v_num_terms( path2saveResults, log_scale=True):
    """
        Function to plot the r squared of the fit vs the number of terms in the model for a single training mode
        :param path2saveResults: Path to save figures
        :param log_scale: If true, (1-r^2) is plotted versus number of terms and the y axis is log scaled
        """
    plt.rcParams['figure.figsize'] = [20, 10]
    fig, ax = plt.subplots(1, 1)

    with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    full_weight_hist = input_data["weight_hist"]
    r2s_ten = input_data["r2"][0]
    r2s_com = input_data["r2"][1]
    r2s_ss = input_data["r2"][2]
    terms = input_data["terms"]

    # Extract just the gains
    if len(full_weight_hist[0]) > terms:  # If number of weights is > number of terms, then every other term is a gain
        full_gain_hist = [[[x[i] for i in range(1, len(x), 2)] for x in y] for y in full_weight_hist]
    else: # IF number of weights = number of terms, then every weight is a gain
        full_gain_hist = full_weight_hist
    # Count number of nonzero gains
    nonzero_term_counts = [sum([int(gain>0) for gain in gain_hist[-1]]) for gain_hist in full_gain_hist]

    colors = ["red", "green", "blue"]
    labels = ["Tension", "Compression", "Shear"]
    for i in range(3):
        r2s = input_data["r2"][i]
        r2s_shape = np.array(r2s).shape
        if r2s_shape[0] > 0:
            tile_shape = [r2s_shape[i] if i > 0 else 1 for i in range(len(r2s_shape))]
            target_shape = [r2s_shape[i] if i == 0 else 1 for i in range(len(r2s_shape))]
            reshaped_term_counts = np.tile(np.array(nonzero_term_counts).reshape(target_shape), tile_shape)
            if log_scale:
                ax.scatter(reshaped_term_counts.flatten(), 1 - np.array(r2s).flatten(), color=colors[i], label=labels[i])
                ax.set_yscale('log')
            else:
                ax.scatter(reshaped_term_counts.flatten(), np.array(r2s).flatten(), color=colors[i], label=labels[i])


    # Add title, axis labels, legend
    fig.suptitle('R Squared vs Number of Nonzero Terms', fontsize=30, weight="bold")
    ax.set_xlabel('\\# of Nonzero Terms')
    ax.set_ylabel('1 - $R^2$' if log_scale else '$R^2$')
    ax.legend(fontsize=30)

    # Save figure to png
    if len(path2saveResults) > 0:
        plt.savefig(f'{path2saveResults}/r2_v_num_terms.png', bbox_inches='tight')
    plt.close()


def plot_r2_bargraph( paths2saveResults, best_reg=2):
    """
    Function to plot the r squared of the fit of the model for various training modes (trained on just 0-90, trained on just 45-135, trained on everything)
    :param paths2saveResults: Path to save figures
    :param best_reg: index that specifies which regularization penalty to use for computing r squared, make 0 for unregularized model (i.e. the regularization penalty used will be alpha = Lalphas[best_reg])
    """

    plt.rcParams['figure.constrained_layout.use'] = True

    cmap = plt.cm.get_cmap('jet_r', 5)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    labels_ws = ["strip-w", "off-w", "equibiax", "off-s", "strip-s"]
    labels_45 = ["strip-x", "off-x", "equibiax", "off-y", "strip-y"]
    labels_all = [labels_ws, labels_ws, labels_45]
    titles = ["w90", "s90", "x45"]
    headings = ['train all', "train 0/90", 'train +45/-45',  'train all but strip s']
    plt.rcParams['figure.figsize'] = [5 * len(paths2saveResults), 15]
    fig, axes = plt.subplots(3, len(paths2saveResults))


    for idx_path2saveResults in range(len(paths2saveResults)):
        path2saveResults = paths2saveResults[idx_path2saveResults]
        with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
            input_data = pickle.load(handle)

        ### Make bar chart of R^2 for particular
        r2s_best = input_data["r2"][0][best_reg]
        r2s_w = r2s_best[0, :, 0]
        r2s_s = r2s_best[0, :, 1]
        r2s_x = r2s_best[1, :, 0]

        r2s_all = [r2s_w, r2s_s, r2s_x]


        for i in range(len(r2s_all)):
            r2 = r2s_all[i]
            labels = labels_all[i]
            # sns.set_theme(style="whitegrid")

            data = pd.DataFrame.from_dict({"label": labels, "value": r2.tolist()})
            c = sns.barplot(
                data=data,
                x="label", y="value", #hue=df_R2_sel3.index,
                palette=cmaplist, alpha=1.0, width=0.7,
                order=labels, ax=axes[i][idx_path2saveResults], saturation=1
            )

            c.set_xticklabels("", rotation=0, ha='right', rotation_mode='anchor')

            c.set_xlabel("")
            c.set_ylabel("$R^2_{" + f"{titles[i]}"+"}$  [-]" if idx_path2saveResults == 0 else "")

            c.set_ylim(0, 1.3)
            c.set_yticks([0, 1.0])

            if i == 0:
                axes[i][idx_path2saveResults].set_title(headings[idx_path2saveResults])

    legend_handles = [Patch(color=color) for color in cmaplist]
    fig.legend(handles=legend_handles, labels=labels_45, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=len(labels_45), fontsize=40)

    engine = fig.get_layout_engine()
    engine.set(rect=(0.005, 0.035, 0.99, 0.96))


    if len(paths2saveResults) > 0:
        plt.savefig(f'{paths2saveResults[0]}/r2_bestfit.pdf', bbox_inches='tight')
    plt.close()

def plot_arch_comp_graph( paths2saveResults, best_reg=2):
    """
    Function to plot the r squared of the fit of the model for various training modes (trained on just 0-90, trained on just 45-135, trained on everything)
    :param paths2saveResults: Path to save figures
    :param best_reg: index that specifies which regularization penalty to use for computing r squared, make 0 for unregularized model (i.e. the regularization penalty used will be alpha = Lalphas[best_reg])
    """
    cmap = plt.cm.get_cmap('jet_r', 5)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    labels_ws = ["strip-w", "off-w", "equibiax", "off-s", "strip-s"]
    labels_45 = ["strip-x", "off-x", "equibiax", "off-y", "strip-y"]
    labels_all = [labels_ws, labels_ws, labels_45]
    titles = ["w90", "s90", "x45"]
    headings = ["two-fiber arch", 'three-fiber arch']
    plt.rcParams['figure.figsize'] = [15, 15]
    fig, axes = plt.subplots(3, len(paths2saveResults))


    for idx_path2saveResults in range(len(paths2saveResults)):
        path2saveResults = paths2saveResults[idx_path2saveResults]
        with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
            input_data = pickle.load(handle)

        ### Make bar chart of R^2 for particular
        r2s_best = input_data["r2"][0][best_reg]
        r2s_w = r2s_best[0, :, 0]
        r2s_s = r2s_best[0, :, 1]
        r2s_x = r2s_best[1, :, 0]

        r2s_all = [r2s_w, r2s_s, r2s_x]


        for i in range(len(r2s_all)):
            r2 = r2s_all[i]
            labels = labels_all[i]
            # sns.set_theme(style="whitegrid")

            data = pd.DataFrame.from_dict({"label": labels, "value": r2.tolist()})
            c = sns.barplot(
                data=data,
                x="label", y="value", #hue=df_R2_sel3.index,
                palette=cmaplist, alpha=1.0, width=0.7,
                order=labels, ax=axes[i][idx_path2saveResults], saturation=1
            )

            c.set_xticklabels("", rotation=0, ha='right', rotation_mode='anchor')

            c.set_xlabel("")
            c.set_ylabel("$R^2_{" + f"{titles[i]}"+"}$  [-]" if idx_path2saveResults == 0 else "")

            c.set_ylim(0, 1.3)
            c.set_yticks([0, 1.0])

            if i == 0:
                axes[i][idx_path2saveResults].set_title(headings[idx_path2saveResults])

    legend_handles = [Patch(color=color) for color in cmaplist]
    fig.legend(handles=legend_handles, labels=labels_45, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=len(labels_45), fontsize=30)

    plt.tight_layout(pad=1)

    if len(paths2saveResults) > 0:
        plt.savefig(f'{paths2saveResults[0]}/r2_archcomp.pdf', bbox_inches='tight')
    plt.close()




def plot_l0_map(paths2saveResults, num_terms, dfs, Region, is_I4beta=True):
    """
    Function to plot the loss of all 2 term and 1 term models in a grid
    :param paths2saveResults: Path to save figures
    :param num_terms: Number of total terms in the model
    """

    plt.rcParams['text.usetex'] = True

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['text.latex.preamble'] = "\n".join([
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\usepackage{amsmath}',
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ])

    data = np.zeros((num_terms, num_terms))
    weights = [[0 for j in range(num_terms)] for i in range (num_terms)]

    for path2saveResults in paths2saveResults:
        with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
            input_data = pickle.load(handle)
        full_loss_hist = input_data["loss_hist"]
        final_loss = full_loss_hist[-1][-1]

        name = path2saveResults.split("/")[-2]
        terms = [int(x) for x in name.split("_")]


        weights[terms[0]][terms[1]] = input_data["weight_hist"][-1][-1]
        weights[terms[1]][terms[0]] = input_data["weight_hist"][-1][-1]

        data[terms[0], terms[1]] = final_loss
        data[terms[1], terms[0]] = final_loss

    ## Print Best Equation
    best_model_idxs = np.argmin(data)
    term1 = best_model_idxs % num_terms
    term2 = int(best_model_idxs / num_terms)
    weights_best = weights[term1][term2]
    mask = [1 if int(i / 2) in [term1, term2] else 0 for i in range(num_terms * 2)]
    weights_best = [weights_best[i] * mask[i] for i in range(num_terms * 2)]
    disp_equation_weights(weights_best, dfs, Region, "0123456789", path2saveResults)

    maxPlot = 0.12
    # minPlot = 0.04
    minPlot = np.min(data)
    # df_loss_sym =
    # df_loss_sym = df_loss_sym.set_index(Weigths_arr)

    cmap = sns.color_palette("RdBu_r", 40)

    fig, axe = plt.subplots(figsize=(20, 18))
    axe.set_aspect('equal', adjustable='box')

    last_inv = "I_{4\\beta}" if is_I4beta else "I_{4s}"
    labels = [x for In in range(1, 3) for x in
              [f"$(I_{In})$", f"$(I_{In})^2$", f"exp$(I_{In})$", f"exp$(I_{In}^2)$"]]
    labels = labels + [x for dir in ["I_{4w}", last_inv] for x in
                       [f"$({dir})^2$", f"exp$({dir})$", f"exp$({dir}^2)$", ]]

    g1 = sns.heatmap(data, annot=False, cmap=cmap, fmt="d", linewidths=.4, vmin=minPlot, vmax=maxPlot, ax=axe, xticklabels=labels, yticklabels=labels)
    axe.set_title("$\\text{\\textbf{" + ('L0 three-fiber architecture' if is_I4beta else 'L0 two-fiber architecture') + "}}$")
    plt.margins(x=0, y=0)
    axe.tick_params(axis='both', which='major', pad=1)
    axe.xaxis.tick_top()
    g1.set_xticks([i+0.5 for i in range(14)], labels=[i+1 for i in range(14)])
    g1.set_yticks([i+0.5 for i in range(14)], labels=[i+1 for i in range(14)])

    nspaces = 25
    g1.collections[0].colorbar.set_label("min" + ("\\enspace " *nspaces) + "$\\text{\\textbf{error}}$" + ("\\enspace " * nspaces) + "max")
    g1.collections[0].colorbar.set_ticks([])
    # g1.collections[0].colorbar.set_ticklabels(["min", "max"], rotation=90, horizontalalignment="left")

    # for tick in g1.collections[0].colorbar.cax.get_majorticklabels():
    #     tick.set_horizontalalignment("left")
    ## uncomment to have terms labelled with term name
    # g1.set_xticks([i+1.0 for i in range(14)], labels=labels)
    # g1.set_xticklabels(g1.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize=18, weight='normal')

    g1.set_xticklabels(g1.get_xticklabels(), rotation=0, horizontalalignment='center', weight='normal')
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, horizontalalignment='right', weight='normal')

    # Save figure to png
    if len(path2saveResults) > 0:
        plt.savefig(f'{paths2saveResults[0]}/../../l0_map.pdf', bbox_inches='tight')
    plt.close()


#
def plot_num_terms_v_epoch(path2saveResults):
    """
    Function to plot the number of nonzero terms in the model and the loss as a function of epoch number
    :param path2saveResults: Path to save figures
    """
    plt.rcParams['figure.figsize'] = [20, 20]
    fig, (ax_terms, ax_loss) = plt.subplots(2, 1)

    with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    full_weight_hist = input_data["weight_hist"]
    full_loss_hist = input_data["loss_hist"]
    Lalphas = input_data["Lalphas"]
    terms = input_data["terms"]

    # Get total number of epochs per regularization weight
    n_epochs_per_lalpha = [len(x) for x in full_weight_hist]
    epochs = [list(range(sum(n_epochs_per_lalpha[0:i]), sum(n_epochs_per_lalpha[0:(i + 1)]))) for i in range(len(full_weight_hist))]

    # Extract just the gains
    if len(full_weight_hist[0]) > terms:  # If number of weights is > number of terms, then every other term is a gain
        full_gain_hist = [[[x[i] for i in range(1, len(x), 2)] for x in y] for y in full_weight_hist]
    else: # IF number of weights = number of terms, then every weight is a gain
        full_gain_hist = full_weight_hist
    # Count number of nonzero terms at each epoch
    nonzero_term_counts = [[sum([int(gain>0) for gain in gains]) for gains in x] for x in full_gain_hist]
    colors = ["red", "orange", "green", "blue", "purple", "pink", "black"]
    # Iterate through number of regularizations weights
    for i in range(len(full_weight_hist)):
        # Create appropriately colored plot of # of terms vs epoch for each regularization weight.
        # Epoch #s start at 0 for 1st regularization weight then pick up where the last regularization weight left off
        ax_terms.plot(epochs[i], nonzero_term_counts[i] , color=colors[i], label=f"$\\alpha$ = {Lalphas[i]}", lw=5)
    # Add labels + legend
    ax_terms.set_ylabel('\\# of Nonzero Terms')
    ax_terms.set_xlabel('Epoch')
    ax_terms.legend(fontsize=30)

    # Iterate through number of regularizations weights
    for i in range(len(full_loss_hist)):
        # Create color coded plot of loss vs epoch for each regularization weight
        ax_loss.plot(epochs[i][1:],  full_loss_hist[i], color=colors[i], lw=5)
    # Add labels + legend
    ax_loss.set_yscale('log')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_xlabel('Epoch')
    plt.tight_layout()

    # Save figure to png
    if len(path2saveResults) > 0:
        plt.savefig(f'{path2saveResults}/num_terms_v_epoch.png', bbox_inches='tight')
    plt.close()


def plot_gains_v_epoch(path2saveResults):
    plt.rcParams['figure.figsize'] = [20, 20]
    fig, (ax_gains, ax_exps) = plt.subplots(2, 1)

    with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    full_weight_hist = input_data["weight_hist"]
    full_loss_hist = input_data["loss_hist"]
    Lalphas = input_data["Lalphas"]
    terms = input_data["terms"]

    # Get total number of epochs per regularization weight
    n_epochs_per_lalpha = [len(x) for x in full_weight_hist]
    epochs = [list(range(sum(n_epochs_per_lalpha[0:i]), sum(n_epochs_per_lalpha[0:(i + 1)]))) for i in range(len(full_weight_hist))]

    # Extract just the gains
    if len(full_weight_hist[0]) > terms:  # If number of weights is > number of terms, then every other term is a gain
        full_gain_hist = [[[x[i] for i in range(1, len(x), 2)] for x in y] for y in full_weight_hist]
    else: # IF number of weights = number of terms, then every weight is a gain
        full_gain_hist = full_weight_hist

    terms = len(full_gain_hist[0][0])
    # Count number of nonzero terms at each epoch
    cmap = plt.cm.get_cmap('jet_r', terms)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    style = ["-", "--"]
    # Iterate through number of regularizations weights
    for i in range(len(full_weight_hist)):
        for j in range(terms):
            curr_gains = np.array([x[2 * j + 1] for x in full_weight_hist[i]])
            curr_gains = curr_gains - 0.5 * (curr_gains <= 0)
            curr_exps = np.array([x[2 * j] for x in full_weight_hist[i]])
            ax_gains.plot(epochs[i], curr_gains, color=cmaplist[j], linestyle=style[i%2], label=f"$\\alpha$ = {Lalphas[i]}", lw=5)
            ax_exps.plot(epochs[i], curr_exps, color=cmaplist[j], linestyle=style[i%2], label=f"$\\alpha$ = {Lalphas[i]}", lw=5)

    # Add labels + legend
    ax_gains.set_ylabel('Gain')
    ax_gains.set_xlabel('Epoch')
    ax_exps.set_ylabel('Exps')
    ax_exps.set_xlabel('Epoch')
    # ax_terms.legend(fontsize=30)

    plt.tight_layout()

    # Save figure to png
    if len(path2saveResults) > 0:
        plt.savefig(f'{path2saveResults}/gains_v_epoch.png', bbox_inches='tight')
    plt.close()

def plot_training(dfs, Psi_model_type, modelFit_mode, p, path2saveResults, best_reg=2, is_I4beta=True):
    """
    Create various figures that show the final model fit and how it changed throughout the training process
    :param dfs: dataframe used for training
    :param Psi_model_type: Function that is called to instantiate model
    :param modelFitMode: String corresponding to which loading directions are used for training
    :param p: p value used for lp regularization
    :param path2saveResults: Path to save created figures and animations to
    :param best_reg: index that specifies which regularization penalty to use for the best fit graph (i.e. the best fit graph will be plotted using alpha = Lalphas[best_reg])
    """

    # Load data from file
    with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    full_weight_hist = input_data["weight_hist"]
    Lalphas = input_data["Lalphas"]
    Region = input_data["Region"]
    P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss, midpoint = getStressStrain(dfs, Region)

    Psi_model, terms = Psi_model_type(lam_ut_all, gamma_ss, P_ut, P_ss, modelFit_mode, 0, True, p)
    is_noiso = (terms < 14)
    model_UT, model_SS, Psi_model, model = modelArchitecture(Region, Psi_model)


    # Flatten first 2 dims of weight_hist_arr so first dimension is just total # of epochs
    weight_hist_arr = [x for y in full_weight_hist for x in y]
    n_epochs_per_lalpha = [len(x) for x in full_weight_hist] # Total number of epochs per regularization weight
    # Get total # of epochs elapset at start of each regularization weight
    first_epoch_per_lalpha = [sum(n_epochs_per_lalpha[0:i]) for i in range(len(n_epochs_per_lalpha))]

    # Plot Best Fit
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = "\n".join([
    #     r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
    #     r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
    #     r'\usepackage[default]{sourcesanspro}',  # set the normal font here
    #     r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
    #     r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    # ])
    fig, axes = plt.subplots(1, 3)
    (ax_w, ax_s, ax_45) = axes
    inputs = reshape_input_output_mesh(lam_ut)
    outputs = reshape_input_output_mesh(P_ut)
    Psi_model.set_weights(weight_hist_arr[first_epoch_per_lalpha[best_reg + 1] if len(first_epoch_per_lalpha) > best_reg + 1 else len(weight_hist_arr) - 1])
    preds = reshape_input_output_mesh(model_UT.predict(lam_ut_all))
    cmap = plt.cm.get_cmap('jet_r', 5)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    labels_ws = ["strip-w", "off-w", "equibiax", "off-s", "strip-s"]
    labels_45 = ["strip-x", "off-x", "equibiax", "off-y", "strip-y"]

    num_points = 17
    for i in range(5):
        ax_w.scatter(resample(inputs[0][i][0], num_points), resample(outputs[0][i][0], num_points), s=300, zorder=25, lw=4, facecolors='w', edgecolors=cmaplist[i], clip_on=False)
        ax_s.scatter(resample(inputs[0][i][1], num_points), resample(outputs[0][i][1], num_points), s=300, zorder=25, lw=4, facecolors='w', edgecolors=cmaplist[i], clip_on=False)
        ax_45.scatter(resample(inputs[1][i][0], num_points), resample(outputs[1][i][0], num_points), s=300, zorder=25, lw=4, facecolors='w', edgecolors=cmaplist[i], clip_on=False)
        ax_w.plot(inputs[0][i][0], preds[0][i][0], color=cmaplist[i], label=labels_ws[i], zorder=25, lw=6)
        ax_s.plot(inputs[0][i][1], preds[0][i][1], color=cmaplist[i], label=labels_ws[i], zorder=25, lw=6,)
        ax_45.plot(inputs[1][i][0], preds[1][i][0], color=cmaplist[i], label=labels_45[i], zorder=25, lw=6,)

    ax_w.set_xlabel("warp stretch [-]")
    ax_w.set_ylabel("warp stress [kPa]")
    ax_s.set_xlabel("shute stretch[-]")
    ax_s.set_ylabel("shute stress [kPa]")
    ax_45.set_xlabel("x stretch [-]")
    ax_45.set_ylabel("x stress [kPa]")

    ax_w.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=30)
    ax_s.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=30)
    ax_45.legend(loc='upper left', fancybox=True, framealpha=0., fontsize=30)

    plt.tight_layout(pad=1)
    plt.savefig(f"{path2saveResults}/best_fit.pdf",
                transparent=False,
                facecolor='white')

    # Create gif of model training across epochs
    # Create figure with multiple subplots to show state of model at a given epoch
    plt.rcParams['figure.figsize'] = [30, 18 if is_noiso else 20]
    plt.rcParams['figure.constrained_layout.use'] = True

    if Region.startswith('mesh'):
        fig, axes = plt.subplots(3, 5)
        axes = flatten([list(x) for x in zip(*axes[0:2])]) + list(axes[2])
        direction_strings = ["w", "s"] * 5 + ["x", "y", "x", "y", "x"]
        n_spaces = 55
        titles = ["strip-w", None, "off-w", None, "equibiax-ws", None, "off-s", None, "strip-s", None, " " * n_spaces + "strip-x", None," " * n_spaces + "off-x", None, "equibiax-xy"]
        inputs = reshape_input_output_mesh(lam_ut)
        inputs = [[x[i] if np.max(x[i]) > 1.0 else ((x[1 - i] - 1) * 1e-9 + 1) for i in range(2)] for y in inputs for x in y] # 10 x 2
        inputs = flatten(inputs)[0:15] # Length 15 list
        outputs = flatten(flatten(reshape_input_output_mesh(P_ut)))[0:15]

        all_plots = [
            plotMap(axes[i], Psi_model, weight_hist_arr[0], DummyModel(), terms, inputs[i], outputs[i], direction_strings[i], titles[i])
            for i in range(len(axes))]  # Create dummy plots, will update in loop

        h, l = axes[0].get_legend_handles_labels()


        cmap = plt.cm.get_cmap('jet_r', 14)  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        last_inv = "I_{4s_{I, II}}" if is_I4beta else "I_{4s}"
        labels = [x for In in range(1, 3) for x in [f"$(I_{In} - 3)$", f"exp$( (I_{In} - 3))$", f"$(I_{In} - 3)^2$", f"exp$( (I_{In} - 3)^2)$"]]
        labels = labels + [x for dir in ["I_{4w}", last_inv] for x in [f"exp$({dir}) -  {dir}$", f"$({dir} - 1)^2$", f"exp$( ({dir} - 1)^2)$", ]]
        if len(labels) > terms: # Handle anisotropic only case
            labels = labels[(len(labels) - terms):]
            cmaplist = cmaplist[(len(cmaplist) - terms):]

        legend_handles = [Patch(color=c) for c in cmaplist] + [all_plots[0][-1]]
        labels += ["data"]
        leg = fig.legend(loc="lower center", ncols=4, handles=legend_handles, labels=labels,
                         mode="expand", fontsize=40)
        leg.get_frame().set_alpha(0)


    else:
        fig, axes = plt.subplots(1, 3)
        models = [model_SS, model_UT, model_UT]
        inputs = [gamma_ss, lam_ut_all[:(midpoint + 1)], lam_ut_all[midpoint:]]
        outputs = [P_ss, P_ut[:(midpoint + 1)], P_ut[midpoint:]]
        maxima = [np.max(output) for output in outputs]  # used for scaling plots
        all_plots = [
            plotMap(axes[i], Psi_model, weight_hist_arr[0], models[i], terms, inputs[i], outputs[i], "", "") for i
            in range(len(models))]

    # Size of both of these arrays is M x N x P where M is the number of time steps we are plotting at, N is the number
    # of terms in the model, and P is the number of plots
    all_paths = [] # List of all paths that define the shaded regions in the plots
    all_uppers = [] # List of all curves that define the boundaries between shaded regions

    # # Adjust axes for psi plots
    # for i in range(3, 6):
    #     axes[i].set_ylim(0, maxima[i])
    # fig.suptitle(f'$\\alpha$ = {Lalphas[0]} , epoch = {0}',
    #              fontsize=30, weight="bold")
    engine = fig.get_layout_engine()
    leg_height = 0.13 if is_noiso else 0.20
    engine.set(rect=(0.005, leg_height , 0.99, 0.995 - leg_height), wspace=0.04)
    # fig.tight_layout(pad=50)

    # plt.pause(1)  # renders all plots
    n_epochs = len(weight_hist_arr)
    n_frames = 0  # Typical size of gif
    if n_frames > 0:
        steps = [int(i * (n_epochs - 1) / n_frames) for i in range(n_frames + 1)] # list of epochs when we will plot
    else:
        steps = [x - 1 for x in first_epoch_per_lalpha]  + [n_epochs - 1]
        steps[0] = 0
    # Iterate through epochs and precompute the appropriate paths to draw


    for i in steps:
        model_weights = weight_hist_arr[i]
        lalpha_idx = sum([i >= x for x in first_epoch_per_lalpha]) - 1
        # Create length 6 list of M x N numpy arrays where M is the number of points in the
        predictions = [np.zeros([output.shape[0], terms]) for output in outputs]
        curr_path = []
        curr_upper = []
        # Compute contribution from one term at a time
        for term in range(terms):
            model_plot = GetZeroList(model_weights)
            if len(model_weights) > terms: # If number of weights is > number of terms, then it is alternate CANN
                # Make 2 weights (gain and exponent) nonzero at a time
                model_plot[2 * term] = model_weights[2 * term]
                model_plot[2 * term + 1] = model_weights[2 * term + 1]
            else:
                # Otherwise, make one weight nonzero at a time
                model_plot[term] = model_weights[term]

            Psi_model.set_weights(model_plot)

            # Add up all the terms BEFORE the current term to get the lower bound for the shaded region
            lowers = [np.sum(prediction, axis=1) for prediction in predictions]
            # Compute contribution from current term
            pred_shear = model_SS.predict(gamma_ss) if gamma_ss != [] else []
            pred_ut = model_UT.predict(lam_ut, verbose=0)
            for plot_id in range(len(predictions)):
                if Region.startswith('mesh'):
                    predictions[plot_id][:, term] = flatten(flatten(reshape_input_output_mesh(pred_ut)))[plot_id].flatten()
                else:
                    predictions[plot_id][:, term] = models[plot_id].predict(inputs[plot_id])[:].flatten()

            # Add up all the terms INCLUDING the current term to get the upper bound for the shaded region
            uppers = [np.sum(prediction, axis=1) for prediction in predictions]

            # Create a path from the upper and lower bounds and add to array
            paths = [create_verts(inputs[k], lowers[k].flatten(), uppers[k].flatten()) for k in range(len(inputs))]
            curr_path.append(paths)
            curr_upper.append(uppers)

        # Add array that has all the terms at the current timestep to another array
        all_paths.append(curr_path)
        all_uppers.append(curr_upper)

    # Set up limits properly
    for k in range(len(axes)):
        min_P = np.min(outputs[k])
        max_P = np.max(outputs[k])
        min_x = np.min(inputs[k])
        max_x = np.max(inputs[k])
        if np.max(inputs[k]) - np.min(inputs[k]) < 1e-6:
            axes[k].set_xticks([np.min(inputs[k]), np.max(inputs[k])])
            axes[k].set_xticklabels(['1', '1'])
        if abs(min_P) < abs(max_P):
            # Tension / Shear
            axes[k].set_xlim([min_x, max_x])
            axes[k].set_ylim([0.0, max_P])
        else:
            # Compression
            axes[k].set_xlim([max_x, min_x])
            axes[k].set_ylim([0.0, min_P])


    # Once paths have been precomputed, begin animation
    for i in range(len(all_uppers)):
        # At each time step update all the plots
        for term in range(terms):
            for plot_id in range(len(all_plots)):
                all_plots[plot_id][2 * term].set_paths(all_paths[i][term][plot_id])
                all_plots[plot_id][2 * term + 1][0].set_ydata(all_uppers[i][term][plot_id])

        # Get current epoch number
        epoch_number = steps[i]
        lalpha_idx = sum([epoch_number >= x for x in first_epoch_per_lalpha]) - 1
        epoch_number = epoch_number - first_epoch_per_lalpha[lalpha_idx]
        # Put epoch # and whether there is regularization in compression plot title (so it is in the top middle)
        # fig.suptitle(f'$\\alpha$ = {Lalphas[lalpha_idx]} , epoch = {epoch_number}',
        #           fontsize=30, weight="bold")

        # for j in range(5):

        for plot_id in range(len(axes)):
            if (steps[i] + 1) in first_epoch_per_lalpha:
                curr_reg = first_epoch_per_lalpha.index(steps[i] + 1) - 1
                r2 = input_data["r2"][0][curr_reg].flatten()[plot_id]
                axes[plot_id].get_shared_y_axes().get_siblings(axes[plot_id])[0].set_xlabel(f"$R^2$ = {r2:.4f}", labelpad=-50)
            elif steps[i] == n_epochs - 1:
                r2 = input_data["r2"][0][-1].flatten()[plot_id]
                axes[plot_id].get_shared_y_axes().get_siblings(axes[plot_id])[0].set_xlabel(f"$R^2$ = {r2:.4f}",
                                                                                            labelpad=-50)
            else:
                axes[plot_id].get_shared_y_axes().get_siblings(axes[plot_id])[0].set_xlabel("", labelpad=-50)

        rect_height = 0.297 if is_noiso else 0.275
        for j in range(5):
            rec = plt.Rectangle((0.2 * j, leg_height + rect_height), 0.2, 1 - leg_height - rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        for j in range(2):
            rec = plt.Rectangle((0.4 * j, leg_height), 0.4, rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        rec = plt.Rectangle((0.8, leg_height), 0.2, rect_height, fill=False, lw=2)
        rec.set_zorder(1000)
        rec = fig.add_artist(rec)
        # for term in range(terms):
        #     leg.get_texts()[term].set_text("%.2f" % weight_hist_arr[steps[i]][2 * term + 1])
        # fig.tight_layout(pad=0)
        # plt.subplots_adjust(top=0.95, bottom=0.20)

        # Render and save plot
        plt.savefig(f"{path2saveResults}/img_{i}.pdf",
                    transparent=False,
                    facecolor='white')
        plt.savefig(f"{path2saveResults}/img_{i}.png",
                    transparent=False,
                    facecolor='white')

    plt.close()

    # Create image from snapshots in training process
    first_step_per_lalpha = [[step >= first_epoch for step in steps].index(True) - 1 for first_epoch in first_epoch_per_lalpha]
    first_step_per_lalpha[0] = 0 # First image should be at epoch 0
    first_step_per_lalpha.append(len(steps) - 1)  # Last epoch should be final result
    extra_pad = 5
    training_img = None
    for i in range(len(first_step_per_lalpha)):
        # Load image and add to frame list
        path = f"{path2saveResults}/img_{first_step_per_lalpha[i]}.pdf"
        new_path = f"{path2saveResults}/training_{i}.pdf"
        shutil.copyfile(path, new_path)

        im = convert_from_path(path, poppler_path='/opt/homebrew/Cellar/poppler/24.04.0/bin')[0]
        if i == 0:
            training_img = Image.new('RGB', (im.width * 2 + extra_pad, im.height * 4 + 3 * extra_pad))
        training_img.paste(im, ((i % 2) * (im.width + extra_pad), int(i / 2) * (im.height + extra_pad)))

    training_img.save(f"{path2saveResults}/training.pdf")

    # Repeat but formatted for paper
    if len(first_epoch_per_lalpha) > 1:
        plot_indices = [0, 5, 4, 9]
        num_regs = 2
        offset = 300
        for i in range(num_regs):
            # Load image and add to frame list
            path = f"{path2saveResults}/img_{first_step_per_lalpha[i + 1]}.pdf"
            im = convert_from_path(path, poppler_path='/opt/homebrew/Cellar/poppler/24.04.0/bin')[0]
            if i == 0:
                W = im.width
                H = im.height
                w = int(W / 5)
                h = int(H * 0.95 / 3)
                training_img = Image.new('RGB', (w * num_regs + (num_regs - 1) * extra_pad, h * len(plot_indices) + (len(plot_indices) - 1) * extra_pad))
            for j in range(len(plot_indices)):
                row = int(plot_indices[j] / 5)
                col = plot_indices[j] % 5
                im_crop = im.crop((col * w, H - h * (3 - row) - offset, (col + 1) * w, H - h * (2 - row) - offset))
                training_img.paste(im_crop, (i * (w + extra_pad), j * (h + extra_pad)))

        training_img.save(f"{path2saveResults}/regularization.pdf")

    # Create array of all frames in GIF
    frames = []
    for i in range(len(steps)):
        # Load image and add to frame list
        path = f"{path2saveResults}/img_{i}.png"
        path2 = f"{path2saveResults}/img_{i}.pdf"
        image = imageio.v2.imread(path)
        frames.append(image)
        # Delete image to avoid clutter in results folder
        os.remove(path)
        os.remove(path2)

    # Create GIF from array of frames
    imageio.mimsave(f"{path2saveResults}/training.gif",  # output path
                    frames,  # array of input frames
                    format="GIF",
                    duration=100)  # optional: duration of frames in ms




def plotMap(ax, Psi_model, model_weights, model, terms, input, output, direction_string, title, num_points=17):
    """
    Create color coded graph of different terms in a tuned model
    :param ax: matplotlib axis to use for plotting
    :param Psi_model: Strain energy model
    :param model_weights: model weights to use for plotting
    :param model: complete model
    :param terms: number of terms in model
    :param input: Input to model (stretch, shear strain, etc)
    :param output: measured value of stress that model is trying to predict
    :param direction_string: string to use in axis labels (i.e. 'x' for 'x stress')
    :param num_points: Maximum number of raw data points that should be plotted (if length of input exceeds num_points then we subsample)
    :return: list of plots generated (so they can be modified later)
    """
    cmap = plt.cm.get_cmap('jet_r', 14)  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if terms < len(cmaplist):
        cmaplist = cmaplist[len(cmaplist)-terms:]
    all_plots = color_map(ax, input, model, model_weights, Psi_model, cmaplist, terms)
    if input.shape[0] > num_points:
        input_old = input
        input = np.linspace(np.min(input), np.max(input), num_points)
        output = np.interp(input, input_old, output)

    scatter_handle = ax.scatter(input, output, s=300, zorder=25, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    ax.set_xlabel(direction_string + " stretch [-]", labelpad=-40)
    ax.set_ylabel(direction_string + " stress [kPa]", labelpad=-40)
    ax.minorticks_on()

    xt = [np.min(input), np.max(input)]
    ax.set_xticks(xt)
    yt = [np.min(output), np.max(output)]
    ax.set_yticks(yt)

    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))

    # secax = ax.secondary_xaxis('top')
    secax = ax.twiny()
    secax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    labelbottom=False, labeltop=False)
    secax.set_xlabel("Temp", labelpad=-50)

    if  title is not None:
        ax.set_title(title, y=1.05, usetex=False)
    # plt.tight_layout(pad=1)
    return all_plots + [scatter_handle]

# Create set of vertices that outline a given term given the upper and lower boundaries of the term
def create_verts(xs, upper, lower):
    N = xs.shape[0]
    verts_out = np.zeros((2 * N, 2))
    verts_out[0:N, 0] = xs
    verts_out[N:, 0] = np.flipud(xs)
    verts_out[0:N, 1] = lower
    verts_out[N:, 1] = np.flipud(upper)
    return [verts_out]





def plot_raw_data():

    file_names = ['../input/all_sigmas_plotting_0_90.xlsx'] * 4 +  ['../input/all_sigmas_plotting_45_135.xlsx'] * 2

    sheetnames = ["sigma_x_load", "sigma_x_unload", "sigma_y_load", "sigma_y_unload", "sigma_x_load", "sigma_x_unload"]
    ylabels = ["w stress [kPa]", "s stress [kPa]", "x stress [kPa]"]
    xlabels = ["w stretch [-]", "s stretch [-]", "x stretch [-]"]
    n_spaces = 55
    titles = [["strip-w", "off-w", "equibiax-ws", "off-s", "strip-s"], [""] * 5,
             [" " * n_spaces + "strip-x", "", " " * n_spaces + "off-x", "", "equibiax-xy"]]
    loads = ["loading", "unloading"]
    # Plot Best Fit
    plt.rcParams['figure.figsize'] = [30, 16]
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = "\n".join([
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\usepackage{amsmath}',
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ])


    for load_idx in range(2):

        plt.rcParams['figure.constrained_layout.use'] = True

        # Load Data
        stress_data_all = [pd.read_excel(file_names[i], sheet_name=sheetnames[i], engine='openpyxl').to_numpy() for i in range(load_idx, 6, 2)] ## 100 rows x 10 columns
        max_strains = [1.1, 1.1, 1.1, 1.05, 1.000001]
        max_strains_all = [max_strains, list(reversed(max_strains)), max_strains]

        # Plot Figure
        fig, axes = plt.subplots(3, 5)
        for j in range(3):
            for k in range(5):
                strain_data = np.linspace(1, max_strains_all[j][k], stress_data_all[0].shape[0])
                stress_idx = k if j < 2 else [0, 4, 1, 3, 2][k]
                stress_mean = stress_data_all[j][:, 2 * stress_idx]
                stress_std = stress_data_all[j][:, 2 * stress_idx + 1]
                axes[j][k].fill_between(strain_data, stress_mean-stress_std, stress_mean + stress_std)
                axes[j][k].plot(strain_data, stress_mean, "k-")
                axes[j][k].set_xlabel(xlabels[j], labelpad=-40)
                axes[j][k].set_ylabel(ylabels[j], labelpad=-40)
                if j == 2 and k % 2 == 1:
                    axes[j][k].set_xlabel("y stretch [-]", labelpad=-40)
                    axes[j][k].set_ylabel("y stress [kPa]", labelpad=-40)
                if max_strains_all[j][k] < 1.01:
                    axes[j][k].set_xticks([np.min(strain_data), np.max(strain_data)])
                    axes[j][k].set_xticklabels(['1.00', '1.00'])
                axes[j][k].set_title(titles[j][k], y=1.05)
                axes[j][k].minorticks_on()

                xt = [np.min(strain_data), np.max(strain_data)]
                axes[j][k].set_xticks(xt)
                axes[j][k].set_xlim(xt)

                yt = [0, np.max(stress_mean + stress_std)]
                axes[j][k].set_yticks(yt)
                axes[j][k].set_ylim(yt)


                axes[j][k].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
                axes[j][k].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))

        # plt.suptitle(suptitles[load_idx])


        for j in range(5):
            rec = plt.Rectangle((0.2 * j, 0.345), 0.2, 0.655, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        for j in range(2):
            rec = plt.Rectangle((0.4 * j, 0), 0.4, 0.345, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        rec = plt.Rectangle((0.8, 0), 0.2, 0.345, fill=False, lw=2)
        rec.set_zorder(1000)
        rec = fig.add_artist(rec)


        engine = fig.get_layout_engine()
        engine.set(rect=(0.005, 0.005, 0.99, 0.99))
        plt.savefig(f"../Results/figures/data_{loads[load_idx]}.pdf",
                    transparent=False,
                    facecolor='white')


