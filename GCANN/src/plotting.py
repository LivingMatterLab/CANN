import matplotlib.pyplot as plt

from gcann import NLL
from src.CANN.models import *
import matplotlib
import csv
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

## Override default matplotlib setting so plots look better
plt.rcParams["font.family"] = "Source Sans 3"
plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35
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

## Create plot for gcann predictions vs data
# stretches and stresses are the experimental data
# model is the trained gcann
# lam_ut_all is a reformatted version of stretches that can be passed into the trained model directly
# terms is true if we want to plot the contribution of each term to the model output, false if we just want to plot the predicted stress distribution
# model_id is the name of the model, which is used to determine where the plots are saved
# modelFit_mode specifies which data was used for training the model, which is used to determine where the plots are saved
# plot_dist is true if we want to plot the distribution of the experimental data, and false if we want to plot the raw experimental data
# blank is true if we want to only plot the experimental data and ignore the model prediction
def plot_gcann(stretches, stresses, model, lam_ut_all, terms, model_id, modelFit_mode, n_samples, plot_dist=True, blank=False, base_path="../Results"):
    ### Compute train NLL
    lam_ut_all = [[stretches.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model, lam_ut_all, P_ut_all,
                                                                       model, [], [], model, 0)
    output_temp = tf.keras.backend.stack(flatten(output_train) + flatten(sample_weights), axis=1)  #
    output_temp = tf.cast(output_temp, tf.float32)
    train_nll2 = 5.0/12 * NLL(output_temp, model(input_train))

    #### Get model Predictions ####
    model_weights = model.get_weights()
    if blank: # if we want a blank plot then we just set the model weights to 0
        model.set_weights([x * 0.0 for x in model_weights])
    Stress_predict_UT = model.predict(lam_ut_all).reshape((-1, 2, 2, 2))  # N x 2 x 2 x 2 (45, xy, mean/var)
    stress_pred_mean = np.array(Stress_predict_UT[:, :, :, 0]).transpose((1, 0, 2)).reshape((2, n_samples, 5, -1, 2))[:, 0, :, :,
                      :]  # 2 x 5 x 100 x 2( x/y)
    stress_pred_std = np.sqrt(
        np.array(Stress_predict_UT[:, :, :, 1]).transpose((1, 0, 2)).reshape((2, n_samples, 5, -1, 2)))[:, 0, :, :,
                          :]  # 2 x 5 x 100 x 2( x/y)

    # Compute stress contribution (to the mean stress) per term
    n_terms = len(model_weights) // 2
    stress_pred_terms = np.zeros(shape = stress_pred_mean.shape + (n_terms + 1,))
    for i in range(n_terms):
        temp_weights = [model_weights[j] * (i == j // 2 or j==n_terms* 2) for j in range(n_terms * 2 + 1)]
        model.set_weights(temp_weights)
        temp_preds = model.predict(lam_ut_all).reshape((-1, 2, 2, 2))
        stress_pred_terms[:, :, :, :, i + 1] = stress_pred_terms[:, :, :, :, i] + np.array(temp_preds[:, :, :, 0]).transpose((1, 0, 2)).reshape((2, n_samples, 5, -1, 2))[:, 0, :, :, :]
    model.set_weights(model_weights)

    # Pass predicted stress mean, stress standard deviation, and stress contribution per term into separate function to make plot
    return plot_gcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, stress_pred_terms, terms, model_id, modelFit_mode, blank, plot_dist, n_samples, base_path) + (train_nll2,)

## Create plot for gcann predictions vs data
# stress_pred_mean is the mean stress predicted by the model
# stress_pred_std is the stress std deviation predicted by the model
# stress pred_terms is the contribution to the predicted mean stress for each term
# All other parameters are identical to plot_gcann
def plot_gcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, stress_pred_terms, terms, model_id, modelFit_mode, blank, plot_dist, n_samples, base_path):
    # Define indices for dev and test sets for purpose of computing overall NLL values
    idxs_train = [int(x, 16) for x in modelFit_mode]
    # print(idxs_train)

    # Reshape data
    stretch_plot = stretches.reshape((2, n_samples, 5, -1, 2))[0, 0, :, :, :]  # 5 x 100 x 2
    stress_true = stresses.reshape((2, n_samples, 5, -1, 2)) # 2 x 5(n_ex) x 5 x 100 x 2( x/y)
    stress_pred_mean = stress_pred_mean.reshape((2, 5, -1, 2)) # 2 x 5(n_ex) x 5 x 100 x 2( x/y)
    stress_pred_std  = stress_pred_std.reshape((2, 5, -1, 2)) # 2 x 5(n_ex) x 5 x 100 x 2( x/y)

    # Setup plot parameters
    plt.rcParams['text.usetex'] = False
    plt.rcParams['figure.figsize'] = [30, 20 if terms else 15]
    plt.rcParams['figure.constrained_layout.use'] = True

    # Create plots
    fig, axes = plt.subplots(3, 5)
    axes = flatten([list(x) for x in zip(*axes[0:2])]) + list(axes[2])
    direction_strings = ["w", "s"] * 5 + ["x", "y", "x", "y", "x"] # labels for axes
    n_spaces = 55 # used to center strip-x and off-x over corresponding graphs
    titles = ["strip-w", None, "off-w", None, "equibiax-ws", None, "off-s", None, "strip-s", None,
              " " * n_spaces + "strip-x", None," " * n_spaces + "off-x", None, "equibiax-xy"] # titles for plots
    # Reshape input stretches to work well for plotting
    inputs = [[stretch_plot[j, :, k] for k in range(2)] for i in range(2) for j in range(5) ]
    inputs = [[x[i] if np.max(x[i]) > 1.0 else ((x[1 - i] - 1) * 1e-9 + 1) for i in range(2)] for x in inputs]
    inputs = flatten(inputs)[0:15]

    # Reshape experimental stresses for plotting
    outputs = [stress_true[i, :, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15] # 5 x 100 each

    # Reshape model predictions for plotting
    if terms:
        pred_terms = [stress_pred_terms[i, j, :, k, :] for i in range(2) for j in range(5) for k in range(2)][0:15]
    pred_mean = [stress_pred_mean[i, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15]
    pred_std = [stress_pred_std[i, j, :, k] for i in range(2) for j in range(5) for k in range(2)][0:15]

    # Create colormap for shading model contributions by term
    cmap = plt.cm.get_cmap('jet_r', 17)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    num_points = 17
    train_losses = []
    test_losses = []
    for i in range(len(axes)): # Iterate over all 15 plots
        if terms: # if plotting term contributions...
            n_terms = stress_pred_terms.shape[-1] - 1 # stress_pred terms marks upper and lower bounds of each term
            for j in range(n_terms):
                # Create plot that fills between the lower and upper bound
                axes[i].fill_between(inputs[i], pred_terms[i][:, j], pred_terms[i][:, j + 1], lw=0,
                                     zorder=j + 1, color=cmaplist[j],
                                     label=j + 1)
                # Plot a black line for each upper bound
                axes[i].plot(inputs[i], pred_terms[i][:, j + 1],  lw=0.4, zorder=23, color='k')
        else: # If plotting predicted distribution
            # Shade area between 1 std deviation below and 1 std deviation above the mean
            axes[i].fill_between(inputs[i], pred_mean[i] - pred_std[i], pred_mean[i] + pred_std[i], lw=0, zorder=0, color="#384ebc", alpha = 0.25,
                             label=i + 1)

            # Compute negative log likelihood loss for the current graph
            eps = 1e-6
            errors = 0.5 * (np.log(2 * np.pi * (pred_std[i] ** 2 + eps)) + (outputs[i][:, :] - pred_mean[i]) ** 2 / (
                        pred_std[i] ** 2 + eps))

            nll = np.mean(errors)

            # Add NLL to train or test loss
            if i in idxs_train:
                train_losses += [nll]
                # if not blank:
                #     print("Error2: ")
                #     print(tf.reduce_mean(errors))
            else:
                test_losses += [nll]

        # Draw a black line on the predicted mean
        axes[i].plot(inputs[i], pred_mean[i], lw=4, zorder=24, color='k')

        # Set axis limits
        min_P = np.min(outputs[i])
        max_P = np.max(outputs[i])
        min_x = np.min(inputs[i])
        max_x = np.max(inputs[i])
        if np.max(inputs[i]) - np.min(inputs[i]) < 1e-6:
            axes[i].set_xticks([np.min(inputs[i]), np.max(inputs[i])])
            axes[i].set_xticklabels(['1', '1'])
        axes[i].set_xlim([min_x, max_x])
        axes[i].set_ylim([0.0, max_P])

        # Downsample number of experimental data points to plot
        if inputs[i].shape[0] > num_points:
            input_old = inputs[i]
            inputs[i] = np.linspace(np.min(input_old), np.max(input_old), num_points)
            outputs[i] = np.array([np.interp(inputs[i], input_old, outputs[i][j, :]) for j in range(n_samples)])

        # Compute mean and variance of experimental data
        data_mean = np.mean(outputs[i], axis=0)
        data_std = np.std(outputs[i], axis=0)
        data_std_sample = np.std(outputs[i], axis=0, ddof=1)
        # Compute NLL if model mean and variance exactly matched data mean and variance
        eps = 1e-6
        errors = 0.5 * (np.log(2 * np.pi * (data_std ** 2 + eps)) + (outputs[i][:, :] - data_mean) ** 2 / (
                data_std ** 2 + eps))  # Result should be 5 x 100
        nll_min = np.mean(errors)
        if terms or not plot_dist: # If plotting raw data
            for j in range(n_samples): # Create a scatter plot for each sample
                scatterplot = axes[i].scatter(inputs[i], outputs[i][j, :], s=300, zorder=25, lw=3, facecolors='w', edgecolors='k',
                                        clip_on=False)
        else: # If plotting distribution of experimental data
            # shade area from one std deviation below to one std deviation above data mean
            axes[i].fill_between(inputs[i], data_mean - data_std_sample, data_mean + data_std_sample, lw=0, zorder=1,
                                 color="#FF0000", alpha = 0.25, label=i + 1)
            # Draw line for mean of data
            axes[i].plot(inputs[i], data_mean, lw=4, zorder=24, color='#FF0000')


        # Create axis labels
        n_digits = min(int(np.log10(max_P)), 0) + 1
        axes[i].set_xlabel(direction_strings[i] + " stretch [-]", labelpad=-40)
        axes[i].set_ylabel(direction_strings[i] + " stress [kPa]", labelpad=-20 - 10 * n_digits)
        axes[i].minorticks_on()

        # Create axis ticks
        xt = [np.min(inputs[i]), np.max(inputs[i])]
        axes[i].set_xticks(xt)
        yt = [np.min(outputs[i]), np.max(outputs[i])]
        axes[i].set_yticks(yt)
        max_out = np.max(outputs[i])
        axes[i].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        axes[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i' if max_out > 10 else '%.2f'))

        # Create secondary axis for creating the NLL label
        secax = axes[i].twiny()
        secax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,
            labelbottom=False, labeltop=False)

        if not terms:
            secax.set_xlabel(f"Extra NLL = {nll - nll_min:.2f}", labelpad=-50) # Label extra NLL for each plot

        if titles[i] is not None:
            axes[i].set_title(titles[i], y=1.05, usetex=False) # Set title for each plot


    train_nll = None
    test_nll = None
    if terms:
        labels = [x for In in range(1, 3) for x in
                  [f"$(I_{In} - 3)$", f"exp$( (I_{In} - 3))$", f"$(I_{In} - 3)^2$", f"exp$( (I_{In} - 3)^2)$"]]
        labels = labels + [x for dir in ["I_{4w}", "I_{4s}", "I_{4s_{I, II}}"] for x in
                           [f"exp$({dir}) -  {dir}$", f"$({dir} - 1)^2$", f"exp$( ({dir} - 1)^2)$", ]]
        legend_handles = [Patch(color=c) for c in cmaplist] + [scatterplot]
        labels += ["data"]
        leg = fig.legend(loc="lower center", ncols=4, handles=legend_handles, labels=labels,
                         mode="expand", fontsize=40)
        leg.get_frame().set_alpha(0)

        leg_height = 0.24
        rect_height = 0.263
        x_offset = 0.0
        engine = fig.get_layout_engine()
        engine.set(rect=(0.005, leg_height, 0.99, 0.995 - leg_height), wspace=0.04)
        for j in range(5):
            rec = plt.Rectangle((0.2 * j, leg_height + rect_height), 0.2, 1 - leg_height - rect_height, fill=False,
                                lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        for j in range(2):
            rec = plt.Rectangle((0.4 * j, leg_height), 0.4, rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        rec = plt.Rectangle((0.8, leg_height), 0.2, rect_height, fill=False, lw=2)
        rec.set_zorder(1000)
        rec = fig.add_artist(rec)
    else:

        train_nll = sum(train_losses) / len(train_losses)
        test_nll = sum(test_losses) / len(test_losses) if len(test_losses) > 0 else 0

        rect_height = 0.349
        x_offset = 0.
        engine = fig.get_layout_engine()
        engine.set(rect=(0.005, 0.005, 0.99, 0.99), wspace=0.04)
        for j in range(5):
            rec = plt.Rectangle((0.2 * j-x_offset * (j < 3), rect_height), 0.2 + x_offset * (j == 2), 1 - rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        for j in range(2):
            rec = plt.Rectangle((0.4 * j-x_offset, 0), 0.4 + x_offset * (j==1), rect_height, fill=False, lw=2)
            rec.set_zorder(1000)
            rec = fig.add_artist(rec)
        rec = plt.Rectangle((0.8, 0), 0.2, rect_height, fill=False, lw=2)
        rec.set_zorder(1000)
        rec = fig.add_artist(rec)

    # Render and save plot
    name = "terms" if terms else ("variance2" if plot_dist else "variance1")
    if blank:
        plt.savefig(f"{base_path}/{modelFit_mode}/raw_data.pdf",
                    transparent=False,
                    facecolor='white')
    else:
        plt.savefig(f"{base_path}/{modelFit_mode}/gcann_{model_id}_{name}.pdf",
                    transparent=False,
                    facecolor='white')

    plt.close()
    return train_nll, test_nll

