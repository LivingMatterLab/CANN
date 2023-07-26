from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 14
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.axisbelow'] = True


def r2_score_own(truth, pred):
    r2 = r2_score(truth, pred)
    return max(r2, 0.0)


# plot training and validation losses
def plotLoss(history, savePath, trialName):
    fig, axe = plt.subplots(figsize=[6, 5])  # inches
    axe.plot(history.history['loss'], label='Training')
    axe.plot(history.history['val_loss'], label='Validation')
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(savePath + 'Plot_loss_' + trialName + '.png')
    plt.close()


# plot model predictions for all five data sets
def plotPred(model, inputAll, outputAll, savePath, trialName):
    fig1 = plt.figure(figsize=(18, 15))
    spec1 = gridspec.GridSpec(ncols=5, nrows=5, figure=fig1)

    predictions = model.predict([np.expand_dims(inputAll[:, :, 1], axis=2), np.expand_dims(inputAll[:, :, 0], axis=2)])

    legendLbl = ['data', r'$[I_1-3]$', r'$exp([I_1-3])$', r'$ln(1-[I_1-3])$',
                 r'$[I_1-3]^2$', r'$exp([I_1-3]^2)$', r'$ln(1-[I_1-3]^2)$',
                 r'$[I_2-3]$', r'$exp([I_2-3])$', r'$ln(1-[I_2-3])$',
                 r'$[I_2-3]^2$', r'$exp([I_2-3]^2)$', r'$ln(1-[I_2-3]^2)$']

    times = []

    for j in range(len(inputAll)):
        timeArr = np.zeros((len(inputAll[j]), 1))

        count = 0
        for k in range(len(timeArr)):
            timeArr[k] = count + inputAll[j, k, 0]
            count = timeArr[k]

        times.append(timeArr)

    for j in range(5):
        ax = fig1.add_subplot(spec1[j, 0])

        inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
        figSavePath = savePath + trialName + '.png'
        plotFit(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), '', figSavePath,
                '', legendLbl)
        model.load_weights(savePath + trialName + '.tf')


# display invariant based model predictions
def plotFit(fig_ax1, fitModel, inData, time, sig, stress_predict, plotType, path2saveResults, modelFit_mode, labels):
    ColorI = [0.177423, 0.437527, 0.557565]

    fig_ax1.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
    fig_ax1.set_xticklabels(['0', '', '', '', '', '', '', '350'])
    fig_ax1.set_xlim(0, 360)
    fig_ax1.set_xlabel('time [s]', labelpad=-17)

    fig_ax1.set_yticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
    fig_ax1.set_yticklabels(['-2.5', '', '', '', '', '0.0'])
    fig_ax1.set_ylim(-2.5, 0.1)
    fig_ax1.set_ylabel(r'$\sigma$ [kPa]', labelpad=-28)

    if labels == '':
        dataLbl = ''
        fitLbl = ''
    else:
        dataLbl = labels[0]
        fitLbl = labels[-1]

    fig_ax1.scatter(time, sig, s=20, zorder=25, lw=0.5, facecolors='w', edgecolors='k', clip_on=False, label=dataLbl)

    if plotType == 'c':

        model_weights_0 = fitModel.get_weights()
        cann_weights = model_weights_0[12].flatten()
        terms = len(cann_weights)

        cmap = plt.cm.get_cmap('jet', 30)
        cmaplist = [cmap(i) for i in range(cmap.N)]

        color_map(fig_ax1, inData, fitModel, model_weights_0, cmaplist, terms, time, labels)

    else:
        fig_ax1.plot(time, stress_predict, zorder=26, lw=1, color=ColorI, label=fitLbl)

    score = r2_score_own(sig, stress_predict)
    nrmse = np.sqrt(mean_squared_error(sig, stress_predict)) / np.abs(np.mean(sig))

    if modelFit_mode == 'train':
        fig_ax1.text(120, -2, r'$R_{train}^2$' + f"={score:.2f}", fontsize=18)
    elif modelFit_mode == 'test':
        fig_ax1.text(120, -2, r'$R_{test}^2$' + f"={score:.2f}", fontsize=18)
    else:
        fig_ax1.text(120, -2, r'$R^2$' + f"={score:.2f}", fontsize=18)

    fig_ax1.text(120, -1.6, f"NRMSE={nrmse:.2f}", fontsize=18)
    plt.tight_layout()
    plt.savefig(path2saveResults)
    return


# display principal stretch based model predictions
def plotFit_pStr(fig_ax1, fitModel, inData, time, sig, stress_predict, plotType, path2saveResults, modelFit_mode, labels):
    ColorI = [0.177423, 0.437527, 0.557565]

    fig_ax1.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
    fig_ax1.set_xticklabels(['0', '', '', '', '', '', '', '350'])
    fig_ax1.set_xlim(0, 360)
    fig_ax1.set_xlabel('time [s]', labelpad=-17)

    fig_ax1.set_yticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
    fig_ax1.set_yticklabels(['-2.5', '', '', '', '', '0.0'])
    fig_ax1.set_ylim(-2.5, 0.1)
    fig_ax1.set_ylabel(r'$\sigma$ [kPa]', labelpad=-28)

    if labels == '':
        dataLbl = ''
    else:
        dataLbl = labels[0]

    fig_ax1.scatter(time, sig, s=20, zorder=25, lw=0.5, facecolors='w', edgecolors='k', clip_on=False, label=dataLbl)

    if plotType == 'c':

        model_weights_0 = fitModel.get_weights()
        ogden_weights = model_weights_0[0].flatten()
        terms = len(ogden_weights)

        cmap = plt.cm.get_cmap('jet', 50)
        cmaplist = [cmap(i) for i in range(cmap.N)]

        color_map_pStr(fig_ax1, inData, fitModel, model_weights_0, cmaplist, terms, time, labels)

    else:
        fig_ax1.plot(time, stress_predict, zorder=26, lw=1, color=ColorI, label=labels[-1])

    score = r2_score_own(sig, stress_predict)
    nrmse = np.sqrt(mean_squared_error(sig, stress_predict)) / np.abs(np.mean(sig))

    if modelFit_mode == 'train':
        fig_ax1.text(120, -2, r'$R_{train}^2$' + f"={score:.2f}", fontsize=18)
    else:
        fig_ax1.text(120, -2, r'$R_{test}^2$' + f"={score:.2f}", fontsize=18)

    fig_ax1.text(120, -1.6, f"NRMSE={nrmse:.2f}", fontsize=18)
    plt.tight_layout()
    plt.savefig(path2saveResults)

    return


# get arrays of zeros in the shape of the model weight arrays
def GetZeroList(model_weights):
    model_zeros = []
    for i in range(len(model_weights)):
        model_zeros.append(np.zeros_like(model_weights[i]))
    return model_zeros


# plot invariant based model fit with color coded contribution of each term
def color_map(ax2, inData, model_SS, model_weights, cmaplist, terms, timePts, labels):
    predictions = np.zeros([inData[0].shape[1], terms])
    indices = [29, 26, 23, 21, 19, 17, 0, 3, 6, 8, 10, 12]
    w_index = [6, 7, 8, 9, 10, 11, 5, 4, 3, 2, 1, 0]
    # w_index = [0, 1, 2, 3, 4, 5, 11, 10, 9, 8, 7, 6]

    for i in range(len(model_weights[12])):
        # print(model_weights[0][i], cmaplist[i])
        model_plot = GetZeroList(model_weights)
        model_plot[12][w_index[i]] = model_weights[12][w_index[i]]
        model_plot[0:11] = model_weights[0:11]
        model_plot[13] = model_weights[13]
        model_plot[14] = model_weights[14]
        model_SS.set_weights(model_plot)

        lower = np.sum(predictions, axis=1)
        upper = lower + model_SS.predict(inData).flatten()
        predictions[:, i] = model_SS.predict(inData).flatten()

        if labels=='':
            lbl = ''
        else:
            lbl = labels[w_index[i]+1]

        ax2.fill_between(timePts.flatten(), lower.flatten(), upper.flatten(), zorder=i + 1, alpha=1.0,
                         color=cmaplist[indices[w_index[i]]], label=lbl)


# plot principal stretch based model fit with color coded contribution of each term
def color_map_pStr(ax2, inData, model_SS, model_weights, cmaplist, terms, timePts, labels):
    predictions = np.zeros([inData[0].shape[1], terms])
    indices = [21, 28, 19, 30, 17, 32, 15, 34, 13, 36, 11, 38, 9, 40, 6, 43, 3, 46, 0, 49]
    w_index = [18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    for i in range(len(model_weights[0])):
        # print(model_weights[0][i], cmaplist[i])
        model_plot = GetZeroList(model_weights)
        model_plot[0][w_index[i]] = model_weights[0][w_index[i]]
        model_plot[1] = model_weights[1]
        model_plot[2] = model_weights[2]
        model_SS.set_weights(model_plot)

        lower = np.sum(predictions, axis=1)
        upper = lower + model_SS.predict(inData).flatten()
        predictions[:, i] = model_SS.predict(inData).flatten()

        if labels=='':
            lbl = ''
        else:
            lbl = labels[w_index[i]+1]

        ax2.fill_between(timePts.flatten(), lower.flatten(), upper.flatten(), zorder=i + 1, alpha=1.0,
                         color=cmaplist[indices[w_index[i]]], label=lbl)


# plot predictions for the vanilla RNN model
def plotPred_rnn(model, inputAll, outputAll, savePath, trialName):
    fig1 = plt.figure(figsize=(18, 15))
    spec1 = gridspec.GridSpec(ncols=5, nrows=5, figure=fig1)

    predictions = model.predict(inputAll)

    times = []

    for j in range(len(inputAll)):
        timeArr = np.zeros((len(inputAll[j]), 1))

        count = 0
        for k in range(len(timeArr)):
            timeArr[k] = count + inputAll[j, k, 0]
            count = timeArr[k]

        times.append(timeArr)

    for j in range(5):
        ax = fig1.add_subplot(spec1[j, 0])

        if j == 4:
            mode = "test"
        else:
            mode = "train"

        inData = [np.expand_dims(inputAll[j, :, 1], axis=(0, 2)), np.expand_dims(inputAll[j, :, 0], axis=(0, 2))]
        figSavePath = savePath + '/' + trialName + '.png'
        plotFit(ax, model, inData, times[j], np.squeeze(outputAll[j]), np.squeeze(predictions[j]), '', figSavePath,
                mode, '')
