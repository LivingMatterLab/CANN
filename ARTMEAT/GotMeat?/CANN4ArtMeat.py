"""
Last modified April 2024

@author: Kevin Linka, Skyler St. Pierre
"""

import tensorflow.keras.backend as K
import json
import pandas as pd
import os
from plottingArtMeat import *
from Models_artmeat_inv import *

from sklearn.metrics import r2_score


filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()


def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)


def flatten(l):
    return [item for sublist in l for item in sublist]


def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth, Prediction)
    return max(R2, 0.0)


def getStressStrain(Region):
    if Region == 'PB_SAUS':  # stretch 1.15 tension
        P_ut = dfs.iloc[:, 1].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 0].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 4].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 3].dropna().astype(np.float64).values

    elif Region == 'TOFURKY':  # stretch 1.1 tension
        P_ut = dfs.iloc[:, 7].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 6].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 10].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 9].dropna().astype(np.float64).values

    elif Region == 'FIRM_TF':  # stretch 1.15 tensino
        P_ut = dfs.iloc[:, 13].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 12].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 16].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 15].dropna().astype(np.float64).values

    elif Region == 'XFIRM_TF':  # stretch 1.1 tension
        P_ut = dfs.iloc[:, 19].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 18].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 22].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 21].dropna().astype(np.float64).values

    elif Region == 'PB_HOTDOG':  # stretch 1.2 tension
        P_ut = dfs.iloc[:, 25].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 24].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 28].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 27].dropna().astype(np.float64).values

    elif Region == 'RL_HOTDOG':  # stretch 1.35 tension
        P_ut = dfs.iloc[:, 31].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 30].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 34].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 33].dropna().astype(np.float64).values

    elif Region == 'SPAM_TK':  # stretch 1.15 tension
        P_ut = dfs.iloc[:, 37].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 36].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 40].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 39].dropna().astype(np.float64).values

    elif Region == 'RL_SAUS':  # stretch 1.15 tension
        P_ut = dfs.iloc[:, 43].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 42].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 46].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 45].dropna().astype(np.float64).values

    return P_ut, lam_ut, P_ss, gamma_ss


def traindata(modelFit_mode):
    if modelFit_mode == 'T':
        model_given = model_UT
        input_train = lam_ut[20:]
        output_train = P_ut[20:]
        sample_weights = np.array([1.0] * input_train.shape[0]) / np.max(np.abs(P_ut[20:]))

    elif modelFit_mode == "C":
        model_given = model_UT
        input_train = lam_ut[:21]
        output_train = P_ut[:21]
        sample_weights = np.array([1.0] * input_train.shape[0]) / np.max(np.abs(P_ut[:21]))

    elif modelFit_mode == "SS":
        model_given = model_SS
        input_train = gamma_ss
        output_train = P_ss
        sample_weights = np.array([1.0] * input_train.shape[0]) / np.max(np.abs(P_ss[20:]))

    elif modelFit_mode == "TC_and_SS":
        model_given = model
        input_train = [[lam_ut], [gamma_ss]]
        output_train = [[P_ut], [P_ss]]
        sample_weights_tc = np.array([1.0] * lam_ut.shape[0])
        sample_weights_tc[20:] = 1/np.max(np.abs(P_ut[20:]))  # weight by max tension
        sample_weights_tc[:20] = 1/np.max(np.abs(P_ut[:20]))  # weight by max compression
        sample_weights_ss = np.array([1.0] * gamma_ss.shape[0]) / np.max(np.abs(P_ss))  # weight by max shear
        sample_weights = [[sample_weights_tc], [sample_weights_ss]]

    return model_given, input_train, output_train, sample_weights



def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights):
    mse_loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]
    opti1 = tf.optimizers.Adam(learning_rate=0.001)  # learning_rate=0.001

    model_given.compile(loss=mse_loss,
                        optimizer=opti1,
                        metrics=metrics)

    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=300, restore_best_weights=True)
    # patience 3000
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="loss",
        filepath=path_checkpoint,
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model_given.fit(input_train,
                              output_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              callbacks=[es_callback, modelckpt_callback],
                              shuffle=True,
                              verbose=2,
                              sample_weight=sample_weights)

    return model_given, history


# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]


# Stress for uniaxial tension, P11
def Stress_calc_TC(inputs):
    (dPsidI1, dPsidI2, Stretch) = inputs
    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')
    minus = two * (dPsidI1 * 1 / K.square(Stretch) + dPsidI2 * 1 / K.pow(Stretch, 3))
    stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus

    return stress


# Simple stress, P12
def Stress_cal_SS(inputs):
    (dPsidI1, dPsidI2, gamma) = inputs
    two = tf.constant(2.0, dtype='float32')
    stress = two * gamma * (dPsidI1 + dPsidI2)

    return stress


# Complete model architecture definition
def modelArchitecture(Psi_model):
    # Stretch and Gamma as input
    Stretch = keras.layers.Input(shape=(1,),
                                 name='Stretch')
    Gamma = keras.layers.Input(shape=(1,),
                               name='gamma')

    # specific Invariants UT
    I1_UT = keras.layers.Lambda(lambda x: x ** 2 + 2.0 / x)(Stretch)
    I2_UT = keras.layers.Lambda(lambda x: 2.0 * x + 1 / x ** 2)(Stretch)
    # specific Invariants SS
    I1_SS = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma)
    I2_SS = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma)

    # % load specific models
    Psi_UT = Psi_model([I1_UT, I2_UT])
    Psi_SS = Psi_model([I1_SS, I2_SS])

    # derivative UT
    dWI1_UT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I1_UT])
    dWdI2_UT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I2_UT])
    # derivative SS
    dWI1_SS = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS, I1_SS])
    dWdI2_SS = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS, I2_SS])

    # Stress UT
    Stress_UT = keras.layers.Lambda(function=Stress_calc_TC,
                                    name='Stress_UT')([dWI1_UT, dWdI2_UT, Stretch])
    # Stress SS
    Stress_SS = keras.layers.Lambda(function=Stress_cal_SS,
                                    name='Stress_SS')([dWI1_SS, dWdI2_SS, Gamma])

    # Define model for different load case
    model_UT = keras.models.Model(inputs=Stretch, outputs=Stress_UT)
    model_SS = keras.models.Model(inputs=Gamma, outputs=Stress_SS)
    # Combined model
    model = keras.models.Model(inputs=[model_UT.inputs, model_SS.inputs], outputs=[model_UT.outputs, model_SS.outputs])

    return model_UT, model_SS, Psi_model, model


### User parameters to change ################
train = True  # train or used previously trained weighted (True/False)
epochs = 20000  # used 20000 as upper bound on epochs, most cases stopped well before
batch_size = 8  # used 8
folder_name = 'MeatModels'  # make folder with results; rename each time you train
weight_flag = True  # print out weights (True/False)
weight_plot_Map = True  # plot results (True/False)

### Choose regularization type & penalty amount (pen = 0 used for Got Meat? paper)
# Option: 'L1', 'L2'
reg = 'L1'
pen = 0  # Use 0 for no regularization

### Choose which model type to build CANN architecture with
# Options: 'Invariant'
# 'Invariant' is invariant-based and contains I1, I2, I1^2, I2^2 and all with () and exp() activations, 8 total terms
model_type = 'Invariant'

### Choose which loading modes to train with
# Options: 'T', 'C', 'SS', 'TC_and_SS' (tension, compression, simple shear, tension/compression & simple shear)
modelFit_mode_all = ['TC_and_SS']

### Choose which types of artificial meat to train with
# Options: ['PB_SAUS', 'FIRM_TF', 'XFIRM_TF', 'PB_HOTDOG', 'RL_HOTDOG', 'TOFURKY', 'SPAM_TK', 'RL_SAUS']
Region_all = ['RL_HOTDOG', 'PB_HOTDOG']

################################################

path2saveResults_0 = 'Results/' + filename + '/' + folder_name
makeDIR(path2saveResults_0)
Model_summary = path2saveResults_0 + '/Model_summary.txt'

### Import excel file ###
# file_name = 'input/ArtMeat.xlsx'
file_name = 'input/GotMeat_full.xlsx'

dfs = pd.read_excel(file_name, sheet_name='Sheet1', engine='openpyxl')
count = 1
for id1, Region in enumerate(Region_all):  # iterate over regions

    for id2, modelFit_mode in enumerate(modelFit_mode_all):  # iterate over loading modes

        print(40 * '=')
        print("Comp {:d} / {:d}".format(count, len(Region_all) * len(modelFit_mode_all)))
        print(40 * '=')
        print("Region: ", Region, "| Fitting Mode: ", modelFit_mode)
        print(40 * '=')
        count += 1

        path2saveResults = os.path.join(path2saveResults_0, Region, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults, 'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)

        P_ut, lam_ut, P_ss, gamma_ss = getStressStrain(Region)
        # CANN models
        Psi_model, terms = StrainEnergy_invariant(reg, pen)  # Invariant-based model

        model_UT, model_SS, Psi_model, model = modelArchitecture(Psi_model)

        with open(Model_summary, 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

        # Model training

        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode)

        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        path_checkpoint = path2saveResults_check + '/best_weights'
        if train:
            model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint,
                                                   sample_weights)

            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
            Psi_model.save_weights(Save_weights, overwrite=True)

            # Plot loss function
            loss_history = history.history['loss']
            fig, axe = plt.subplots(figsize=[6, 5])
            plotLoss(axe, history, epochs, path2saveResults, Region, modelFit_mode)
            plt.close(fig)

        else:
            Psi_model.load_weights(Save_weights, by_name=False, skip_mismatch=False)

        # Get CANN model response
        Stress_predict_UT = model_UT.predict(lam_ut)
        Stress_predict_SS = model_SS.predict(gamma_ss)

        # Plotting
        fig1, ax1 = plt.subplots(figsize=(12.5, 8.33))
        plotTen(ax1, lam_ut[20:], P_ut[20:], Stress_predict_UT[20:], Region, path2saveResults, modelFit_mode)
        plt.close(fig1)
        fig2, ax2 = plt.subplots(figsize=(12.5, 8.33))
        plotCom(ax2, lam_ut[:21], P_ut[:21], Stress_predict_UT[:21], Region, path2saveResults, modelFit_mode)
        R2_t = r2_score(P_ut[20:], Stress_predict_UT[20:])
        R2_c = r2_score(P_ut[:21], Stress_predict_UT[:21])
        plt.close(fig2)
        fig3, ax3 = plt.subplots(figsize=(12.5, 8.33))
        plotShear(ax3, gamma_ss[20:], P_ss[20:], Stress_predict_SS[20:], Region, path2saveResults, modelFit_mode)
        R2ss = r2_score(P_ss, Stress_predict_SS)
        plt.close(fig3)

        if weight_flag:
            weight_matrix = np.empty((terms, 2))
            for i in range(terms):
                value = Psi_model.get_weights()[i][0][0]
                weight_matrix[i, 0] = value
                weight_matrix[:, 1] = Psi_model.get_layer('wx2').get_weights()[0].flatten()
            print("weight_matrix")
            print(weight_matrix)

        Config = {"Region": Region, "modelFit_mode": modelFit_mode, 'model_type': model_type, 'Reg': reg, 'Penalty': pen, "R2_t": R2_t, "R2_c": R2_c, "R2_ss": R2ss,
                  "weights": weight_matrix.tolist()}
        json.dump(Config, open(path2saveResults + "/Config_file.txt", 'w'))

        # plotting_ color map
        model_weights_0 = Psi_model.get_weights()

        if weight_plot_Map:
            fig4, ax4 = plt.subplots(figsize=(12.5, 8))
            plotMapTen(ax4, Psi_model, model_weights_0, model_UT, terms, lam_ut[20:],
                       P_ut[20:], Region, path2saveResults, modelFit_mode, model_type)
            plt.close(fig4)
            fig5, ax5 = plt.subplots(figsize=(12.5, 8))
            plotMapCom(ax5, Psi_model, model_weights_0, model_UT, terms, lam_ut[:21],
                       P_ut[:21], Region, path2saveResults, modelFit_mode, model_type)
            plt.close(fig5)
            fig6, ax6 = plt.subplots(figsize=(12.5, 8))
            plotMapShear(ax6, Psi_model, model_weights_0, model_SS, terms, gamma_ss[20:],
                         P_ss[20:], Region, path2saveResults, modelFit_mode, model_type)
            plt.close(fig6)

