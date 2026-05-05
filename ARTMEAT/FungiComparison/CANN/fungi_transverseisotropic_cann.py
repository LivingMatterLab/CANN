import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import json
import pandas as pd
import os
import copy
from sklearn.metrics import r2_score

# Check Versions (alternate working versions: Numpy 1.19.1, Matplotlib 3.2.2, TF 2.2.0, Pandas 1.0.5)
print('Numpy: ' + np.__version__)  # 1.23.5
print('Matplotlib: ' + matplotlib.__version__)  # 3.2.2
print('Tensorflow: ' + tf.__version__)  # 2.12.0
print('Keras: ' + keras.__version__)
print('Pandas: ' + pd.__version__)  # 1.5.3

# Import excel file, change to match where you saved the file
path = '/Users/ssp/Desktop/CANN_PI/TransverseIsotropy_Meat/'  # change to where you download this
dfs = pd.read_excel(path + 'CANNs_Fungi_Data.xlsx', sheet_name='Sheet1', engine='openpyxl')  # download openpyxl needed


# Make path to save results to
def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)


filename = 'Fungi_TransverseIsotropic_CANN'  # Change to keep track of different data e.g. Brain, Skin, Muscle, etc.
path2saveRaw = path + 'Results/' + filename + '/RawData'
makeDIR(path2saveRaw)


def getStressStrain(Sample):
    if Sample == 'OMNI':
        P_ut_CP = dfs.iloc[3:, 1].dropna().astype(np.float64)
        lam_ut_CP = dfs.iloc[3:, 0].dropna().astype(np.float64)

        P_ss_CP = dfs.iloc[3:, 3].dropna().astype(np.float64).values
        gamma_ss_CP = dfs.iloc[3:, 2].dropna().astype(np.float64).values

        P_ut_IP = dfs.iloc[3:, 6].dropna().astype(np.float64)
        lam_ut_IP = dfs.iloc[3:, 5].dropna().astype(np.float64)

        P_ss_IP = dfs.iloc[3:, 8].dropna().astype(np.float64).values
        gamma_ss_IP = dfs.iloc[3:, 7].dropna().astype(np.float64).values
    if Sample == 'MEATI':
        P_ut_CP = dfs.iloc[3:, 11].dropna().astype(np.float64)
        lam_ut_CP = dfs.iloc[3:, 10].dropna().astype(np.float64)

        P_ss_CP = dfs.iloc[3:, 13].dropna().astype(np.float64).values
        gamma_ss_CP = dfs.iloc[3:, 12].dropna().astype(np.float64).values

        P_ut_IP = dfs.iloc[3:, 16].dropna().astype(np.float64)
        lam_ut_IP = dfs.iloc[3:, 15].dropna().astype(np.float64)

        P_ss_IP = dfs.iloc[3:, 18].dropna().astype(np.float64).values
        gamma_ss_IP = dfs.iloc[3:, 17].dropna().astype(np.float64).values
    if Sample == 'BEYOND':
        P_ut_CP = dfs.iloc[3:, 21].dropna().astype(np.float64)
        lam_ut_CP = dfs.iloc[3:, 20].dropna().astype(np.float64)

        P_ss_CP = dfs.iloc[3:, 23].dropna().astype(np.float64).values
        gamma_ss_CP = dfs.iloc[3:, 22].dropna().astype(np.float64).values

        P_ut_IP = dfs.iloc[3:, 26].dropna().astype(np.float64)
        lam_ut_IP = dfs.iloc[3:, 25].dropna().astype(np.float64)

        P_ss_IP = dfs.iloc[3:, 28].dropna().astype(np.float64).values
        gamma_ss_IP = dfs.iloc[3:, 27].dropna().astype(np.float64).values

    return P_ut_CP, P_ut_IP, lam_ut_CP, lam_ut_IP, P_ss_CP, P_ss_IP, gamma_ss_CP, gamma_ss_IP


# Define different loading protocols
def traindata(modelFit_mode):  # [CP, IP]
    if modelFit_mode == 'T':
        model_given = model_UT
        input_train = [[lam_ut_CP[20:]], [lam_ut_IP[20:]]]  # INDEX 20 TO END IN THE COLUMN ARE TENSION
        output_train = [[P_ut_CP[20:]], [P_ut_IP[20:]]]
        sample_weights = [[np.array([1.0] * lam_ut_CP[20:].shape[0])], [np.array([1.0] * lam_ut_IP[20:].shape[0])]]

    elif modelFit_mode == "C":
        model_given = model_UT
        input_train = [[lam_ut_CP[:21]], [lam_ut_IP[:21]]]  # INDEX 0 TO 20 IN THE COLUMN ARE COMPRESSION
        output_train = [[P_ut_CP[:21]], [P_ut_IP[:21]]]
        sample_weights = [[np.array([1.0] * lam_ut_CP[:21].shape[0])], [np.array([1.0] * lam_ut_IP[:21].shape[0])]]

    elif modelFit_mode == "SS":
        model_given = model_SS
        input_train = [[gamma_ss_CP], [gamma_ss_IP]]
        output_train = [[P_ss_CP], [P_ss_IP]]
        sample_weights = [[np.array([1.0] * gamma_ss_CP.shape[0])], [np.array([1.0] * gamma_ss_IP.shape[0])]]

    elif modelFit_mode == "TC_and_SS":
        model_given = model
        input_train = [[lam_ut_CP], [lam_ut_IP], [gamma_ss_CP], [gamma_ss_IP]]
        output_train = [[P_ut_CP], [P_ut_IP], [P_ss_CP], [P_ss_IP]]
        # normalize each Ten, Com, Shr by respective max absolute stress
        sample_weights_ut_CP = np.array([1.0] * lam_ut_CP.shape[0])
        sample_weights_ut_CP[20:] = 1 / np.max(np.abs(P_ut_CP[20:]))  # weight by max tension
        sample_weights_ut_CP[:21] = 1 / np.max(np.abs(P_ut_CP[:21]))  # weight by max compression

        sample_weights_ut_IP = np.array([1.0] * lam_ut_IP.shape[0])
        sample_weights_ut_IP[20:] = 1 / np.max(np.abs(P_ut_IP[20:]))  # weight by max tension
        sample_weights_ut_IP[:21] = 1 / np.max(np.abs(P_ut_IP[:21]))  # weight by max compression

        sample_weights_ss_CP = np.array([1.0] * gamma_ss_CP.shape[0]) / np.max(np.abs(P_ss_CP))  # weight by max shear
        sample_weights_ss_IP = np.array([1.0] * gamma_ss_IP.shape[0]) / np.max(np.abs(P_ss_IP))  # weight by max shear

        sample_weights = [[sample_weights_ut_CP], [sample_weights_ut_IP], [sample_weights_ss_CP], [sample_weights_ss_IP]]
    return model_given, input_train, output_train, sample_weights



def regularize(reg, pen):
    if reg == 'L2':
        return keras.regularizers.l2(pen)
    if reg == 'L1':
        return keras.regularizers.l1(pen)


initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1,
                                                      seed=np.random.randint(0, 10000))  # use random integer as seed
initializer_1 = 'glorot_normal'


# Self defined activation functions for exp term
def activation_Exp(x):
    return 1.0 * (tf.math.exp(x) - 1.0)


# Define network block
def SingleInvNet(I1_ref, idi, reg, pen):
    # input: invariant
    I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(
                                I1_ref)  # no activation
    I_1_w21 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(2 + idi) + '1')(I1_ref)  # exp activation

    # input: invariant^2
    I_1_w31 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(3 + idi) + '1')(tf.math.square(I1_ref))  # no activation
    I_1_w41 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(4 + idi) + '1')(tf.math.square(I1_ref))  # exp activation

    collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out


# Define network block
def SingleInvNet_i4_i5(I1_ref, idi, reg, pen):
    # input: invariant^2
    I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(tf.math.square(I1_ref))  # no activation
    I_1_w21 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(2 + idi) + '1')(tf.math.square(I1_ref))  # exp activation

    collect = [I_1_w11, I_1_w21]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out



def StrainEnergyCANN_invariant(reg, pen):
    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4_in = tf.keras.Input(shape=(1,), name='I4')
    I5_in = tf.keras.Input(shape=(1,), name='I5')

    # Put invariants in the reference configuration
    I1_ref = keras.layers.Lambda(lambda x: (x - 3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x - 3.0))(I2_in)
    I4_ref = keras.layers.Lambda(lambda x: (x - 1.0))(tf.math.maximum(I4_in, 1))  # fibers are not load bearing in compr
    I5_ref = keras.layers.Lambda(lambda x: (x - 1.0))(tf.math.maximum(I5_in, 1))  # fibers are not load bearing in compr

    I1_out = SingleInvNet(I1_ref, 0, reg, pen)
    terms = I1_out.get_shape().as_list()[1]  # 4 terms
    I2_out = SingleInvNet(I2_ref, terms, reg, pen)  # 4 terms
    I4_out = SingleInvNet_i4_i5(I4_ref, 8, reg, pen)  # 2 terms
    I5_out = SingleInvNet_i4_i5(I5_ref, 10, reg, pen)  # 2 terms

    ALL_I_out = [I1_out, I2_out, I4_out, I5_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out, axis=1)

    # second layer
    W_ANN = keras.layers.Dense(1, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularize(reg, pen),
                               use_bias=False, activation=None, name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4_in, I5_in], outputs=[W_ANN], name='Psi')

    return Psi_model, 12  # 12 terms with I1, I2, I4, I5


# Defining stress from Transversely Isotropic Paper

def Stress_calc_TC_IP(inputs):  
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, Stretch) = inputs
    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')
    four = tf.constant(4.0, dtype='float32')

    minus = two * (dPsidI1 * 1 / tf.math.pow(Stretch, 2) + dPsidI2 * 1 / tf.math.pow(Stretch, 3))
    stress_matrix = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus
    stress = stress_matrix + two * Stretch * dPsidI4 + four * tf.math.pow(Stretch, 3) * dPsidI5

    return stress  # matrix and fibers


def Stress_calc_TC_CP(inputs):  
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, Stretch) = inputs
    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')

    minus = two * (dPsidI1 * 1 / tf.math.pow(Stretch, 2) + dPsidI2 * 1 / tf.math.pow(Stretch, 3))
    stress_matrix = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus

    return stress_matrix  # only matrix contributes (I1, I2)


# Simple shear stress P12
def Stress_cal_SS_IP(inputs):
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, gamma) = inputs
    two = tf.constant(2.0, dtype='float32')
    six = tf.constant(6.0, dtype='float32')
    four = tf.constant(4.0, dtype='float32')

    stress = two * gamma * (dPsidI1 + dPsidI2 + dPsidI4) + (six * gamma + four * tf.math.pow(gamma, 3)) * dPsidI5

    return stress  # matrix, fiber stretch (I4), and fiber shear (I5)


def Stress_cal_SS_CP(inputs):
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, gamma) = inputs
    two = tf.constant(2.0, dtype='float32')

    stress = two * gamma * (dPsidI1 + dPsidI2 + dPsidI5)

    return stress  # matrix and fiber shear but no fiber stretch


# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]


def modelArchitecture(Psi_model):
    # Stretch and Gamma as input, from excel
    Stretch_CP = keras.layers.Input(shape=(1,), name='Stretch_CP')
    Stretch_IP = keras.layers.Input(shape=(1,), name='Stretch_IP')
    Gamma_CP = keras.layers.Input(shape=(1,), name='gamma_CP')
    Gamma_IP = keras.layers.Input(shape=(1,), name='gamma_IP')

    # specific Invariants UT_IP
    I1_UT_IP = keras.layers.Lambda(lambda x: x ** 2 + 2.0 / x)(Stretch_IP)
    I2_UT_IP = keras.layers.Lambda(lambda x: 2.0 * x + 1 / x ** 2)(Stretch_IP)
    I4_UT_IP = keras.layers.Lambda(lambda x: x ** 2)(Stretch_IP)
    I5_UT_IP = keras.layers.Lambda(lambda x: x ** 4)(Stretch_IP)

    # specific Invariants UT_CP
    I1_UT_CP = keras.layers.Lambda(lambda x: x ** 2 + 2.0 / x)(Stretch_CP)
    I2_UT_CP = keras.layers.Lambda(lambda x: 2.0 * x + 1 / x ** 2)(Stretch_CP)
    I4_UT_CP = keras.layers.Lambda(lambda x: 1/x)(Stretch_CP)  # doesn't go into stress eq
    I5_UT_CP = keras.layers.Lambda(lambda x: 1/x**2)(Stretch_CP)  # doesn't go into stress eq

    # # specific Invariants SS_IP
    I1_SS_IP = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma_IP)
    I2_SS_IP = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma_IP)
    I4_SS_IP = keras.layers.Lambda(lambda x: 1 + x ** 2)(Gamma_IP)
    I5_SS_IP = keras.layers.Lambda(lambda x: (1 + x ** 2) ** 2 + x ** 2)(Gamma_IP)

    # # specific Invariants SS_CP
    I1_SS_CP = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma_CP)
    I2_SS_CP = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma_CP)
    I4_SS_CP = keras.layers.Lambda(lambda x: x)(Gamma_CP)  # doesn't go into stress eq (should be 1 but doesn't matter)
    I5_SS_CP = keras.layers.Lambda(lambda x: 1 + x ** 2)(Gamma_CP)

    # % load specific models
    Psi_UT_IP = Psi_model([I1_UT_IP, I2_UT_IP, I4_UT_IP, I5_UT_IP])  # IP
    Psi_SS_IP = Psi_model([I1_SS_IP, I2_SS_IP, I4_SS_IP, I5_SS_IP])
    Psi_UT_CP = Psi_model([I1_UT_CP, I2_UT_CP, I4_UT_CP, I5_UT_CP])  # CP
    Psi_SS_CP = Psi_model([I1_SS_CP, I2_SS_CP, I4_SS_CP, I5_SS_CP])

    # derivative UT_IP
    dWdI1_UT_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_IP, I1_UT_IP])
    dWdI2_UT_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_IP, I2_UT_IP])
    dWdI4_UT_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_IP, I4_UT_IP])
    dWdI5_UT_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_IP, I5_UT_IP])
    # # derivative SS_IP
    dWdI1_SS_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_IP, I1_SS_IP])
    dWdI2_SS_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_IP, I2_SS_IP])
    dWdI4_SS_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_IP, I4_SS_IP])
    dWdI5_SS_IP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_IP, I5_SS_IP])
    # derivative UT_CP
    dWdI1_UT_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_CP, I1_UT_CP])
    dWdI2_UT_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_CP, I2_UT_CP])
    dWdI4_UT_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_CP, I4_UT_CP])
    dWdI5_UT_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT_CP, I5_UT_CP])
    # # derivative SS_CP
    dWdI1_SS_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_CP, I1_SS_CP])
    dWdI2_SS_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_CP, I2_SS_CP])
    dWdI4_SS_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_CP, I4_SS_CP])
    dWdI5_SS_CP = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS_CP, I5_SS_CP])

    # Stress UT
    Stress_UT_IP = keras.layers.Lambda(function=Stress_calc_TC_IP,
                                    name='Stress_UT_IP')(
        [dWdI1_UT_IP, dWdI2_UT_IP, dWdI4_UT_IP, dWdI5_UT_IP, Stretch_IP])
    # # Stress SS
    Stress_SS_IP = keras.layers.Lambda(function=Stress_cal_SS_IP,
                                    name='Stress_SS_IP')([dWdI1_SS_IP, dWdI2_SS_IP, dWdI4_SS_IP, dWdI5_SS_IP, Gamma_IP])
    # Stress UT
    Stress_UT_CP = keras.layers.Lambda(function=Stress_calc_TC_CP,
                                    name='Stress_UT_CP')(
        [dWdI1_UT_CP, dWdI2_UT_CP, dWdI4_UT_CP, dWdI5_UT_CP, Stretch_CP])
    # # Stress SS
    Stress_SS_CP = keras.layers.Lambda(function=Stress_cal_SS_CP,
                                    name='Stress_SS_CP')([dWdI1_SS_CP, dWdI2_SS_CP, dWdI4_SS_CP, dWdI5_SS_CP, Gamma_CP])

    # Define model for different load case
    model_UT_CP = keras.models.Model(inputs=Stretch_CP, outputs=Stress_UT_CP)
    model_UT_IP = keras.models.Model(inputs=Stretch_IP, outputs=Stress_UT_IP)
    model_SS_CP = keras.models.Model(inputs=Gamma_CP, outputs=Stress_SS_CP)
    model_SS_IP = keras.models.Model(inputs=Gamma_IP, outputs=Stress_SS_IP)
    model_UT = keras.models.Model(inputs=[model_UT_CP.inputs, model_UT_IP.inputs], outputs=[model_UT_CP.outputs, model_UT_IP.outputs])
    model_SS = keras.models.Model(inputs=[model_SS_CP.inputs, model_SS_IP.inputs], outputs=[model_SS_CP.outputs, model_SS_IP.outputs])
    # Combined model
    model = keras.models.Model(inputs=[model_UT_CP.inputs, model_UT_IP.inputs, model_SS_CP.inputs, model_SS_IP.inputs],
                               outputs=[model_UT_CP.outputs, model_UT_IP.outputs, model_SS_CP.outputs, model_SS_IP.outputs])
    return model_UT, model_SS, Psi_model, model, model_UT_CP, model_UT_IP, model_SS_CP, model_SS_IP


# Optimization utilities
def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights):
    mse_loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]
    opti1 = tf.optimizers.Adam(learning_rate=0.001)

    model_given.compile(loss=mse_loss,
                        optimizer=opti1,
                        metrics=metrics)

    # if training loss starts to increase, stop training after 3000 additional epochs = "patience"
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=3000, restore_best_weights=True)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="loss",
        filepath=path_checkpoint,
        verbose=0,
        save_weights_only=True,
        save_best_only=True,  # save only the best weights across all epochs
    )

    history = model_given.fit(input_train,
                              output_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              callbacks=[es_callback, modelckpt_callback],
                              # save best weights if stop early or go through all epochs
                              shuffle=True,
                              verbose=0,  # verbose = 2 will print loss each epoch
                              sample_weight=sample_weights)

    return model_given, history



def plotLoss(axe, history):
    axe.plot(history)
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')



def color_map(ax, stretch, model, model_weights, Psi_model, cmaplist, terms, model_type):
    predictions = np.zeros([stretch.shape[0], terms])
    model_plot = copy.deepcopy(model_weights)  # deep copy model weights
    for i in range(terms):
        if model_type == 'Stretch':
            model_plot = np.zeros_like(model_weights)  # wx1 all set to zero
            model_plot[i] = model_weights[i]  # wx1[i] set to trained value
        else:  # for architectures with multiple layers (invariant)
            model_plot[-1] = np.zeros_like(model_weights[-1])  # wx2 all set to zero
            model_plot[-1][i] = model_weights[-1][i]  # wx2[i] set to trained value

        Psi_model.set_weights(model_plot)
        lower = np.sum(predictions, axis=1)
        upper = lower + model.predict(stretch, verbose=0)[:].flatten()
        predictions[:, i] = model.predict(stretch, verbose=0)[:].flatten()
        ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[i], label=i + 1)
        # plt.legend()
        # if i == 0:  # one or two term models, get the correct color
        #     ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[0],
        #                      label=i + 1)
        # else:
        #     ax.fill_between(stretch[:], lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[4],
        #                      label=i + 1)

        ax.plot(stretch, upper, lw=0.4, zorder=34, color='k')


plt.rcParams['xtick.major.pad'] = 16  # NORMALLY 14 WITH I4 AND I5 ADDED
plt.rcParams['ytick.major.pad'] = 16


def plotMapTen_CP(ax, Psi_model, model_weights, model_UT, terms, lam_ut, P_ut, Sample, path2saveResults, modelFit_mode,
               model_type, numTerms):
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap with the number of terms from the full network
    # this way, we can use 1 or 2 term models and have the colors be the same for those terms
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([])
    ax.set_xlim(1, 1.1)
    if Sample == 'OMNI':
        ax.set_yticks([])  # UPDATE THESE VALUES
        ax.set_ylim(0, 3.4653)
    elif Sample == 'MEATI':
        ax.set_yticks([])
        ax.set_ylim(0, 2.3759)
    elif Sample == 'BEYOND':
        ax.set_yticks([])
        ax.set_ylim(0, 8.4669)
    color_map(ax, lam_ut, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut, P_ut, s=800, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    # plt.legend()
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/TensionCP_' + 'Train' + modelFit_mode + '_' + 'Sample' + Sample + '.pdf')
    plt.close()
    
    
def plotMapTen_IP(ax, Psi_model, model_weights, model_UT, terms, lam_ut, P_ut, Sample, path2saveResults, modelFit_mode,
               model_type, numTerms):
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap with the number of terms from the full network
    # this way, we can use 1 or 2 term models and have the colors be the same for those terms
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([])
    ax.set_xlim(1, 1.1)
    if Sample == 'OMNI':
        ax.set_yticks([])
        ax.set_ylim(0, 4.9381)
    elif Sample == 'MEATI':
        ax.set_yticks([])
        ax.set_ylim(0, 9.7241)
    elif Sample == 'BEYOND':
        ax.set_yticks([])
        ax.set_ylim(0, 8.90652)
    color_map(ax, lam_ut, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut, P_ut, s=800, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/TensionIP_' + 'Train' + modelFit_mode + '_' + 'Sample' + Sample + '.pdf')
    plt.close()


def plotMapCom_CP(ax, Psi_model, model_weights, model_UT, terms, lam_ut, P_ut, Sample, path2saveResults, modelFit_mode,
               model_type, numTerms):
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap with the number of terms from the full network
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([])
    ax.set_xlim(1, 0.9)
    if Sample == 'OMNI':
        ax.set_yticks([])  # UPDATE THESE VALUES
        ax.set_ylim(0, -2.4174)
    elif Sample == 'MEATI':
        ax.set_yticks([])
        ax.set_ylim(0, -2.7768)
    elif Sample == 'BEYOND':
        ax.set_yticks([])
        ax.set_ylim(0, -8.7729)
    color_map(ax, lam_ut, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut, P_ut, s=800, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/CompressionCP_' + 'Train' + modelFit_mode + '_' + 'Sample' + Sample + '.pdf')
    plt.close()


def plotMapCom_IP(ax, Psi_model, model_weights, model_UT, terms, lam_ut, P_ut, Sample, path2saveResults, modelFit_mode,
               model_type, numTerms):
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap with the number of terms from the full network
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([])
    ax.set_xlim(1, 0.9)
    if Sample == 'OMNI':
        ax.set_yticks([])  # UPDATE THESE VALUES
        ax.set_ylim(0, -2.0508)
    elif Sample == 'MEATI':
        ax.set_yticks([])
        ax.set_ylim(0, -3.3028)
    elif Sample == 'BEYOND':
        ax.set_yticks([])
        ax.set_ylim(0, -8.4722)
    color_map(ax, lam_ut, model_UT, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(lam_ut, P_ut, s=800, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/CompressionIP_' + 'Train' + modelFit_mode + '_' + 'Sample' + Sample + '.pdf')
    plt.close()


def plotMapShear_CP(ax, Psi_model, model_weights, model_SS, terms, gamma_ss, P_ss, Sample, path2saveResults, modelFit_mode,
                 model_type, numTerms):
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap with the number of terms from the full network
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([])
    ax.set_xlim(0, 0.1)
    if Sample == 'OMNI':
        ax.set_yticks([])  # UPDATE THESE VALUES
        ax.set_ylim(0, 1.2368)
    elif Sample == 'MEATI':
        ax.set_yticks([])
        ax.set_ylim(0, 1.0404)
    elif Sample == 'BEYOND':
        ax.set_yticks([])
        ax.set_ylim(0, 2.5889)
    color_map(ax, gamma_ss, model_SS, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(gamma_ss, P_ss, s=800, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/ShearCP_' + 'Train' + modelFit_mode + '_' + 'Sample' + Sample + '.pdf')
    plt.close()
    
    
def plotMapShear_IP(ax, Psi_model, model_weights, model_SS, terms, gamma_ss, P_ss, Sample, path2saveResults, modelFit_mode,
                 model_type, numTerms):
    cmap = plt.cm.get_cmap('jet_r', numTerms)  # define the colormap with the number of terms from the full network
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ax.set_xticks([])
    ax.set_xlim(0, 0.1)
    if Sample == 'OMNI':
        ax.set_yticks([])
        ax.set_ylim(0, 0.86795)
    elif Sample == 'MEATI':
        ax.set_yticks([])
        ax.set_ylim(0, 1.2737)
    elif Sample == 'BEYOND':
        ax.set_yticks([])
        ax.set_ylim(0, 2.9934)
    color_map(ax, gamma_ss, model_SS, model_weights, Psi_model, cmaplist, terms, model_type)
    ax.scatter(gamma_ss, P_ss, s=800, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.tight_layout(pad=2)
    plt.savefig(path2saveResults + '/ShearIP_' + 'Train' + modelFit_mode + '_' + 'Sample' + Sample + '.pdf')
    plt.close()


### User parameters ###
train = True
epochs = 20000  # try 5,000-20,000 epochs for a good fit
batch_size = 8
folder_name = 'TransIso_All_Fungi_L1'  # name the folder for your results

### Choose regularization type & penalty amount
# Option: 'L1', 'L2'
reg = 'L1'
pen = 0.05  # Use 0 for no regularization

### Specify model type to build CANN architecture with
model_type = 'Invariant'

### Number of terms to build colormap with (same as Terms except e.g. restricting to only neo Hooke but keeping same colors)
numTerms = 12

### Choose which loading modes to train with
# Options: 'T', 'C', 'SS', 'TC_and_SS' (tension, compression, simple shear, tension/compression & simple shear)
modelFit_mode_all = ['T']

### Choose which types of artificial meat to train with
# Options: 'OMNI', 'MEATI', 'BEYOND'
Sample_all = ['BEYOND']
################################################

path2saveResults_0 = path + 'Results/' + filename + '/' + folder_name
makeDIR(path2saveResults_0)

Model_summary = path2saveResults_0 + '/Model_summary.txt'

# #  Training and validation loop
count = 1
for id1, Sample in enumerate(Sample_all):  # loop through Meati and Omni data

    # R2_all_Samples = []
    for id2, modelFit_mode in enumerate(modelFit_mode_all):  # loop through model training modes

        print(40 * '=')
        print("Comp {:d} / {:d}".format(count, len(Sample_all) * len(modelFit_mode_all)))
        print(40 * '=')
        print("Sample: ", Sample, "| Fitting Mode: ", modelFit_mode)
        print(40 * '=')
        count += 1

        path2saveResults = os.path.join(path2saveResults_0, Sample, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults, 'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)

        P_ut_CP, P_ut_IP, lam_ut_CP, lam_ut_IP, P_ss_CP, P_ss_IP, gamma_ss_CP, gamma_ss_IP = getStressStrain(Sample)  # stress/stretch/shear pairs

        # Model selection
        Psi_model, terms = StrainEnergyCANN_invariant(reg, pen)  # build invariant-based model
        model_UT, model_SS, Psi_model, model, model_UT_CP, model_UT_IP, model_SS_CP, model_SS_IP = modelArchitecture(Psi_model)  # build uniaxial and shear models

        with open(Model_summary, 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))  # summarize layers in architecture

        # %%  Model training
        model_given, input_train, output_train, sample_weights = traindata(
            modelFit_mode)  # model type, input/output pairs

        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        path_checkpoint = path2saveResults_check + '/best_weights'
        if train:  # use compile/fit parameters to train specific model (UT, SS, both) with specific input/output pairs
            model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint,
                                                   sample_weights)

            model_given.load_weights(path_checkpoint, by_name=False,
                                     skip_mismatch=False)  # load the weights saved in the path (the best ones)
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)  # save the model
            Psi_model.save_weights(Save_weights, overwrite=True)  # save the weights

            # Plot loss function
            loss_history = history.history['loss']
            fig, axe = plt.subplots(figsize=[6, 5])  # inches
            plotLoss(axe, loss_history)
            plt.savefig(path2saveResults + '/Plot_loss_' + Sample + '_' + modelFit_mode + '.pdf')
            # plt.show()
            plt.close()

        else:  # if already trained, just load the saved weights
            Psi_model.load_weights(Save_weights, by_name=False, skip_mismatch=False)

        # Get CANN model response
        Stress_predict_UT_CP = model_UT_CP.predict(lam_ut_CP, verbose=0)
        Stress_predict_UT_IP = model_UT_IP.predict(lam_ut_IP, verbose=0)
        Stress_predict_SS_CP = model_SS_CP.predict(gamma_ss_CP, verbose=0)
        Stress_predict_SS_IP = model_SS_IP.predict(gamma_ss_IP, verbose=0)

        # Show weights (remember: weights are output in the order they are built)
        weight_matrix = np.empty((terms, 2))
        for i in range(terms):
            value = Psi_model.get_weights()[i][0][0]
            weight_matrix[i, 0] = value  # inner layer is first column
            weight_matrix[:, 1] = Psi_model.get_layer('wx2').get_weights()[0].flatten()  # outer layer is second column
        print("weight_matrix")
        print(weight_matrix)

        # Get the trained weights
        model_weights_0 = Psi_model.get_weights()

        # Plot the contributions of each term to the output of the model
        fig, ax = plt.subplots(figsize=(12.5, 8.33))
        plotMapTen_CP(ax, Psi_model, model_weights_0, model_UT_CP, terms, lam_ut_CP[20:], P_ut_CP[20:], Sample,
                   path2saveResults, modelFit_mode, model_type, numTerms)
        
        fig2, ax2 = plt.subplots(figsize=(12.5, 8.33))
        plotMapCom_CP(ax2, Psi_model, model_weights_0, model_UT_CP, terms, lam_ut_CP[:21], P_ut_CP[:21], Sample,
                   path2saveResults, modelFit_mode, model_type, numTerms)
        
        fig3, ax3 = plt.subplots(figsize=(12.5, 8.33))
        plotMapShear_CP(ax3, Psi_model, model_weights_0, model_SS_CP, terms, gamma_ss_CP[20:], P_ss_CP[20:], Sample,
                     path2saveResults, modelFit_mode, model_type, numTerms)

        fig4, ax4 = plt.subplots(figsize=(12.5, 8.33))
        plotMapTen_IP(ax4, Psi_model, model_weights_0, model_UT_IP, terms, lam_ut_IP[20:], P_ut_IP[20:], Sample,
                      path2saveResults, modelFit_mode, model_type, numTerms)

        fig5, ax5 = plt.subplots(figsize=(12.5, 8.33))
        plotMapCom_IP(ax5, Psi_model, model_weights_0, model_UT_IP, terms, lam_ut_IP[:21], P_ut_IP[:21], Sample,
                      path2saveResults, modelFit_mode, model_type, numTerms)

        fig6, ax6 = plt.subplots(figsize=(12.5, 8.33))
        plotMapShear_IP(ax6, Psi_model, model_weights_0, model_SS_IP, terms, gamma_ss_IP[20:], P_ss_IP[20:], Sample,
                        path2saveResults, modelFit_mode, model_type, numTerms)
        
        R2_t_CP = r2_score(P_ut_CP[20:], Stress_predict_UT_CP[20:])
        R2_c_CP = r2_score(P_ut_CP[:21], Stress_predict_UT_CP[:21])
        R2ss_CP = r2_score(P_ss_CP, Stress_predict_SS_CP)
        R2_t_IP = r2_score(P_ut_IP[20:], Stress_predict_UT_IP[20:])
        R2_c_IP = r2_score(P_ut_IP[:21], Stress_predict_UT_IP[:21])
        R2ss_IP = r2_score(P_ss_IP, Stress_predict_SS_IP)
        print('R2 tension CP = ', R2_t_CP)
        print('R2 compression CP = ', R2_c_CP)
        print('R2 shear CP = ', R2ss_CP)
        print('R2 tension IP = ', R2_t_IP)
        print('R2 compression IP = ', R2_c_IP)
        print('R2 shear IP = ', R2ss_IP)


        # Save trained weights and R2 values to txt file
        Config = {"Sample": Sample, "modelFit_mode": modelFit_mode, 'model_type': model_type, 'Reg': reg,
                  'Penalty': pen, "R2_t_CP": R2_t_CP, "R2_c_CP": R2_c_CP, "R2_ss_CP": R2ss_CP,
                  "R2_t_IP": R2_t_IP, "R2_c_IP": R2_c_IP, "R2_ss_IP": R2ss_IP,
                  "weights": weight_matrix.tolist()}
        json.dump(Config, open(path2saveResults + "/Config_file.txt", 'w'))
