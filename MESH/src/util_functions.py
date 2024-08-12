# File containing various utility functions

from sklearn.metrics import r2_score
import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
import shutil
import pandas as pd
import pickle
# Dummy model that returns zeros for all stress values
class DummyModel:
    def predict(self, input):
        return np.zeros_like(input)

def reshape_input_output_mesh(array):
    """
    Reshape stretch or stress data from mesh such that it is a 2 x 5 x 2 list of numpy arrays (2 mounting directions, 5 experiments, 2 for x and y stretch / stress)
    :param array: 2 x 2 list of numpy arrays (first index correesponds to
    :return: 2 x 5 x 2 list of numpy arrays
    """
    array = [[np.array_split(x, 5) for x in y] for y in array]
    array = [[[array[i][j][k] for j in range(len(array[0]))] for k in
                         range(len(array[0][0]))] for i in range(len(array))]
    return array

def resample(arr, num_points):
    """
    Resample array at more or fewer points than initially given, assuming they are equally spaced
    :param arr: 1 dimensional array of length M
    :param num_points: number of points to sample at
    :return: 1 dimensional array of length num_points
    """
    return np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, arr.shape[0]), arr)
def get_metrics(modelFit_mode, model, tension_loss, n_ten, comp_loss, n_comp, shear_loss, n_shear):
    """
    Compute test and train loss and other statistical metrics given individual losses and number of points
    :param modelFit_mode: string describing which data is used for training
    :param model: trained model
    :param tension_loss: sum of squared error from tension data
    :param n_ten: number of data points in tension data
    :param comp_loss: sum of squared error for compression data (0 for biaxial testing)
    :param n_comp: number of data points in compression data (0 for biaxial testing)
    :param shear_loss: sum of squared error for shear data (0 for biaxial testing)
    :param n_shear: number of data points in shear data (0 for biaxial testing)
    :return: list of 4 metrics that represent model complexity and performance: number of parameters, train set MSE,
    test set MSE, chi squared statistic over training data
    """
    train_loss = 0
    test_loss = 0
    n_samples_train = 0
    n_samples_test = 0
    if "T" in modelFit_mode:
        train_loss += tension_loss
        n_samples_train += n_ten
    else:
        test_loss += tension_loss
        n_samples_test += n_ten

    if "C" in modelFit_mode:
        train_loss += comp_loss
        n_samples_train += n_comp
    else:
        test_loss += comp_loss
        n_samples_test += n_comp

    if "SS" in modelFit_mode:
        train_loss += shear_loss
        n_samples_train += n_shear

    else:
        test_loss += shear_loss
        n_samples_test += n_shear

    # Calculate Statistics
    m_params = model.count_params()
    chi_squared = train_loss / (n_samples_train - m_params) if n_samples_train > m_params else -1
    metrics = [m_params, train_loss / n_samples_train, test_loss / n_samples_test if n_samples_test > 0 else -1,
               chi_squared]
    return metrics

def load_df(Region):
    """
    Load data from excel files and return appropriate dataframe
    :param Region: which dataset to use. Options include 'mesh', 'heart', 'BG', 'CX', etc
    :return: Dataframe corresponding to chosen region
    """
    # Import excel file
    file_name = '../input/CANNsBRAINdata.xlsx'
    file_name_fakemeat = '../input/ArtMeat.xlsx'
    file_name_heart = '../input/CANNsHEARTdata.xlsx'
    file_name_mesh = '../input/aligned_fiber.xlsx'
    file_name_mesh_45 = '../input/aligned_45deg.xlsx'

    dfs_brain = pd.read_excel(file_name, sheet_name='Sheet1', engine='openpyxl')
    dfs_fakemeat = pd.read_excel(file_name_fakemeat, sheet_name='Sheet1', engine='openpyxl')
    dfs_heart = pd.read_excel(file_name_heart, sheet_name='Sheet1', engine='openpyxl')
    dfs_mesh = pd.read_excel(file_name_mesh, sheet_name='Sheet1', engine='openpyxl')
    dfs_mesh_45 = pd.read_excel(file_name_mesh_45, sheet_name='Sheet1', engine='openpyxl')

    dfs = dfs_fakemeat if Region == 'AC' else dfs_heart if Region == 'heart' else [dfs_mesh, dfs_mesh_45] if Region == 'mesh' else dfs_brain

    return dfs

## Returns path where all results are saved
def get_base_path():
    return '../Results'


def get_model_summary_path():
    return get_base_path() + '/Model_summary.txt'


## Returns path where data for a certain region and training mode is saved
def get_path(Region, modelFit_mode):
    return os.path.join(get_base_path(), Region, modelFit_mode)


## Returns the same value that is passed as an input
def identity(x):
    return x


## Delete directory specified by path if it exists
def delDIR(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


## Create folder specified by path if it doesn't already exist
def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)


## Flatten list, analogous to np.flatten for numpy array
def flatten(l):
    return [item for sublist in l for item in sublist]


# Get r squared for data set but force it to be nonnegative
def r2_score_nonnegative(Truth, Prediction):
    R2 = r2_score(Truth, Prediction)
    return max(R2, 0.0)


# Take gradient of a with respect to b
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]


def getStressStrain(dfs, Region):
    """
    Get relevant stress and strain for a specified tissue type by extracting the relevant entries from the dataframe dfs
    :param dfs: Dataframe corresponding to current region (output of load_df)
    :param Region: String representing tissue type to use for training (i.e. mesh, CX, BG)
    :return:
    """
    if Region == 'mesh':
        # Assume we have 5 biaxial tests
        # biaxial test 1:1
        P_ut = [[dfs[i].iloc[:, j + 2].dropna().astype(np.float64).values for j in range(2)] for i in range(len(dfs))]
        lam_ut = [[dfs[i].iloc[:, j].dropna().astype(np.float64).values for j in range(2)] for i in range(len(dfs))]
        midpoint = 0
        P_ss = []
        gamma_ss = []

    elif Region == 'CX':
        P_ut = dfs.iloc[3:, 1].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[3:, 0].dropna().astype(np.float64).values

        P_ss = dfs.iloc[3:, 3].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[3:, 2].dropna().astype(np.float64).values
    elif Region == 'CR':
        P_ut = dfs.iloc[3:, 6].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[3:, 5].dropna().astype(np.float64).values

        P_ss = dfs.iloc[3:, 8].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[3:, 7].dropna().astype(np.float64).values
    elif Region == 'BG':
        P_ut = dfs.iloc[3:, 11].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[3:, 10].dropna().astype(np.float64).values

        P_ss = dfs.iloc[3:, 13].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[3:, 12].dropna().astype(np.float64).values
    elif Region == 'CC':
        P_ut = dfs.iloc[3:, 16].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[3:, 15].dropna().astype(np.float64).values

        P_ss = dfs.iloc[3:, 18].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[3:, 17].dropna().astype(np.float64).values
    elif Region == 'AC':  # Artificial Chicken
        P_ut = dfs.iloc[:, 1].dropna().astype(np.float64).values
        lam_ut = dfs.iloc[:, 0].dropna().astype(np.float64).values

        P_ss = dfs.iloc[:, 4].dropna().astype(np.float64).values
        gamma_ss = dfs.iloc[:, 3].dropna().astype(np.float64).values
    else:
        print("Invalid Region:", Region)
        assert False
    P_ut_all = P_ut
    lam_ut_all = lam_ut

    if Region != 'mesh':
        midpoint = int((lam_ut.shape[0] - 1) / 2)

    return P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss, midpoint


# Get relevant training data given model fitting mode and whether you are fitting the strain energy or the stress
def traindata(modelFit_mode, model_UT, lam_ut, P_ut, model_SS, gamma_ss, P_ss, model, midpoint):
    if modelFit_mode == 'T':
        model_given = model_UT
        input_train = lam_ut[midpoint:]
        output_train = P_ut[midpoint:]
        sample_weights = np.array([1.0] * input_train.shape[0])

    elif modelFit_mode == "C":
        model_given = model_UT
        input_train = lam_ut[:(midpoint + 1)]
        output_train = P_ut[:(midpoint + 1)]
        sample_weights = np.array([1.0] * input_train.shape[0])

    elif modelFit_mode == "TC":
        model_given = model_UT
        input_train = lam_ut
        output_train = P_ut

        if type(output_train) is list:
            sample_weights = [[np.concatenate([np.ones_like(z) / np.max(z) for z in np.array_split(y, 5)]) for y in x] for x in output_train]
            # if type(output_train[0]) is list:
            #     sample_weights = [[np.array([1.0 / np.max(np.abs(x[0]))] * x[0].shape[0])] for x in output_train]
            # else:
            #     sample_weights = np.array([1.0] * input_train[0].shape[0])
        else:
            sample_weights = np.array([1.0] * input_train.shape[0])
    elif modelFit_mode == "TSS":
        model_given = model
        input_train = [[lam_ut[midpoint:]], [gamma_ss[midpoint:]]]
        output_train = [[P_ut[midpoint:]], [P_ss[midpoint:]]]
        sample_weights = [[np.array([1.0 / np.max(x)] * x[0].shape[0])] for x in output_train]

    elif modelFit_mode == "CSS":
        model_given = model
        input_train = [[lam_ut[:(midpoint + 1)]], [gamma_ss[midpoint:]]]
        output_train = [[P_ut[:(midpoint + 1)]], [P_ss[midpoint:]]]
        sample_weights = [[np.array([1.0 / np.max(np.abs(x))] * x[0].shape[0])] for x in output_train]

    elif modelFit_mode == "SS":
        model_given = model_SS
        input_train = gamma_ss  # symmetric around gamma=0
        output_train = P_ss
        if type(output_train) is list:
            sample_weights = [[np.array([1.0 / np.max(np.abs(x[0]))] * x[0].shape[0])] for x in output_train]
        else:
            sample_weights = np.array([1.0] * input_train.shape[0])

    elif modelFit_mode == "TC_and_SS":
        model_given = model
        if type(lam_ut) is list:
            input_train = lam_ut + gamma_ss
            output_train = P_ut + P_ss
        else:
            input_train = [[lam_ut], [gamma_ss]]
            output_train = [[P_ut], [P_ss]]
        sample_weights = [[np.array([1.0 / np.max(np.abs(x[0]))] * x[0].shape[0])] for x in output_train]


    elif modelFit_mode.isnumeric():
        model_given = model
        input_train = lam_ut + gamma_ss
        output_train = P_ut + P_ss
        # Assume we are using mesh
        modes = [int(x) for x in modelFit_mode]

        sample_weights = [[np.concatenate([np.ones_like(np.array_split(y, 5)[j]) *
                                           ((1 / np.max(np.array_split(y, 5)[j])) ** 2 if (5 * i + j) in modes else 0.0) for j in range(5)])
                           for y in output_train[i]] for i in range(len(output_train))]
        sample_weights = [sample_weights[0], [x / 2.0 for x in sample_weights[1]]] # This weights +/- 45 less since there are half as many distinct experiments
    else:
        print("Invalid Model Fit Mode:", modelFit_mode)
        assert False


    return model_given, input_train, output_train, sample_weights


# Perform training of model, return fit model, training history, and weight history
def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, batch_size):

    mse_loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]
    opti1 = tf.optimizers.Adam(learning_rate=0.001)

    model_given.compile(loss=mse_loss,
                        optimizer=opti1,
                        metrics=metrics)
    # Stop early if loss doesn't decrease for 2000 epochs
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-6, patience=2000,
                                                restore_best_weights=True)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="loss",
        filepath=path_checkpoint,
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
    )
    # Create array to store weight history
    weight_hist_arr = []
    # Create callback to append model weights to weight_hist_arr every epoch
    weight_hist_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs:
        weight_hist_arr.append(model_given.get_weights()))

    history = model_given.fit(input_train,
                              output_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              callbacks=[es_callback, modelckpt_callback, weight_hist_callback],
                              shuffle=True,
                              verbose=0,
                              sample_weight=sample_weights)

    return model_given, history, weight_hist_arr

def export_graphs():
    output_folder = os.path.join(get_base_path(), "figures")
    makeDIR(output_folder)
    Region = "mesh"
    modelFit_mode = "0123456789"
    base_path = get_path(Region, modelFit_mode)

    shutil.copy(os.path.join(base_path, "cann_I4ws", "l0_map.pdf"), os.path.join(output_folder, "l0_map_ws.pdf"))
    shutil.copy(os.path.join(base_path, "cann_I4w_theta", "l0_map.pdf"), os.path.join(output_folder, "l0_map_theta.pdf"))
    shutil.copy(os.path.join(base_path, "cann_I4ws", "0", "training_2.pdf"), os.path.join(output_folder, "training_ws.pdf"))
    shutil.copy(os.path.join(base_path, "cann_I4w_theta", "0", "training_2.pdf"), os.path.join(output_folder, "training_theta.pdf"))
    shutil.copy(os.path.join(base_path, "cann_I4ws_noiso", "0", "training_2.pdf"), os.path.join(output_folder, "training_ws_noiso.pdf"))
    shutil.copy(os.path.join(base_path, "cann_I4w_theta_noiso", "0", "training_2.pdf"), os.path.join(output_folder, "training_theta_noiso.pdf"))

    shutil.copy(os.path.join(base_path, "cann_I4ws_noiso", "0", "training_1.pdf"), os.path.join(output_folder, "training_ws_noiso_noreg.pdf"))
    shutil.copy(os.path.join(base_path, "cann_I4w_theta_noiso", "0", "training_1.pdf"), os.path.join(output_folder, "training_theta_noiso_noreg.pdf"))
    shutil.copy(os.path.join(get_path(Region, "0123456789"), "cann_I4w_theta", "0", "r2_bestfit.pdf"), os.path.join(output_folder, "r2_bestfit.pdf"))
    shutil.copy(os.path.join(get_path(Region, "0123456789"), "cann_I4ws", "0", "r2_archcomp.pdf"), os.path.join(output_folder, "r2_archcomp.pdf"))

