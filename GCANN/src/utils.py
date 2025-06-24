import numpy as np
import pandas as pd
import tensorflow as tf
# Format number as 2 significant figures but not in scientific notation
def format_2sigfigs(x):
    return f"{x:.0f}" if x >= 100 else f"{x:.2g}"



# Load stress data from files
def load_data(n_samples=5):
    # List of length 2 * Ns where Ns is the number of samples
    # List file and sheet names where data is stored, first 5 sheets are all 5 samples in the 0/90 orientation, next 5 sheets are the 5 samples in the +/-45 orientation
    file_names = ["../inputs/sample_stresses_90.xlsx"] * n_samples +  ["../inputs/sample_stresses_45.xlsx"] * n_samples
    sheetnames = [f"Sheet{i+1}" for i in range(n_samples)] * 2
    # Load from each sheet which should have dimensions 500 x 4
    # 5 experiments * 100 datapoints per experiment, 4 columns are x stretch, y stretch, x stress, y stress
    # Reshape to be 2 x (Ns*500) x 4
    loading_data_all = np.array([pd.read_excel(file_names[i], sheet_name=sheetnames[i], engine='openpyxl').to_numpy() for i in
                       range(len(file_names))]).reshape(2, -1, 4)
    # Seperate into stretch and stress (both 2 x (Ns*500) x 2)
    stretches = loading_data_all[:, :, 0:2] # 2 x Ns*5*100 x 2
    stresses_90 = loading_data_all[0, :, 2:]
    stresses_45 = loading_data_all[1, :, 2:]
    stresses_45_reshaped = stresses_45.reshape((n_samples, 5, -1, 2))
    stresses_45_mirrored = stresses_45_reshaped[:, ::-1, :, ::-1]
    stresses_45 = ((stresses_45_reshaped + stresses_45_mirrored) / 2).reshape((-1, 2))
    stresses = np.stack((stresses_90, stresses_45), axis=0)
    # stress_split = np.array_split(stresses[0, :, 0], 25)
    # stresses[0, :, 0] = np.concatenate([stress_split[i] * (10.0 if i%5 == 2 else 1.0) for i in range(25)])
    return stretches, stresses, n_samples

# Stack along new axis in tf
def tf_stack(array, axis=-1):
    return tf.keras.layers.concatenate([tf.expand_dims(x, axis=axis) for x in array], axis=axis)

# Concatenate model type and alpha (with decimal place replaced with p) to get model id (e.g. "independent0p1")
def get_model_id(model_type, alpha):
    alpha_str =  f"{alpha:f}".replace(".", "p").rstrip("0")
    if alpha_str.endswith("p"):
        alpha_str += "0"
    return model_type + alpha_str

# Utility function to flatten python list of lists
def flatten(arr):
    return [x for y in arr for x in y]