#!/usr/bin/env python
# coding: utf-8



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

from models_brain import*
from plotting_brain import*

from sklearn.metrics import r2_score


convexity_weight = 20
#%% Uts
filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()

def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)



def flatten(l):
    return [item for sublist in l for item in sublist]
    
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)


#%% Functions


def getStressStrain(Region):

    if Region =='leap':
        lam_ut = dfs.iloc[:,0].astype(np.float64)
        P_ut = dfs.iloc[:,1].astype(np.float64)
        P_ut_std = dfs.iloc[:,2].astype(np.float64)
        gamma_ss = dfs.iloc[:,3].astype(np.float64).values
        P_ss = dfs.iloc[:,4].astype(np.float64).values
        P_ss_std = dfs.iloc[:,5].astype(np.float64)
    elif Region =='turbo':
        lam_ut = dfs.iloc[:,6].astype(np.float64)
        P_ut = dfs.iloc[:,7].astype(np.float64)
        P_ut_std = dfs.iloc[:,8].astype(np.float64)
        gamma_ss = dfs.iloc[:,9].astype(np.float64).values
        P_ss = dfs.iloc[:,10].astype(np.float64).values
        P_ss_std = dfs.iloc[:,11].astype(np.float64)

    return lam_ut, P_ut, P_ut_std, gamma_ss, P_ss, P_ss_std


def calculate_relative_weights():
    """Calculate relative weights based on maximum stress in each mode"""
    midpoint = int(len(lam_ut) / 2)
    
    # Calculate maximum stress for each mode
    max_tension = np.max(np.abs(P_ut[midpoint:]))
    max_compression = np.max(np.abs(P_ut[:(midpoint + 1)]))
    max_shear = np.max(np.abs(P_ss))
    
    # Calculate relative weights (inverse of maximum stress to balance training)
    # Higher stress modes get lower weights to prevent dominance
    weight_tension = 1.0 / max_tension ** 2
    weight_compression = 1.0 / max_compression ** 2
    weight_shear = 1.0 / max_shear ** 2
    
    # Normalize weights so they sum to 3 (one for each mode)
    total_weight = weight_tension + weight_compression + weight_shear
    weight_tension = weight_tension / total_weight * 3.0
    weight_compression = weight_compression / total_weight * 3.0
    weight_shear = weight_shear / total_weight * 3.0
    
    
    return weight_tension, weight_compression, weight_shear

def make_sample_weights(weight_ax, weight_trans, weight_shr, inp_len):
    return [np.array([weight_ax]*inp_len), 
        np.array([weight_trans]*inp_len), 
        np.array([weight_shr]*inp_len), 
        np.array([0.0]*inp_len)]

def make_sample_weights_std(weight_ax, weight_trans, weight_shr, inp_len):
    return [weight_ax, 
        np.array([weight_trans]*inp_len), 
        weight_shr, 
        np.array([0.0]*inp_len)]

def display_strain_energy_expression(Psi_model, terms):
    """Display the full strain energy expression as a function of I1, I2, J"""
    
    print("\n" + "="*80)
    print("STRAIN ENERGY EXPRESSION")
    print("="*80)
    
    # Get the weights from the model
    weights = Psi_model.get_weights()
    
    # Extract the final layer weights (these multiply each term)
    final_weights = weights[-1].flatten()
    
    # Define the reduced invariants
    print("Psi(I1_bar, I2_bar, J) where:")
    print("  I1_bar = I1/J^(2/3) - 3")
    print("  I2_bar = I2/J^(4/3) - 3")
    print("  J = J")
    print()
    
    # Build the expression
    expression_terms = []
    term_idx = 0
    
    # I1 terms (4 terms per SingleInvNet)
    print("I1_bar terms:")
    for i in range(4):
        if term_idx < len(final_weights):
            w_outer = final_weights[term_idx]
            if abs(w_outer) > 1e-6:  # Only show significant terms
                # Get the inner weight (coefficient inside the activation function)
                w_inner = weights[i][0][0] if i < len(weights) else 1.0
                
                if i == 0:
                    expr = f"{w_outer * w_inner:.6f} (\\bar I_1 - 3)"
                elif i == 1:
                    expr = f"{w_outer:.6f} (exp({w_inner:.6f} (\\bar I_1 - 3)) - 1)"
                elif i == 2:
                    expr = f"{w_outer * w_inner ** 2:.6f} (\\bar I_1 - 3)^2"
                elif i == 3:
                    expr = f"{w_outer:.6f} (exp({w_inner:.6f} (\\bar I_1 - 3)^2) - 1)"
                expression_terms.append(expr)
            term_idx += 1
    
    # I2 terms (4 terms per SingleInvNet)
    print("\nI2_bar terms:")
    for i in range(4):
        if term_idx < len(final_weights):
            w_outer = final_weights[term_idx]
            if abs(w_outer) > 1e-6:  # Only show significant terms
                # Get the inner weight (coefficient inside the activation function)
                w_inner = weights[i + 4][0][0] if (i + 4) < len(weights) else 1.0
                
                if i == 0:
                    expr = f"{w_outer*w_inner:.6f} (\\bar I_2 - 3)"
                elif i == 1:
                    expr = f"{w_outer:.6f} (exp({w_inner:.6f} (\\bar I_2 - 3)) - 1)"
                elif i == 2:
                    expr = f"{w_outer * w_inner **2:.6f} (\\bar I_2 - 3)^2"
                elif i == 3:
                    expr = f"{w_outer:.6f} (exp({w_inner:.6f} (\\bar I_2 - 3)^2) - 1)"
                expression_terms.append(expr)
            term_idx += 1
    
    # J terms (3 terms from BulkNet)
    print("\nJ terms:")
    for i in range(3):
        if term_idx < len(final_weights):
            w_outer = final_weights[term_idx]
            if abs(w_outer) > 1e-6:  # Only show significant terms
                # Get the inner weight (coefficient inside the activation function)
                w_inner = weights[i + 8][0][0] if (i + 8) < len(weights) else 1.0
                
                if i == 0:
                    expr = f"{w_outer:.6f} (J^{w_inner:.6f} - {w_inner:.6f} log(J) - 1)"
                elif i == 1:
                    expr = f"{w_outer*w_inner:.6f} log(J)^2"
                elif i == 2:
                    expr = f"{w_outer:.6f} (exp({w_inner:.6f} log(J)^2) - 1)"
                expression_terms.append(expr)
            term_idx += 1

    # J terms (3 terms from BulkNet)
    print("\nMixed terms:")
    for i in range(3):
        if term_idx < len(final_weights):
            w_outer = final_weights[term_idx]
            if abs(w_outer) > 1e-6:  # Only show significant terms
                # Get the inner weight (coefficient inside the activation function)
                w_inner = weights[term_idx][0][0] if term_idx < len(weights) else 1.0
                
                if i == 0:
                    expr = f"{w_outer:.6f} * J^({w_inner:.6f}) (\\bar I_1 - 3) "
                elif i == 1:
                    expr = f"{w_outer:.6f} * J^({w_inner:.6f}) (\\bar I_2 - 3)"
                expression_terms.append(expr)
            term_idx += 1
    # Display the full expression
    print(f"\nFull Expression:")
    if expression_terms:
        full_expr = " + ".join(expression_terms)
        print(f"\\Psi(\\bar I_1, \\bar I_2, J) = {full_expr}")
    else:
        print("All weights are negligible")
    
    print("="*80)
    print()
    
    return expression_terms


def traindata(modelFit_mode, zero_trans_tension=False, weight_std=False):
    # The model_given should be the stress-output model, but we need to ensure it has trainable weights
    model_given = model
    midpoint = int(len(lam_ut) / 2)
    
    weight_tension, weight_compression, weight_shear = calculate_relative_weights()
    if modelFit_mode == 'T':
        stretch_in = lam_ut[midpoint:]
        inp_len = stretch_in.shape[0]
        shear_in = np.array([0.0] * inp_len)
        input_train = [stretch_in, shear_in]
        stress_axial = P_ut[midpoint:]
        stress_trans = np.array([0.0] * inp_len)
        stress_shear = np.array([0.0] * inp_len)
        psi_output = np.array([0.0] * inp_len)
        output_train = [stress_axial, stress_trans, stress_shear, psi_output]
        weight_tension_std = P_ut_std[midpoint:]/(P_ut_std[midpoint:] + eps)**3
        sample_weights = make_sample_weights_std(weight_tension_std, np.min(weight_tension_std), np.array([0.0]*inp_len), inp_len) if weight_std else make_sample_weights(weight_tension, weight_tension, 0, inp_len)
    elif modelFit_mode == "C":
        stretch_in = lam_ut[:(midpoint + 1)]
        inp_len = stretch_in.shape[0]
        shear_in = np.array([0.0] * inp_len)
        input_train = [stretch_in, shear_in]
        stress_axial = P_ut[:(midpoint + 1)]
        stress_trans = np.array([0.0] * inp_len)
        stress_shear = np.array([0.0] * inp_len)
        psi_output = np.array([0.0] * inp_len)
        output_train = [stress_axial, stress_trans, stress_shear, psi_output]
        weight_compression_std = P_ut_std[:(midpoint + 1)]/(P_ut_std[:(midpoint + 1)] + eps)**3
        sample_weights = make_sample_weights_std(weight_compression_std, np.min(weight_compression_std), np.array([0.0]*inp_len), inp_len) if weight_std else make_sample_weights(weight_compression, weight_compression, 0, inp_len)
        
    elif modelFit_mode == "TC":
        stretch_in = lam_ut
        inp_len = stretch_in.shape[0]
        shear_in = np.array([0.0] * inp_len)
        input_train = [stretch_in, shear_in]
        stress_axial = P_ut
        stress_trans = np.array([0.0] * inp_len)
        stress_shear = np.array([0.0] * inp_len)
        psi_output = np.array([0.0] * inp_len)
        output_train = [stress_axial, stress_trans, stress_shear, psi_output]
        # For TC mode, use compression weight for first half and tension weight for second half
        weight_tc_axial = np.concatenate([
            np.array([weight_compression] * (midpoint + 1)),  # compression data
            np.array([weight_tension] * (len(lam_ut) - midpoint - 1))  # tension data
        ])
        weight_tc_trans = np.concatenate([
            np.array([weight_compression] * (midpoint + 1)),  # compression data
            np.array([weight_tension if zero_trans_tension else 0.0] * (len(lam_ut) - midpoint - 1))  # tension data
        ])
        weight_tc_axial_std = P_ut_std/(P_ut_std + eps)**3
        weight_tc_trans_std = np.concatenate([
            np.array([np.mean(weight_tc_axial_std)] * (midpoint + 1)),  # compression data
            np.array([np.mean(weight_tc_axial_std) if zero_trans_tension else 0.0] * (len(lam_ut) - midpoint - 1))  # tension data
        ])
        sample_weights = (([weight_tc_axial_std, weight_tc_trans_std] if weight_std else [weight_tc_axial, weight_tc_trans]) + 
            [np.array([0.0]* inp_len)] + [np.array([convexity_weight]*inp_len)])
    elif modelFit_mode == "SS":
        shear_in = gamma_ss
        inp_len = shear_in.shape[0]
        stretch_in = np.array([0.8] * inp_len)
        input_train = [stretch_in, shear_in]
        stress_shear = P_ss
        stress_axial = np.array([0.0] * inp_len)
        stress_trans = np.array([0.0] * inp_len)
        psi_output = np.array([0.0] * inp_len)
        output_train = [stress_axial, stress_trans, stress_shear, psi_output]
        weight_ss_shear_std = 1 / (P_ss_std + eps)**2
        sample_weights = make_sample_weights_std(np.array([0.0]*inp_len), 0.0, weight_ss_shear_std, inp_len) if weight_std else make_sample_weights(0, 0, weight_shear, inp_len)
        
    elif modelFit_mode == "TC_and_SS":
        _, input_train_1, output_train_1, sample_weights_1 = traindata("TC", zero_trans_tension=zero_trans_tension, weight_std=weight_std)
        _, input_train_2, output_train_2, sample_weights_2 = traindata("SS", zero_trans_tension=zero_trans_tension, weight_std=weight_std)
        input_train = [np.concatenate([x, y], axis=0) for (x, y) in zip(input_train_1, input_train_2)]
        output_train = [np.concatenate([x, y], axis=0) for (x, y) in zip(output_train_1, output_train_2)]
        sample_weights = [np.concatenate([x, y], axis=0) for (x, y) in zip(sample_weights_1, sample_weights_2)]

    
    return model_given, input_train, output_train, sample_weights

        

def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights):
    
    # Use MSE for outputs 1-3 (Stress_axial, Stress_trans, Stress_shear)
    # Use MAE (absolute value) for output 4 (error)
    # Note: sample_weights is a list of 4 arrays (one per output)
    # Keras automatically applies each weight array to the corresponding output's loss:
    # total_loss = sample_weights[0] * MSE(output_0) + 
    #              sample_weights[1] * MSE(output_1) + 
    #              sample_weights[2] * MSE(output_2) + 
    #              sample_weights[3] * MAE(output_3)
    mse_loss = keras.losses.MeanSquaredError()
    mae_loss = keras.losses.MeanAbsoluteError()
    loss_list = [mse_loss, mse_loss, mse_loss, mae_loss]
    
    metrics  =[keras.metrics.MeanSquaredError()] * 3 + [keras.metrics.MeanAbsoluteError()]
    opti1    = tf.optimizers.Adam(learning_rate=0.001)
    
    model_given.compile(loss=loss_list,
                  optimizer=opti1,
                  metrics=metrics)
    
    
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=3000, restore_best_weights=True)

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
                        shuffle = True,
                        verbose = 0, 
                        sample_weight = sample_weights)
    
    return model_given, history



def Stress_calc_axial(inputs):
    (dWdI1, dWdI2, dWdJ, Stretch, Gamma) = inputs
    return 2 * Stretch * dWdI1 + 4 * Stretch * dWdI2 + dWdJ

eps = 1e-9
def Stress_calc_axial_principalStretch(inputs):
    (dWdl1, dWdl2, dWdl3, Stretch, Gamma) = inputs
    # dl1dStretch = Stretch * (1 + (Stretch ** 2 + Gamma ** 2 - 1) / (2 * ((1 + Stretch ** 2 + Gamma ** 2)**2 / 4 - Stretch ** 2 + eps) ** 0.5))
    # dl2dStretch = Stretch * (1 - (Stretch ** 2 + Gamma ** 2 - 1) / (2 * ((1 + Stretch ** 2 + Gamma ** 2)**2 / 4 - Stretch ** 2 + eps) ** 0.5))
    # return dWdl1 * dl1dStretch + dWdl2 * dl2dStretch
    return Stretch * (dWdl1 + dWdl2)

def Stress_calc_shear_principalStretch(inputs):
    (dWdl1, dWdl2, dWdl3, Stretch, Gamma) = inputs
    dl1dGamma = Gamma * (1 + (Stretch ** 2 + Gamma ** 2 + 1) / (2 * ((1 + Stretch ** 2 + Gamma ** 2)**2 / 4 - Stretch ** 2 + eps) ** 0.5))
    dl2dGamma = Gamma * (1 - (Stretch ** 2 + Gamma ** 2 + 1) / (2 * ((1 + Stretch ** 2 + Gamma ** 2)**2 / 4 - Stretch ** 2 + eps) ** 0.5))
    return dWdl1 * dl1dGamma + dWdl2 * dl2dGamma

def Stress_calc_shear(inputs):
    (dWdI1, dWdI2, dWdJ, Stretch, Gamma) = inputs
    return 2 * Gamma * dWdI1 + 2 * Gamma * dWdI2

def Stress_calc_trans(inputs):
    (dWdI1, dWdI2, dWdJ, Stretch, Gamma) = inputs
    return 2 * dWdI1 + 2 * (1 + Stretch ** 2) * dWdI2 + Stretch * dWdJ

def Stress_calc_trans_biaxial(inputs):
    (dWdI1, dWdI2, dWdJ, StretchAxial, StretchTrans) = inputs
    return 2 * StretchTrans * dWdI1 + 2 * StretchTrans * (StretchTrans ** 2 + StretchAxial ** 2) * dWdI2 + StretchAxial * StretchTrans * dWdJ

def Convexity_calc_biaxial(inputs):
    (W1, W2, WJJ, W11, W22, W1J, StretchAxial, StretchTrans) = inputs
    ## I1 = a^2 + b^2 + c^2
    ## I2 = a^2b^2 + b^2c^2 + a^2c^2
    ## J = abc
    ## I1_b = 2b
    ## I1_bb = 2
    ## I2_b = 2b(a^2 + c^2) = 2b(a^2 + b^2)
    ## I2_bb = 2(a^2 + c^2) = 2(a^2 + b^2)
    ## J_b = ac = ab
    ## J_bb = 0
    ## W_bb = W_I1 * I1_bb + W_I1I1 * (I1_b)^2 
    #       + W_I2 * I2_bb + W_I2I2 * (I2_b)^2
    ##      + W_JJ * J_b^2 + 2W_JI1 * I1_b J_b
    I1p = 2 * StretchTrans
    I1pp = 2
    I2p = 2 * StretchTrans * (StretchTrans ** 2 + StretchAxial ** 2)
    I2pp = 2 * (StretchTrans ** 2 + StretchAxial ** 2)
    Jp = StretchTrans * StretchAxial 

    return W1 * I1pp + W11 * I1p ** 2 + W2 * I2pp + W22 * I2p ** 2 + WJJ * Jp ** 2 + 2 * W1J * I1p * Jp
class PsiModelWrapper(keras.layers.Layer):
    """
    Custom layer that wraps Psi_model to ensure its trainable weights are properly tracked
    by the parent model. This is necessary because using a model inside Lambda layers
    doesn't always result in proper variable tracking.
    
    When a Keras Model is called inside a Layer's call() method (not a Lambda closure),
    TensorFlow automatically tracks its trainable variables as part of the parent model.
    """
    def __init__(self, psi_model, **kwargs):
        super(PsiModelWrapper, self).__init__(**kwargs)
        self.psi_model = psi_model
        # Store the model so it can be called in call()
        # When the model is called inside this layer's call method, TensorFlow will
        # automatically track its trainable variables as part of this layer
    
    def call(self, inputs):
        i1, i2, j = inputs
        # Call the model - TensorFlow will track its variables because it's called
        # within a Layer's call method, not inside a Lambda closure
        # This ensures Psi_model's trainable weights are part of the parent model
        return self.psi_model([i1, i2, j])
    
    def get_config(self):
        config = super(PsiModelWrapper, self).get_config()
        # Note: We can't serialize the model in get_config, but that's okay
        # since we're passing it in __init__
        return config

class ComputeDerivativesLayer(keras.layers.Layer):
    """
    Custom layer that computes derivatives of Psi_model using GradientTape.
    This ensures Psi_model's variables are tracked because it's called inside
    a Layer's call() method, not a Lambda closure.
    """
    def __init__(self, psi_wrapper, **kwargs):
        super(ComputeDerivativesLayer, self).__init__(**kwargs)
        self.psi_wrapper = psi_wrapper
    
    def call(self, inputs):
        i1, i2, j = inputs
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2, j])
            # Use the wrapper - TensorFlow will track its variables
            psi = self.psi_wrapper([i1, i2, j])
        
        dWdI1 = tape.gradient(psi, i1)
        dWdI2 = tape.gradient(psi, i2)
        dWdJ = tape.gradient(psi, j)
        
        # Return as separate outputs that can be unpacked
        return dWdI1, dWdI2, dWdJ

class ComputeConstraintsLayer(keras.layers.Layer):
    """
    Custom layer that computes constraint values using nested GradientTape.
    This ensures Psi_model's variables are tracked.
    """
    def __init__(self, psi_wrapper, **kwargs):
        super(ComputeConstraintsLayer, self).__init__(**kwargs)
        self.psi_wrapper = psi_wrapper
    
    def call(self, j):
        i1_bar = 3.0 + j * 0
        i2_bar = 3.0 + j * 0
        i1_0 = j * 0.0
        i2_0 = j * 0.0


        
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(j)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(j)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch([i1_bar, i2_bar, j])
                    i1 = i1_bar * j ** (2/3)
                    i2 = i2_bar * j ** (4/3)
                    # Use the wrapper - TensorFlow will track its variables
                    psi = self.psi_wrapper([i1, i2, j])
                
                dWdI1 = tape3.gradient(psi, i1_bar)
                dWdI2 = tape3.gradient(psi, i2_bar)
                psi_0 = self.psi_wrapper([i1_0, i2_0, j])
            dWdJ = tape2.gradient(psi_0, j)
            d2WdI1dJ = tape2.gradient(dWdI1, j)
        d3WdI1dJ2 = tape1.gradient(d2WdI1dJ, j)
        dWdJ2 = tape1.gradient(dWdJ, j)

        A = (4/9) * dWdI1 + (8/3) * j * d2WdI1dJ + j ** 2 * d3WdI1dJ2
        B = (10/9) * dWdI1 - (4/3) * j * d2WdI1dJ + j ** 2 * d3WdI1dJ2
        C = j ** 2 * dWdJ2
        output = tf.nn.relu(-B) + tf.nn.relu(-C - 3 * (A * B ** 2) ** (1/3))
        return output

def extract_term_contributions(model, stretch_input, gamma_input):
    """
    Extract individual term contributions to STRESS using weight zeroing approach.
    This function works with both StrainEnergyCANN and StrainEnergyPrincipalStretch architectures.
    """
    # Get the Psi_model from the main model (4th output)
    # Get the full model prediction first (use eager call to avoid predict shape issues)
    full_prediction = model([stretch_input, gamma_input], training=False)
    # Ensure numpy arrays
    full_stress = full_prediction[0].numpy() if hasattr(full_prediction[0], 'numpy') else full_prediction[0]
    
    # Detect which architecture we're using
    # Check if Psi_model has a SinglePrincipalStretch model (Principal Stretch architecture)
    SinglePrincipalStretch_model = None
    for layer in Psi_model.layers:
        if isinstance(layer, keras.models.Model) and layer.name == 'SinglePrincipalStretch':
            SinglePrincipalStretch_model = layer
            break
    
    if SinglePrincipalStretch_model is not None:
        # Principal Stretch architecture
        final_layer = SinglePrincipalStretch_model.layers[-1]
        term_names = ["exp(ln(λ))-ln(λ)-1", "ln(λ)²", "exp(ln(λ)²)-1", "ln(λ)³", "exp(ln(λ)³)-1"]
    else:
        # StrainEnergyCANN architecture - find the final dense layer
        final_layer = None
        for layer in reversed(Psi_model.layers):
            if isinstance(layer, keras.layers.Dense) and layer.name == 'wx2':
                final_layer = layer
                break
        
        if final_layer is None:
            raise ValueError("Could not find final dense layer in StrainEnergyCANN model")
    
    # Check if the layer has weights
    if not final_layer.get_weights():
        raise ValueError("The final layer has no weights")
    
    original_weights = final_layer.get_weights()[0].copy()
    
    # Get the actual number of terms from the weight shape
    num_terms = original_weights.shape[0]

    # Build term names dynamically based on the structure
    # For StrainEnergyCANN: 4 I1 terms, 4 I2 terms, 3 J terms, 2 Mixed terms (total = 13)
    if SinglePrincipalStretch_model is None:
        term_names = []
        if num_terms >= 4:
            term_names.extend(["I1", "exp(I1)-1", "I1²", "exp(I1²)-1"])
        if num_terms >= 8:
            term_names.extend(["I2", "exp(I2)-1", "I2²", "exp(I2²)-1"])
        if num_terms >= 11:
            term_names.extend(["Jᵐ - m ln(J)", "ln(J)²", "exp(ln(J)²)"])
        if num_terms > 11:
            term_names.extend(["I1·J", "I2·J"])
        
        # If there are more or fewer terms than expected, adjust
        if len(term_names) < num_terms:
            for i in range(len(term_names), num_terms):
                term_names.append(f"Term {i+1}")
        elif len(term_names) > num_terms:
            term_names = term_names[:num_terms]
    
    stress_contributions = []
    
    # Extract individual term contributions by zeroing out other weights
    for i in range(num_terms):
        # Create a copy of the weights
        new_weights = np.zeros_like(original_weights)
        
        # Set only the i-th weight to its original value
        if i < original_weights.shape[0]:
            new_weights[i, 0] = original_weights[i, 0]
        
        # Set the new weights
        final_layer.set_weights([new_weights])
        
        # Get prediction with only this term active
        term_prediction = model([stretch_input, gamma_input], training=False)
        term_stress = term_prediction[0].numpy() if hasattr(term_prediction[0], 'numpy') else term_prediction[0]
        # Normalize shape to (n_points,)
        ts = np.array(term_stress)
        ts = np.squeeze(ts)
        n_pts = np.array(stretch_input).shape[0]
        if ts.ndim == 2:
            if ts.shape[0] == n_pts:
                ts = ts[:, 0] if ts.shape[1] == 1 else ts[:, ...]
            elif ts.shape[1] == n_pts:
                ts = ts.T
                ts = ts[:, 0] if ts.ndim == 2 and ts.shape[1] == 1 else ts
        elif ts.ndim == 0:
            ts = np.full((n_pts,), float(ts))
        stress_contributions.append(np.array(ts).reshape(-1))
    
    # Restore original weights
    final_layer.set_weights([original_weights])
    
    return stress_contributions, term_names

def extract_shear_term_contributions(model, Psi_model, stretch_input, gamma_input):
    """
    Extract individual term contributions to SHEAR STRESS using weight zeroing approach.
    This function works with both StrainEnergyCANN and StrainEnergyPrincipalStretch architectures.
    """    
    # Get the full model prediction first (use eager call)
    full_prediction = model([stretch_input, gamma_input], training=False)
    full_stress = full_prediction[2].numpy() if hasattr(full_prediction[2], 'numpy') else full_prediction[2]
    
    # Detect which architecture we're using
    # Check if Psi_model has a SinglePrincipalStretch model (Principal Stretch architecture)
    SinglePrincipalStretch_model = None
    for layer in Psi_model.layers:
        if isinstance(layer, keras.models.Model) and layer.name == 'SinglePrincipalStretch':
            SinglePrincipalStretch_model = layer
            break
    
    if SinglePrincipalStretch_model is not None:
        # Principal Stretch architecture
        final_layer = SinglePrincipalStretch_model.layers[-1]
        term_names = ["exp(ln(λ))-ln(λ)-1", "ln(λ)²", "exp(ln(λ)²)-1", "ln(λ)³", "exp(ln(λ)³)-1"]
    else:
        # StrainEnergyCANN architecture - find the final dense layer
        final_layer = None
        for layer in reversed(Psi_model.layers):
            if isinstance(layer, keras.layers.Dense) and layer.name == 'wx2':
                final_layer = layer
                break
        
        if final_layer is None:
            raise ValueError("Could not find final dense layer in StrainEnergyCANN model")
    
    # Check if the layer has weights
    if not final_layer.get_weights():
        raise ValueError("The final layer has no weights")
    
    original_weights = final_layer.get_weights()[0].copy()
    
    # Get the actual number of terms from the weight shape
    num_terms = original_weights.shape[0]
    
    # Build term names dynamically based on the structure
    # For StrainEnergyCANN: 4 I1 terms, 4 I2 terms, 3 J terms, 2 Mixed terms (total = 13)
    if SinglePrincipalStretch_model is None:
        term_names = []
        if num_terms >= 4:
            term_names.extend(["I1", "exp(I1)-1", "I1²", "exp(I1²)-1"])
        if num_terms >= 8:
            term_names.extend(["I2", "exp(I2)-1", "I2²", "exp(I2²)-1"])
        if num_terms >= 11:
            term_names.extend(["Jᵐ - m ln(J)", "ln(J)²", "exp(ln(J)²)"])
        if num_terms >= 13:
            term_names.extend(["I1·J", "I2·J"])
        
        # If there are more or fewer terms than expected, adjust
        if len(term_names) < num_terms:
            for i in range(len(term_names), num_terms):
                term_names.append(f"Term {i+1}")
        elif len(term_names) > num_terms:
            term_names = term_names[:num_terms]
    
    stress_contributions = []
    
    # Extract individual term contributions by zeroing out other weights
    for i in range(num_terms):
        # Create a copy of the weights
        new_weights = np.zeros_like(original_weights)
        
        # Set only the i-th weight to its original value
        if i < original_weights.shape[0]:
            new_weights[i, 0] = original_weights[i, 0]
        
        # Set the new weights
        final_layer.set_weights([new_weights])
        
        # Get prediction with only this term active
        term_prediction = model([stretch_input, gamma_input], training=False)
        term_stress = term_prediction[2].numpy() if hasattr(term_prediction[2], 'numpy') else term_prediction[2]
        ts = np.array(term_stress)
        ts = np.squeeze(ts)
        n_pts = np.array(stretch_input).shape[0]
        if ts.ndim == 2:
            if ts.shape[0] == n_pts:
                ts = ts[:, 0] if ts.shape[1] == 1 else ts[:, ...]
            elif ts.shape[1] == n_pts:
                ts = ts.T
                ts = ts[:, 0] if ts.ndim == 2 and ts.shape[1] == 1 else ts
        elif ts.ndim == 0:
            ts = np.full((n_pts,), float(ts))
        stress_contributions.append(np.array(ts).reshape(-1))
    
    # Restore original weights
    final_layer.set_weights([original_weights])
    
    return stress_contributions, term_names

def modelArchitectureBiaxial(Psi_model):
    # Stretch and Gamma as input
    StretchAxial = keras.layers.Input(shape = (1,), name = 'StretchAxial')
    StretchTrans = keras.layers.Input(shape = (1,), name = 'StretchTrans')

    # Compute invariants
    ## I1 = a^2 + b^2 + c^2
    ## I2 = a^2b^2 + b^2c^2 + a^2c^2
    ## J = abc
    ## I1_b = 2b
    ## I1_bb = 2
    ## I2_b = 2b(a^2 + c^2) = 2b(a^2 + b^2)
    ## I2_bb = 2(a^2 + c^2) = 2(a^2 + b^2)
    ## J_b = ac = ab
    ## J_bb = 0
    ## W_bb = W_I1 * I1_bb + W_I1I1 * (I1_b)^2 
    #       + W_I2 * I2_bb + W_I2I2 * (I2_b)^2
    ##      + W_JJ * J_b^2 + 2W_JI1 * I1_b J_b
    I1 = keras.layers.Lambda(lambda x: x[0]**2 + 2 * x[1] ** 2)([StretchAxial, StretchTrans])
    I2 = keras.layers.Lambda(lambda x: x[1] ** 4 + 2 * x[0] ** 2 * x[1] ** 2)([StretchAxial, StretchTrans])
    J = keras.layers.Lambda(lambda x: x[0] * x[1] ** 2)([StretchAxial, StretchTrans])

    # Get strain energy - this ensures the model has trainable weights
    Psi = Psi_model([I1, I2, J])

    # Compute derivatives using GradientTape approach
    def compute_derivatives(inputs):
        i1, i2, j = inputs
        with tf.GradientTape(persistent=True) as tape1:

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([i1, i2, j])
                psi = Psi_model([i1, i2, j])
        
            W1 = tape2.gradient(psi, i1)
            W2 = tape2.gradient(psi, i2)
            WJ = tape2.gradient(psi, j)
        W11 = tape1.gradient(W1, i1, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        W22 = tape1.gradient(W2, i2, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        WJJ = tape1.gradient(WJ, j, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        W1J = tape1.gradient(W1, j, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return W1, W2, WJJ, W11, W22, W1J

    # Compute derivatives
    W1, W2, WJJ, W11, W22, W1J = keras.layers.Lambda(compute_derivatives, 
                                               output_shape=[(None, 1)] * 6)([I1, I2, J])

    # Compute stresses
    
    Convexity = keras.layers.Lambda(function=Convexity_calc_biaxial,
                                       name='Stress_trans')([W1, W2, WJJ, W11, W22, W1J, StretchAxial, StretchTrans])

    # Define model - include Psi in outputs to preserve trainable weights
    model = keras.models.Model(inputs=[StretchAxial, StretchTrans], outputs= [Convexity])
    return model
    
# Complete model architecture definition
def modelArchitecture(Psi_model):
    # Stretch and Gamma as input
    Stretch = keras.layers.Input(shape = (1,), name = 'Stretch')
    Gamma = keras.layers.Input(shape = (1,), name = 'gamma')

    # Compute invariants
    I1 = keras.layers.Lambda(lambda x: 2 + x[0]**2 + x[1] ** 2)([Stretch, Gamma])
    I2 = keras.layers.Lambda(lambda x: 1 + 2 * x[0] ** 2 + x[1] ** 2)([Stretch, Gamma])
    J = keras.layers.Lambda(lambda x: x[0])([Stretch, Gamma])
    
    # CRITICAL FIX: Use custom Layer classes instead of Lambda functions
    # Lambda functions create closures that don't properly track nested model variables.
    # Custom Layer classes ensure Psi_model's variables are tracked by TensorFlow.
    psi_wrapper = PsiModelWrapper(Psi_model, name='psi_wrapper')
    
    # Get strain energy - this ensures the model has trainable weights
    Psi = psi_wrapper([I1, I2, J])

    # Use custom Layer classes instead of Lambda functions for derivative computation
    # This ensures Psi_model's variables are properly tracked
    compute_derivatives_layer = ComputeDerivativesLayer(psi_wrapper, name='compute_derivatives')
    dWdI1, dWdI2, dWdJ = compute_derivatives_layer([I1, I2, J])

    # Compute stresses
    Stress_axial = keras.layers.Lambda(function = Stress_calc_axial,
                                name = 'Stress_axial')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])
    Stress_trans = keras.layers.Lambda(function=Stress_calc_trans,
                                       name='Stress_trans')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])
    Stress_shear = keras.layers.Lambda(function = Stress_calc_shear,
                                name = 'Stress_shear')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])

    # J_max = keras.layers.Lambda(lambda x: x[0] * 0.0 + 1.3, name = 'J_max')([J])
    # Use custom Layer class instead of Lambda function for constraints
    compute_constraints_layer = ComputeConstraintsLayer(psi_wrapper, name='compute_constraints')
    constraint_error = compute_constraints_layer(J) ## Changed from J_max to J, may need higher penalty to reach 0 error
    
    # Create the model - the PsiModelWrapper ensures Psi_model's weights are tracked
    # By using psi_wrapper([I1, I2, J]) directly in the graph (not just in Lambda closures),
    # TensorFlow should automatically track Psi_model's trainable variables
    model = keras.models.Model(inputs=[Stretch, Gamma], outputs= [Stress_axial, Stress_trans, Stress_shear, constraint_error])
    
    # Verify variable tracking - this helps diagnose if the fix is working
    psi_trainable_count = len(Psi_model.trainable_variables)
    model_trainable_count = len(model.trainable_variables)
    
    if model_trainable_count < psi_trainable_count:
        print(f"WARNING: Model has {model_trainable_count} trainable variables, but Psi_model has {psi_trainable_count}")
        print("This indicates Psi_model's weights are NOT being tracked by the parent model.")
        print("\nPossible causes:")
        print("1. PsiModelWrapper is not being recognized as part of the model")
        print("2. Lambda function closures are preventing proper variable tracking")
        print("\nThe wrapper should fix this - verify that psi_wrapper is used in the main graph.")
    else:
        print(f"✓ Success: Model tracks {model_trainable_count} trainable variables")
        print(f"  (Psi_model contributes {psi_trainable_count} variables)")
    
    return Psi_model, model


# Complete model architecture definition
def modelArchitecturePrincipalStretch(Psi_model):
    # Stretch and Gamma as input
    Stretch = keras.layers.Input(shape = (1,), name = 'Stretch')
    Gamma = keras.layers.Input(shape = (1,), name = 'gamma')

    # Compute invariants
    lambda_1 = keras.layers.Lambda(lambda x: (1 + x[0]**2 + x[1] ** 2) / 2 + ((1 + x[0]**2 + x[1] ** 2) ** 2 / 4 - x[0] ** 2) ** 0.5)([Stretch, Gamma])
    lambda_2 = keras.layers.Lambda(lambda x: (1 + x[0]**2 + x[1] ** 2) / 2 - ((1 + x[0]**2 + x[1] ** 2) ** 2 / 4 - x[0] ** 2) ** 0.5)([Stretch, Gamma])
    lambda_3 = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Stretch, Gamma])
    
    # Get strain energy - this ensures the model has trainable weights
    Psi = Psi_model([lambda_1, lambda_2, lambda_3])

    # Compute derivatives using GradientTape approach
    def compute_derivatives(inputs):
        l1, l2, l3 = inputs
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([l1, l2, l3])
            psi = Psi_model([l1, l2, l3])
        
        dWdl1 = tape.gradient(psi, l1)
        dWdl2 = tape.gradient(psi, l2)
        dWdl3 = tape.gradient(psi, l3)
        
        return dWdl1, dWdl2, dWdl3

    # Compute derivatives
    dWdl1, dWdl2, dWdl3 = keras.layers.Lambda(compute_derivatives, 
                                               output_shape=[(None, 1), (None, 1), (None, 1)])([lambda_1, lambda_2, lambda_3])

    # Compute stresses
    Stress_axial = keras.layers.Lambda(function = Stress_calc_axial_principalStretch,
                                name = 'Stress_axial')([dWdl1, dWdl2, dWdl3, Stretch, Gamma])
    Stress_trans = keras.layers.Lambda(lambda x: tf.zeros_like(x[0]), 
                                       output_shape=(None, 1))([Stress_axial])
    # Stress_trans = keras.layers.Lambda(function=Stress_calc_trans,
    #                                    name='Stress_trans')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])
    Stress_shear = keras.layers.Lambda(function = Stress_calc_shear_principalStretch,
                                name = 'Stress_shear')([dWdl1, dWdl2, dWdl3, Stretch, Gamma])

    # Define model - include Psi in outputs to preserve trainable weights
    model = keras.models.Model(inputs=[Stretch, Gamma], outputs= [Stress_axial, Stress_trans, Stress_shear, Psi])
    
    return Psi_model, model






#%% Init
train = True
epochs = 20000
batch_size = 64
model_type = 'convex_mixed_terms_l1_0.01'
weight_flag = True
principal_stretch_flag = False
mixed_flag = True
zero_trans_tension_flag = True
weight_std_flag = False
no_I2_flag = True

# L1 regularization strength
l1_reg = 0.01



path2saveResults_0 = 'Results/'+filename+'/'+model_type
makeDIR(path2saveResults_0)
Model_summary = path2saveResults_0 + '/Model_summary.txt'


# Select loading modes and brain regions of interest
# #modelFit_mode_all = ['SS', 'C', 'T', 'TC', "TC_and_SS"]
        
modelFit_mode_all = ["TC_and_SS"] 
Region_all = ['leap', 'turbo']
# Region_all = ['leap']

# prepare R2 array
R2_all = np.zeros([len(Region_all),len(modelFit_mode_all)+2])

#Import excel file
file_name = 'input/FoamData.xlsx'
dfs = pd.read_excel(file_name, sheet_name='Sheet1')

#%%  Training and validation loop 
count = 1
for id1, Region in enumerate(Region_all):
    
    R2_all_Regions = []
    for id2, modelFit_mode in enumerate(modelFit_mode_all):
        
        print(40*'=')
        print("Comp {:d} / {:d}".format(count, len(Region_all)*len(modelFit_mode_all)))
        print(40*'=')
        print("Region: ", Region ,"| Fitting Mode: ", modelFit_mode)
        print(40*'=')
        count += 1
        
        path2saveResults = os.path.join(path2saveResults_0,Region, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
        
        lam_ut, P_ut, P_ut_std, gamma_ss, P_ss, P_ss_std = getStressStrain(Region)
        
        #%% PI-CANN
        if principal_stretch_flag:
            Psi_model, terms = StrainEnergyPrincipalStretch(l1_reg)
            Psi_model, model = modelArchitecturePrincipalStretch(Psi_model)
        else:   
            Psi_model, terms = StrainEnergyCANN(l1_reg, include_mixed=mixed_flag, no_I2_flag=no_I2_flag)
            Psi_model, model = modelArchitecture(Psi_model)

        
        with open(Model_summary,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
            
        #%%  Model training


        
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, zero_trans_tension=zero_trans_tension_flag, weight_std=weight_std_flag)
            
        Save_path = path2saveResults + '/model.keras'
        Save_weights = path2saveResults + '/weights.weights.h5'
        path_checkpoint = path2saveResults_check + '/best_weights.weights.h5'
        if train:

            ##
            
            model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights)
            # model_given.load_weights(path_checkpoint,  skip_mismatch=False)
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
            Psi_model.save_weights(Save_weights, overwrite=True)
            
            
            # Plot loss function
            # loss_history = history.history['loss']
            # fig, axe = plt.subplots(figsize=[6, 5])  # inches
            # plotLoss(axe, loss_history)
            # plt.savefig(path2saveResults+'/Plot_loss_'+Region+'_'+modelFit_mode+'.pdf')
            # plt.close()
            
        else:
            # Psi_model = tf.keras.models.load_model(Save_path)
            try:
                Psi_model.load_weights(Save_weights, skip_mismatch=False)
            except ValueError as e:
                print(f"Warning: Could not load weights for {Region} - {modelFit_mode}: {e}")
                print("Continuing with randomly initialized weights...")
        
        # Display the strain energy expression
        display_strain_energy_expression(Psi_model, terms)
            
        
        # PI-CANN  get model response
        lam_ut_model = np.linspace(np.amin(lam_ut),np.amax(lam_ut),200)
        gamma_model = np.linspace(np.amin(gamma_ss),np.amax(gamma_ss),200)



        # PI-CANN get model response at data points
        Stress_predict_axial, Stress_predict_trans, _, convexity_error = model.predict([lam_ut, gamma_ss * 0.0])
        _, _, Stress_predict_shear, _ = model.predict([lam_ut * 0 + 0.8, gamma_ss])
        print("Convexity error: ", np.mean(convexity_error)) ## 94 @ weight of 1.0, 
        # _, _, Stress_predict_shear, convexity_error = model([lam_ut * 0 + 0.8, gamma_ss], training=False)
        # print("Convexity error: ", convexity_error)
        # Extract individual term contributions at data points
        stress_contributions, term_names = extract_term_contributions(model, lam_ut, gamma_ss * 0.0)

        #%% Plotting - Create both old combined plot and new separate plots
        
        # Original combined plot (keep the old functionality)
        fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec[0,0])
        # ax2 = fig.add_subplot(spec[1,0])
        # ax3 = fig.add_subplot(spec[2,0])

        # R2, R2_c, R2_t = plotTenCom(fig_ax1, lam_ut, P_ut, Stress_predict_axial, Region)
        plotTrans(ax1, lam_ut, P_ut * 0.0, Stress_predict_trans, Region)
        # R2ss = plotShear(ax2, gamma_ss, P_ss, Stress_predict_shear, Region)
        fig.tight_layout()        

        plt.savefig(path2saveResults+'/Plot_Trans_'+Region+'_'+modelFit_mode+'.pdf')
        plt.close()
        
        # New separate plots with stacked contributions
        
        # 1. Tension plot
        fig_tension, R2_t = plotTensionWithContributions(
            lam_ut, P_ut, Stress_predict_axial,
            stress_contributions, term_names, Region
        )
        plt.savefig(path2saveResults+'/Plot_Tension_Contributions_'+Region+'_'+modelFit_mode+'.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Compression plot
        fig_compression, R2_c = plotCompressionWithContributions(
            lam_ut, P_ut, Stress_predict_axial,
            stress_contributions, term_names, Region
        )
        plt.savefig(path2saveResults+'/Plot_Compression_Contributions_'+Region+'_'+modelFit_mode+'.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Shear plot - need to extract contributions for shear case
        stress_contributions_shear, _ = extract_shear_term_contributions(model, Psi_model, lam_ut * 0 + 0.8, gamma_ss)
        fig_shear, R2ss = plotShearWithContributions(
            gamma_ss, P_ss, Stress_predict_shear,
            stress_contributions_shear, term_names, Region
        )
        plt.savefig(path2saveResults+'/Plot_Shear_Contributions_'+Region+'_'+modelFit_mode+'.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()





        # terms =4*7
        #%% Show weights
        weight_matrix = np.empty([int(terms), 2])
        weight_matrix[:] = np.nan        
        Num_add_var = 0
        
        if weight_flag:
            # Get all weights from the model
            all_weights = Psi_model.get_weights()
            
            # Check if we have enough weights
            if len(all_weights) > terms + Num_add_var:
                second_layer = all_weights[terms+Num_add_var].flatten()
            else:
                second_layer = np.array([])
            
            j=0
            if weight_flag:
                for i in range(int(terms)):
                    try:
                        if i + Num_add_var < len(all_weights):
                            value = all_weights[i + Num_add_var][0][0]
                        else:
                            value = np.nan
                        
                        if j < len(second_layer):
                            weight_matrix[i,1] = second_layer[j]
                            j+=1
                        else:
                            weight_matrix[i,1] = np.nan
                    except (ValueError, IndexError):
                        value = np.nan
                        weight_matrix[i,1] = np.nan
                    
                    weight_matrix[i,0] = value
                
            
    
    
        
        #%% Plotting
        
        Config = {Region:Region, modelFit_mode:modelFit_mode, "R2_c":R2_c, "R2_t": R2_t, "R2_ss": R2ss, "weigths": weight_matrix.tolist()}
        json.dump(Config, open(path2saveResults+"/Config_file.txt",'w'))
        
        if modelFit_mode == 'T':
            R2_cur = [R2_t]
        elif modelFit_mode == "C":
            R2_cur = [R2_c]
        elif modelFit_mode == "TC":
            R2_cur = [R2] 
        elif modelFit_mode == "SS":
            R2_cur = [R2ss]    
        elif modelFit_mode == "TC_and_SS":
            R2_cur = [R2ss, R2_c, R2_t]
         
        R2_all_Regions.append(R2_cur)
        print("R2: ", R2_cur) 


        #### Transverse Stress Plot Convexity
        model_biaxial = modelArchitectureBiaxial(Psi_model)
        Convexity = model_biaxial.predict([lam_ut * 0 + 1.3, lam_ut])
        plt.plot(lam_ut, Convexity)
        plt.show()
            

    R2_all[id1,:] = np.array(flatten(R2_all_Regions))
    
    


#%% Summarizing results    
modelFit_mode_all_table = ['SS', 'C', 'T']  # Match the actual R2 outputs
R2_mean = np.expand_dims(np.mean(R2_all,axis=0), axis=0)
R2_sd = np.expand_dims(np.std(R2_all,axis=0), axis=0)
R2_all_mean = np.concatenate((R2_all,R2_mean,R2_sd), axis=0)
R2_df = pd.DataFrame(R2_all_mean, index=Region_all + ['mean', 'SD'], columns=modelFit_mode_all_table)
R2_df.to_latex(path2saveResults_0+'/R2_table.tex',index=True)
R2_df.to_csv(path2saveResults_0+'/R2_table.csv',index=True)

