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
from models_brain import SinglePrincipalStretchLayer, OuterLayer  # Import explicitly for type checking
from plotting_brain import*

from sklearn.metrics import r2_score


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



## Load data from df into variables for stretch and stress
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
    """Calculate relative sample weights based on maximum stress in each mode"""
    midpoint = int(len(lam_ut) / 2)
    
    # Calculate maximum stress for each mode
    max_tension = np.max(np.abs(P_ut[midpoint:]))
    max_compression = np.max(np.abs(P_ut[:(midpoint + 1)]))
    max_shear = np.max(np.abs(P_ss))
    
    # Calculate relative weights (inverse of maximum stress to balance training)
    # Higher stress modes get lower weights to prevent dominance
    weight_tension = 1.0 / max_tension ** 2
    weight_compression = 1.0 / max_compression ** 2
    weight_shear = 0.5 / max_shear ** 2 ## Reduced weight for shear since each shear value appears twice in the data (positive and negative)
    
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
                
                if i == 2:
                    expr = f"{w_outer:.6f} (J^{w_inner:.6f} - {w_inner:.6f} log(J) - 1)"
                elif i == 0:
                    expr = f"{w_outer*w_inner:.6f} log(J)^2"
                elif i == 1:
                    expr = f"{w_outer:.6f} (exp({w_inner:.6f} log(J)^2) - 1)"
                expression_terms.append(expr)
            term_idx += 1

    # Mixed terms (1-2 terms from MixedNet)
    print("\nMixed terms:")
    for i in range(2):
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


def traindata(modelFit_mode):
    # The model_given should be the stress-output model, but we need to ensure it has trainable weights
    model_given = model
    midpoint = int(len(lam_ut) / 2)
    
    weight_tension, weight_compression, weight_shear = calculate_relative_weights()
    stretch_shear = lam_ut * 0 + 0.8
    input_train = [lam_ut, stretch_shear, gamma_ss]
    psi_output = P_ut * 0.0 # Psi output is not used for training, but is needed for the model to be able to compute the stress contributions
    stress_trans = P_ut * 0.0 # Target transverse stress is zero
    output_train = [P_ut, stress_trans, psi_output, P_ss, psi_output]
    inp_len = len(lam_ut)

    
    if modelFit_mode == 'T':
        weight_compression = 0.0
        weight_shear = 0.0
    elif modelFit_mode == "C":
        weight_tension = 0.0
        weight_shear = 0.0
    elif modelFit_mode == "TC":
        weight_shear = 0.0
    elif modelFit_mode == "SS":
        weight_tension = 0.0
        weight_compression = 0.0
        

    weights_axial_trans = np.concatenate([
            np.array([weight_compression] * (midpoint + 1)),  # compression data
            np.array([weight_tension] * (len(lam_ut) - midpoint - 1))  # tension data
        ])
    sample_weights = [weights_axial_trans, weights_axial_trans, np.array([0.0]* inp_len), np.array([weight_shear]*inp_len), np.array([0.0]*inp_len)]
    return model_given, input_train, output_train, sample_weights

        

def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, model_unreg=None):

    # If model_unreg is provided, train the unregularized model first
    if model_unreg is not None:
        # Define loss, metrics and optimizer
        mse_loss = keras.losses.MeanSquaredError()
        metrics  =[keras.metrics.MeanSquaredError()] * 5
        opti1    = tf.optimizers.Adam(learning_rate=0.001)
        
        # Compile the unregularized model
        model_unreg.compile(loss=mse_loss,
                  optimizer=opti1,
                  metrics=metrics)

        # Create callbacks to terminate training if the loss does not improve for 3000 epochs, and to save the best weights
        es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=3000, restore_best_weights=True)

        modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="loss",
            filepath=path_checkpoint,
            verbose=0,
            save_weights_only=True,
            save_best_only=True,
        )

        # Train the unregularized model
        history_unreg = model_unreg.fit(input_train,
                        output_train,
                        batch_size=batch_size,
                        epochs=5000,
                        validation_split=0.0,
                        callbacks=[es_callback, modelckpt_callback],
                        shuffle = True,
                        verbose = 0, 
                        sample_weight = sample_weights)

        # Set the initial weights of the regularized model to the final weights of the unregularized model
        model_given.load_weights(path_checkpoint)

    # Train the regularized model
    # Define loss, metrics and optimizer
    mse_loss = keras.losses.MeanSquaredError()
    metrics  =[keras.metrics.MeanSquaredError()] * 5
    opti1    = tf.optimizers.Adam(learning_rate=0.01)
    
    # Compile the regularized model
    model_given.compile(loss=mse_loss,
                  optimizer=opti1,
                  metrics=metrics)
    
    
    # Create callbacks to terminate training if the loss does not improve for 3000 epochs, and to save the best weights
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=3000, restore_best_weights=True)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="loss",
    filepath=path_checkpoint,
    verbose=0,
    save_weights_only=True,
    save_best_only=True,
    )
    
    # Train the regularized model
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



def Stress_calc_shear_principalStretch(inputs):
    eps = 1e-9 # Small epsilon to avoid division by zero
    (dWdl1, dWdl2, dWdl3, Stretch, Gamma) = inputs
    # Compute derivatives of the principal stretches with respect to gamma
    dl1dGamma = Gamma * (1 + (Stretch ** 2 + Gamma ** 2 + 1) / (2 * ((1 + Stretch ** 2 + Gamma ** 2)**2 / 4 - Stretch ** 2 + eps) ** 0.5))
    dl2dGamma = Gamma * (1 - (Stretch ** 2 + Gamma ** 2 + 1) / (2 * ((1 + Stretch ** 2 + Gamma ** 2)**2 / 4 - Stretch ** 2 + eps) ** 0.5))
    # Use chain rule to compute derivatives of the strain energy with respect to gamma
    return dWdl1 * dl1dGamma + dWdl2 * dl2dGamma

class ComputeDerivativesLayer(keras.layers.Layer):
    """
    Custom layer that computes derivatives of Psi_model using GradientTape.
    This ensures Psi_model's variables are tracked because it's called inside
    a Layer's call() method, not a Lambda closure.
    """
    def __init__(self, psi_model, **kwargs):
        super(ComputeDerivativesLayer, self).__init__(**kwargs)
        self.psi_model = psi_model
    
    def call(self, inputs):
        l1, l2, l3 = inputs
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([l1, l2, l3])
            # Call the model directly - TensorFlow will track its variables
            psi = self.psi_model([l1, l2, l3])
        
        dWdl1 = tape.gradient(psi, l1)
        dWdl2 = tape.gradient(psi, l2)
        dWdl3 = tape.gradient(psi, l3)
        
        # Return as separate outputs that can be unpacked
        return dWdl1, dWdl2, dWdl3


def extract_term_contributions(model, stretch_input, gamma_input):
    """
    Extract individual term contributions to STRESS using weight zeroing approach.
    This function works with StrainEnergyPrincipalStretch architectures.
    """
    
    # Find Psi_model or SinglePrincipalStretchLayer by traversing the model structure
    def find_layer_recursive(layer, target_class, target_name=None):
        """Recursively search for a layer of a specific class or name"""
        if isinstance(layer, target_class):
            if target_name is None or layer.name == target_name:
                return layer
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                result = find_layer_recursive(sublayer, target_class, target_name)
                if result is not None:
                    return result
        return None
    
    # Find Psi_model and then find SingleOutputDenseLayer in it
    Psi_model = find_layer_recursive(model, keras.models.Model, 'Psi')
    
    if Psi_model is None:
        raise ValueError("Could not find Psi_model in the main model")
    
    # Find SingleOutputDenseLayer in Psi_model
    final_layer = None
    for layer in reversed(Psi_model.layers):
        if isinstance(layer, OuterLayer):
            final_layer = layer
            break
    
    if final_layer is None:
        # Try to find it directly in the model
        final_layer = find_layer_recursive(model, OuterLayer, None)
    
    if final_layer is None:
        raise ValueError("Could not find SingleOutputDenseLayer in the model")
    
    # Check if the layer has weights
    if not final_layer.get_weights():
        raise ValueError("The final layer has no weights")
    
    original_weights = final_layer.get_weights()[0].copy()
    
    # Get the actual number of terms from the weight shape
    num_terms = original_weights.shape[0]
    print(num_terms)

    # Build term names dynamically based on the structure
    ## TODO: Fix this to work with modular model
    term_names = []
    if num_terms >= 4:
        term_names.extend(["I1", "exp(I1)-1", "I1²", "exp(I1²)-1"])
    if num_terms >= 8:
        term_names.extend(["I2", "exp(I2)-1", "I2²", "exp(I2²)-1"])
    if num_terms >= 11:
        term_names.extend(["Jᵐ - m ln(J)", "exp(ln(J)²)"])
    if num_terms > 11:
        term_names.extend(["λiᵐ - m ln(λi)"])
    
    # If there are more or fewer terms than expected, adjust
    if len(term_names) < num_terms:
        for i in range(len(term_names), num_terms):
            term_names.append(f"Term {i+1}")
    elif len(term_names) > num_terms:
        term_names = term_names[:num_terms]
    
    stress_contributions = []
    shear_stress_contributions = []
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
        term_prediction = model([stretch_input, stretch_input * 0 + 0.8, gamma_input], training=False)
        term_stress = term_prediction[0].numpy() if hasattr(term_prediction[0], 'numpy') else term_prediction[0]
        term_shear = term_prediction[3].numpy() if hasattr(term_prediction[0], 'numpy') else term_prediction[3]

        # Normalize shape to (n_points,)
        ts = np.array(term_stress)
        ts_shear = np.array(term_shear)
        ts = np.squeeze(ts)
        ts_shear = np.squeeze(ts_shear)
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
        shear_stress_contributions.append(np.array(ts_shear).reshape(-1))
    
    # Restore original weights
    final_layer.set_weights([original_weights])
    
    return stress_contributions, shear_stress_contributions, term_names

# Complete model architecture definition
def modelArchitecture(Psi_model):
    ## Uniaxial tension model

    # Compute principal stretches
    Stretch_ut = keras.layers.Input(shape = (1,), name = 'Stretch_ut')
    lambda_2_ut = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Stretch_ut])
    lambda_3_ut = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Stretch_ut])
    # Compute strain energy
    Psi_ut = Psi_model([Stretch_ut, lambda_2_ut, lambda_3_ut])
    # Compute derivatives of the strain energy with respect to the principal stretches
    compute_derivatives_layer = ComputeDerivativesLayer(Psi_model, name='compute_derivatives_ut')
    dWdl1, dWdl2, dWdl3 = compute_derivatives_layer([Stretch_ut, lambda_2_ut, lambda_3_ut])   
    # Compute stress
    Stress_axial = dWdl1
    Stress_trans = dWdl2
    model_ut = keras.models.Model(inputs=[Stretch_ut], outputs= [Stress_axial, Stress_trans, Psi_ut])

    ## Simple shear model
    # Define inputs
    Stretch_ss = keras.layers.Input(shape = (1,), name = 'Stretch_ss')
    Gamma_ss = keras.layers.Input(shape = (1,), name = 'Gamma_ss')
    # Compute principal stretches
    lambda_1_ss = keras.layers.Lambda(lambda x: (1 + x[0]**2 + x[1] ** 2) / 2 + ((1 + x[0]**2 + x[1] ** 2) ** 2 / 4 - x[0] ** 2) ** 0.5)([Stretch_ss, Gamma_ss])
    lambda_2_ss = keras.layers.Lambda(lambda x: (1 + x[0]**2 + x[1] ** 2) / 2 - ((1 + x[0]**2 + x[1] ** 2) ** 2 / 4 - x[0] ** 2) ** 0.5)([Stretch_ss, Gamma_ss])
    lambda_3_ss = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Gamma_ss])
    # Compute strain energy
    Psi_ss = Psi_model([lambda_1_ss, lambda_2_ss, lambda_3_ss])
    # Compute derivatives of the strain energy with respect to the principal stretches
    compute_derivatives_layer = ComputeDerivativesLayer(Psi_model, name='compute_derivatives_ss')
    dWdl1, dWdl2, dWdl3 = compute_derivatives_layer([lambda_1_ss, lambda_2_ss, lambda_3_ss])
    # Compute stress
    Stress_shear = keras.layers.Lambda(function = Stress_calc_shear_principalStretch,
                                name = 'Stress_shear')([dWdl1, dWdl2, dWdl3, Stretch_ss, Gamma_ss])
    model_ss = keras.models.Model(inputs=[Stretch_ss, Gamma_ss], outputs= [Stress_shear, Psi_ss])

    # Combine the models
    model_combined = keras.models.Model(inputs=[Stretch_ut, Stretch_ss, Gamma_ss], outputs= [Stress_axial, Stress_trans,Psi_ut, Stress_shear, Psi_ss])
    return model_combined





#%% Init
train = True
epochs = 5000
batch_size = 64
# Name the folder to save the results
model_type = 'final_inv_only_lp_0'
# Choose which terms to include in the model architecture
include_invariant_terms = True
include_mixed_terms = False
include_principal_stretch_terms = False
# Lp regularization strength
lp_reg = 0.0
p = 0.5

# Create the path to save the results
path2saveResults_0 = 'Results/'+filename+'/'+model_type
makeDIR(path2saveResults_0)
Model_summary = path2saveResults_0 + '/Model_summary.txt'


# Select loading modes and foam types of interest
# #modelFit_mode_all = ['SS', 'C', 'T', 'TC', "TC_and_SS"]
modelFit_mode_all = ["TC_and_SS"] 
foam_types_all = ['leap', 'turbo']

# prepare R2 array
R2_all = np.zeros([len(foam_types_all),len(modelFit_mode_all)+2])

#Import excel file
file_name = 'input/FoamData.xlsx'
dfs = pd.read_excel(file_name, sheet_name='Sheet1')

#%%  Training and validation loop 
count = 1
for id1, foam_type in enumerate(foam_types_all):
    
    R2_curr_foam_type = []
    for id2, modelFit_mode in enumerate(modelFit_mode_all):
        
        print(40*'=')
        print("Comp {:d} / {:d}".format(count, len(foam_types_all)*len(modelFit_mode_all)))
        print(40*'=')
        print("Foam type: ", foam_type ,"| Fitting Mode: ", modelFit_mode)
        print(40*'=')
        count += 1
        
        # Create new folder to save the results
        path2saveResults = os.path.join(path2saveResults_0,foam_type, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
        
        # Get the stress-strain data from the excel file
        lam_ut, P_ut, P_ut_std, gamma_ss, P_ss, P_ss_std = getStressStrain(foam_type)
        
        # Create the strain energy (Psi) model
        Psi_model_unreg, terms = StrainEnergyCANN(0.0, p, include_invariant_terms, include_mixed_terms, include_principal_stretch_terms)
        model_unreg = modelArchitecture(Psi_model_unreg)
        Psi_model, terms = StrainEnergyCANN(lp_reg, p, include_invariant_terms, include_mixed_terms, include_principal_stretch_terms)
        model = modelArchitecture(Psi_model)

        
        with open(Model_summary,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
            
        #%%  Model training


        
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode)
            
        Save_path = path2saveResults + '/model.keras'
        Save_weights = path2saveResults + '/weights.weights.h5'
        path_checkpoint = path2saveResults_check + '/best_weights.weights.h5'
        if train:

            ##
            
            model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, model_unreg)

            # model_given.load_weights(path_checkpoint,  skip_mismatch=False)
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
            Psi_model.save_weights(Save_weights, overwrite=True)
            
            
            # Plot loss function
            loss_history = history.history['loss']
            fig, axe = plt.subplots(figsize=[6, 5])  # inches
            plotLoss(axe, loss_history)
            plt.savefig(path2saveResults+'/Plot_loss_'+foam_type+'_'+modelFit_mode+'.pdf')
            plt.close()
            
        else:
            # Psi_model = tf.keras.models.load_model(Save_path)
            try:
                Psi_model.load_weights(Save_weights, skip_mismatch=False)
            except ValueError as e:
                print(f"Warning: Could not load weights for {foam_type} - {modelFit_mode}: {e}")
                print("Continuing with randomly initialized weights...")
        
        # Display the strain energy expression
        display_strain_energy_expression(Psi_model, terms)
            
        
        # PI-CANN  get model response
        lam_ut_model = np.linspace(np.amin(lam_ut),np.amax(lam_ut),200)
        gamma_model = np.linspace(np.amin(gamma_ss),np.amax(gamma_ss),200)



        # PI-CANN get model response at data points
        Stress_predict_axial, Stress_predict_trans, _, Stress_predict_shear, _ = model.predict([lam_ut, lam_ut * 0 + 0.8, gamma_ss])
        # Extract individual term contributions at data points
        stress_contributions, shear_stress_contributions, term_names = extract_term_contributions(model, lam_ut, gamma_ss)

        #%% Plotting - Create both old combined plot and new separate plots
        
        # Original combined plot (keep the old functionality)
        fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
        ax1 = fig.add_subplot(spec[0,0])

        print(Stress_predict_axial)
        # R2, R2_c, R2_t = plotTenCom(ax1, lam_ut, P_ut, Stress_predict_axial, Region)
        plotTrans(ax1, lam_ut, P_ut * 0.0, Stress_predict_trans, foam_type)
        # R2_ss = plotShear(ax2, gamma_ss, P_ss, Stress_predict_shear, Region)
        fig.tight_layout()        

        plt.savefig(path2saveResults+'/Plot_Trans_'+foam_type+'_'+modelFit_mode+'.pdf')
        plt.close()
        
        # New separate plots with stacked contributions
        
        # 1. Tension plot
        fig_tension, R2_t = plotTensionWithContributions(
            lam_ut, P_ut, Stress_predict_axial,
            stress_contributions, term_names, foam_type
        )
        plt.savefig(path2saveResults+'/Plot_Tension_Contributions_'+foam_type+'_'+modelFit_mode+'.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Compression plot
        fig_compression, R2_c = plotCompressionWithContributions(
            lam_ut, P_ut, Stress_predict_axial,
            stress_contributions, term_names, foam_type
        )
        plt.savefig(path2saveResults+'/Plot_Compression_Contributions_'+foam_type+'_'+modelFit_mode+'.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Shear plot - need to extract contributions for shear case
        # stress_contributions_shear, _ = extract_shear_term_contributions(model, Psi_model, lam_ut * 0 + 0.8, gamma_ss)
        fig_shear, R2_ss = plotShearWithContributions(
            gamma_ss, P_ss, Stress_predict_shear,
            shear_stress_contributions, term_names, foam_type
        )
        plt.savefig(path2saveResults+'/Plot_Shear_Contributions_'+foam_type+'_'+modelFit_mode+'.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()


        if modelFit_mode == 'T':
            R2_cur = [R2_t]
        elif modelFit_mode == "C":
            R2_cur = [R2_c]
        elif modelFit_mode == "TC":
            R2 = r2_score_own(P_ut, Stress_predict_axial)
            R2_cur = [R2] 
        elif modelFit_mode == "SS":
            R2_cur = [R2_ss]    
        elif modelFit_mode == "TC_and_SS":
            R2_cur = [R2_ss, R2_c, R2_t]
         
        R2_curr_foam_type.append(R2_cur)
        print("R2: ", R2_cur) 


            

    R2_all[id1,:] = np.array(flatten(R2_curr_foam_type))
    
    


#%% Summarizing results    
modelFit_mode_all_table = ['SS', 'C', 'T']  # Match the actual R2 outputs
R2_mean = np.expand_dims(np.mean(R2_all,axis=0), axis=0)
R2_sd = np.expand_dims(np.std(R2_all,axis=0), axis=0)
R2_all_mean = np.concatenate((R2_all,R2_mean,R2_sd), axis=0)
R2_df = pd.DataFrame(R2_all_mean, index=foam_types_all + ['mean', 'SD'], columns=modelFit_mode_all_table)
R2_df.to_latex(path2saveResults_0+'/R2_table.tex',index=True)
R2_df.to_csv(path2saveResults_0+'/R2_table.csv',index=True)

