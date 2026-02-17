#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for CANN4brain_main.py
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
from sklearn.metrics import r2_score
from models_brain import OuterLayer


def count_nonzero_terms(Psi_model, p=0.5, threshold=1e-6):
    """
    Count the number of nonzero terms in a Psi_model by checking the outer_weights.
    
    Args:
        Psi_model: The Keras model containing the OuterLayer
        p: The power parameter used in OuterLayer (default 0.5)
        threshold: Threshold below which a weight is considered zero (default 1e-6)
    
    Returns:
        int: Number of nonzero terms
    """
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
    
    # Find OuterLayer in Psi_model
    final_layer = None
    for layer in reversed(Psi_model.layers):
        if isinstance(layer, OuterLayer):
            final_layer = layer
            break
    
    if final_layer is None:
        # Try to find it recursively
        final_layer = find_layer_recursive(Psi_model, OuterLayer, None)
    
    if final_layer is None:
        raise ValueError("Could not find OuterLayer in the model")
    
    # Get the outer_weights
    if not final_layer.get_weights():
        raise ValueError("The OuterLayer has no weights")
    
    outer_weights = final_layer.get_weights()[0]  # Shape: (num_terms, 1)
    
    # The weights are raised to power 1/p in the call method, so we need to check
    # the actual weights raised to 1/p, or we can check the weights directly
    # Since weights are constrained to be non-negative, we check them directly
    # and count how many are above the threshold
    weights_raised = np.abs(outer_weights ** (1.0 / p))
    nonzero_count = np.sum(weights_raised > threshold)
    
    return int(nonzero_count)


def makeDIR(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def flatten(l):
    """Flatten a nested list"""
    return [item for sublist in l for item in sublist]


def r2_score_own(Truth, Prediction):
    """Calculate R2 score, ensuring it's non-negative"""
    R2 = r2_score(Truth, Prediction)
    return max(R2, 0.0)


def getStressStrain(Region, dfs):
    """
    Load data from df into variables for stretch and stress
    
    Args:
        Region: 'leap' or 'turbo'
        dfs: DataFrame containing the stress-strain data
        
    Returns:
        lam_ut, P_ut, P_ut_std, gamma_ss, P_ss, P_ss_std
    """
    if Region == 'leap':
        lam_ut = dfs.iloc[:, 0].astype(np.float64)
        P_ut = dfs.iloc[:, 1].astype(np.float64)
        P_ut_std = dfs.iloc[:, 2].astype(np.float64)
        gamma_ss = dfs.iloc[:, 3].astype(np.float64).values
        P_ss = dfs.iloc[:, 4].astype(np.float64).values
        P_ss_std = dfs.iloc[:, 5].astype(np.float64)
    elif Region == 'turbo':
        lam_ut = dfs.iloc[:, 6].astype(np.float64)
        P_ut = dfs.iloc[:, 7].astype(np.float64)
        P_ut_std = dfs.iloc[:, 8].astype(np.float64)
        gamma_ss = dfs.iloc[:, 9].astype(np.float64).values
        P_ss = dfs.iloc[:, 10].astype(np.float64).values
        P_ss_std = dfs.iloc[:, 11].astype(np.float64)
    else:
        raise ValueError(f"Unknown Region: {Region}")

    return lam_ut, P_ut, P_ut_std, gamma_ss, P_ss, P_ss_std


def calculate_relative_weights(lam_ut, P_ut, P_ss):
    """
    Calculate relative sample weights based on maximum stress in each mode
    
    Args:
        lam_ut: Uniaxial tension stretch data
        P_ut: Uniaxial tension stress data
        P_ss: Simple shear stress data
        
    Returns:
        weight_tension, weight_compression, weight_shear
    """
    midpoint = int(len(lam_ut) / 2)
    
    # Calculate maximum stress for each mode
    max_tension = np.max(np.abs(P_ut[midpoint:]))
    max_compression = np.max(np.abs(P_ut[:(midpoint + 1)]))
    max_shear = np.max(np.abs(P_ss))
    
    # Calculate relative weights (inverse of maximum stress to balance training)
    # Higher stress modes get lower weights to prevent dominance
    weight_tension = 1.0 / max_tension ** 2
    weight_compression = 1.0 / max_compression ** 2
    weight_shear = 0.5 / max_shear ** 2  # Reduced weight for shear since each shear value appears twice in the data (positive and negative)
    
    # Normalize weights so they sum to 3 (one for each mode)
    total_weight = weight_tension + weight_compression + weight_shear
    weight_tension = weight_tension / total_weight * 3.0
    weight_compression = weight_compression / total_weight * 3.0
    weight_shear = weight_shear / total_weight * 3.0
    
    return weight_tension, weight_compression, weight_shear


def make_sample_weights(weight_ax, weight_trans, weight_shr, inp_len):
    """Create sample weights array for training"""
    return [np.array([weight_ax] * inp_len),
            np.array([weight_trans] * inp_len),
            np.array([weight_shr] * inp_len),
            np.array([0.0] * inp_len)]


def make_sample_weights_std(weight_ax, weight_trans, weight_shr, inp_len):
    """Create sample weights array with standard format"""
    return [weight_ax,
            np.array([weight_trans] * inp_len),
            weight_shr,
            np.array([0.0] * inp_len)]


def format_value_sigfig(value, sigfigs=3):
    """
    Format a value to 3 significant figures, or as an integer if >= 1000.
    
    Args:
        value: The numeric value to format
        sigfigs: Number of significant figures (default 3)
    
    Returns:
        str: Formatted value as string
    """
    # Handle NaN
    if np.isnan(value):
        return "NaN"
    
    # Handle infinity
    if np.isinf(value):
        if value > 0:
            return "inf"
        else:
            return "-inf"
    
    if abs(value) >= 1000:
        return f"{int(value)}"
    else:
        # Format to 3 significant figures
        if value == 0:
            return "0"
        # Calculate the order of magnitude
        abs_value = abs(value)
        if abs_value < 1e-10:  # Handle very small values
            return "0"
        magnitude = np.floor(np.log10(abs_value))
        # Calculate the number of decimal places needed
        decimals = max(0, int(sigfigs - 1 - magnitude))
        # Ensure we don't have too many decimal places
        decimals = min(decimals, 10)
        return f"{value:.{decimals}f}"


def display_strain_energy_expression(Psi_model, terms, mixed_layer, single_principal_stretch_layer, single_principal_stretch_layer2):

    
    """
    Display the full strain energy expression as a function of I1, I2, J
    
    Args:
        Psi_model: The strain energy model
        terms: Number of terms in the model
    """
    print("\n" + "=" * 80)
    print("STRAIN ENERGY EXPRESSION")
    print("=" * 80)

    print("Mixed layer weights:")
    print(mixed_layer.alpha_I1)
    print(mixed_layer.alpha_I2)
    
    print("Single principal stretch layer weights:")
    print(single_principal_stretch_layer.weights)
    print(single_principal_stretch_layer2.weights)
    
    # Get the weights from the model
    weights = Psi_model.get_weights()
    print("Full weights:")
    print(weights)
    
    # Extract the final layer weights (these multiply each term)
    final_weights = weights[-1].flatten()
    print("Number of final weights: ", len(final_weights))
    
    
    # Define the reduced invariants
    print("Psi(I1_bar, I2_bar, J) where:")
    print("  I1_bar = I1/J^(2/3) - 3")
    print("  I2_bar = I2/J^(4/3) - 3")
    print("  J = J")
    print(weights)
    
    # Build the expression
    expression_terms = []
    term_idx = 0
    term_idx_inner = 0
    
    # I1 terms (4 terms per SingleInvNet)
    for i in range(4):
        if term_idx < len(final_weights):
            w_outer = final_weights[term_idx] ** 2 # Because of lp regularization, the weights are squared
            # Get the inner weight (coefficient inside the activation function)
            w_inner = weights[term_idx_inner][0][0]
            if i == 0:
                expr = f"{format_value_sigfig(w_outer)} (\\bar I_1 - 3)"
            elif i == 1:
                expr = f"{format_value_sigfig(w_outer / w_inner)} (\\exp({format_value_sigfig(w_inner)} (\\bar I_1 - 3)) - 1)"
                term_idx_inner += 1
            elif i == 2:
                expr = f"{format_value_sigfig(w_outer)} (\\bar I_1 - 3)^2"
            elif i == 3:
                expr = f"{format_value_sigfig(w_outer / w_inner)} (\\exp({format_value_sigfig(w_inner)} (\\bar I_1 - 3)^2) - 1)"
                term_idx_inner += 1

            if abs(w_outer) > 1e-6:  # Only show significant terms
                expression_terms.append(expr)

            term_idx += 1
    
    # I2 terms (4 terms per SingleInvNet)
    for i in range(4):
        w_outer = final_weights[term_idx] ** 2 # Because of lp regularization, the weights are squared
        # Get the inner weight (coefficient inside the activation function)
        w_inner = weights[term_idx_inner][0][0]
        
        if i == 0:
            expr = f"{format_value_sigfig(w_outer)} (\\bar I_2 - 3)"
        elif i == 1:
            expr = f"{format_value_sigfig(w_outer / w_inner)} (\\exp({format_value_sigfig(w_inner)} (\\bar I_2 - 3)) - 1)"
            term_idx_inner += 1
        elif i == 2:
            expr = f"{format_value_sigfig(w_outer)} (\\bar I_2 - 3)^2"
        elif i == 3:
            expr = f"{format_value_sigfig(w_outer / w_inner)} (\\exp({format_value_sigfig(w_inner)} (\\bar I_2 - 3)^2) - 1)"
            term_idx_inner += 1
        if abs(w_outer) > 1e-6:  # Only show significant terms
            expression_terms.append(expr)
        term_idx += 1
    
    # J terms (3 terms from BulkNet)
    for i in range(2):
        w_outer = final_weights[term_idx] ** 2 # Because of lp regularization, the weights are squared
        # Get the inner weight (coefficient inside the activation function)
        w_inner = weights[term_idx_inner][0][0]
        
        if i == 0:
            expr = f"{format_value_sigfig(2 * w_outer / w_inner ** 2)} (J^{{{format_value_sigfig(w_inner)}}} - {format_value_sigfig(w_inner)} \\ln(J) - 1)"
        elif i == 1:
            expr = f"{format_value_sigfig(w_outer / w_inner)} (\\exp({format_value_sigfig(w_inner)} \\ln(J)^2) - 1)"
        if abs(w_outer) > 1e-6:  # Only show significant terms
            expression_terms.append(expr)
        term_idx_inner += 1
        term_idx += 1

    # Mixed terms (1-2 terms from MixedNet)
    for i in range(2):
        w_outer = final_weights[term_idx] ** 2 # Because of lp regularization, the weights are squared
        # Get the inner weight (coefficient inside the activation function)
        w_inner = weights[term_idx_inner][0][0]
        print(w_inner)
        
        if i == 0:
            expr = f"{format_value_sigfig(w_outer)} J^{{{format_value_sigfig(w_inner)}}} (\\bar I_1 - 3) "
        elif i == 1:
            expr = f"{format_value_sigfig(w_outer)} J^{{{format_value_sigfig(w_inner)}}} (\\bar I_2 - 3)"
        if abs(w_outer) > 1e-6:  # Only show significant terms
            expression_terms.append(expr)
        term_idx_inner += 1
        term_idx += 1

    # PS terms (1-2 terms from Principal Stretch Layer)
    for i in range(2):
        w_outer = final_weights[term_idx]
        # Get the inner weight (coefficient inside the activation function)
        w_inner = weights[term_idx_inner][0][0]
        
        if i == 0:
            expr = f"{format_value_sigfig(w_outer / w_inner ** 2)} \\sum_{{i=1}}^3(\\lambda_i^{{{format_value_sigfig(w_inner)}}} - {format_value_sigfig(w_inner)} \\ln(\\lambda_i) - 1)"
        elif i == 1:
            expr = f"{format_value_sigfig(w_outer / w_inner ** 2)} \\sum_{{i=1}}^3(\\lambda_i^{{{format_value_sigfig(w_inner)}}} - {format_value_sigfig(w_inner)} \\ln(\\lambda_i) - 1)"
        if abs(w_outer) > 1e-6:  # Only show significant terms
            expression_terms.append(expr)
        term_idx_inner += 1
        term_idx += 1
    # Display the full expression
    print(f"\nFull Expression:")
    if expression_terms:
        full_expr = " + ".join(expression_terms)
        print(f"\\Psi(\\bar I_1, \\bar I_2, J) = {full_expr}")
    else:
        print("All weights are negligible")
    
    print("=" * 80)
    print()
    
    return expression_terms


def traindata(modelFit_mode, model, lam_ut, P_ut, P_ss, gamma_ss):
    """
    Prepare training data and sample weights
    
    Args:
        modelFit_mode: Training mode ('T', 'C', 'TC', 'SS', 'TC_and_SS')
        model: The model to train
        lam_ut: Uniaxial tension stretch data
        P_ut: Uniaxial tension stress data
        P_ss: Simple shear stress data
        gamma_ss: Simple shear strain data
        
    Returns:
        model_given, input_train, output_train, sample_weights
    """
    model_given = model
    midpoint = int(len(lam_ut) / 2)
    
    weight_tension, weight_compression, weight_shear = calculate_relative_weights(lam_ut, P_ut, P_ss)
    stretch_shear = lam_ut * 0 + 0.8
    input_train = [lam_ut, stretch_shear, gamma_ss]
    psi_output = P_ut * 0.0  # Psi output is not used for training, but is needed for the model to be able to compute the stress contributions
    stress_trans = P_ut * 0.0  # Target transverse stress is zero
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
    sample_weights = [weights_axial_trans, weights_axial_trans, np.array([0.0] * inp_len), np.array([weight_shear] * inp_len), np.array([0.0] * inp_len)]
    return model_given, input_train, output_train, sample_weights


def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, batch_size, model_unreg=None):
    """
    Compile and fit the model with optional unregularized pre-training
    
    Args:
        model_given: The model to train
        input_train: Training inputs
        output_train: Training outputs
        epochs: Number of training epochs
        path_checkpoint: Path to save checkpoint
        sample_weights: Sample weights for training
        batch_size: Batch size for training
        model_unreg: Optional unregularized model for pre-training
        
    Returns:
        model_given, history
    """
    # If model_unreg is provided, train the unregularized model first
    if model_unreg is not None:
        # Define loss, metrics and optimizer
        mse_loss = keras.losses.MeanSquaredError()
        metrics = [keras.metrics.MeanSquaredError()] * 5
        opti1 = tf.optimizers.Adam(learning_rate=0.01)
        
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
                                        epochs=epochs,
                                        validation_split=0.0,
                                        callbacks=[es_callback, modelckpt_callback],
                                        shuffle=True,
                                        verbose=0,
                                        sample_weight=sample_weights)

        # Set the initial weights of the regularized model to the final weights of the unregularized model
        model_given.load_weights(path_checkpoint)

    # Train the regularized model
    # Define loss, metrics and optimizer
    mse_loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()] * 5
    opti1 = tf.optimizers.Adam(learning_rate=0.01)
    
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
                              shuffle=True,
                              verbose=0,
                              sample_weight=sample_weights)
    
    return model_given, history


def Stress_calc_shear_principalStretch(inputs):
    """
    Calculate shear stress from principal stretch derivatives
    
    Args:
        inputs: Tuple of (dWdl1, dWdl2, dWdl3, Stretch, Gamma)
        
    Returns:
        Shear stress
    """
    eps = 1e-9  # Small epsilon to avoid division by zero
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
    
    Args:
        model: The full model
        stretch_input: Stretch input data
        gamma_input: Gamma input data
        
    Returns:
        stress_contributions, shear_stress_contributions, term_names
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

    # Build term names dynamically based on the structure
    term_names = ["(I1-3)", "exp(I1-3)-1", "(I1-3)²", "exp((I1-3)²)-1", "(I2-3)", "exp(I2-3)-1", "(I2-3)²", "exp((I2-3)²)-1", "Jᵐ - m ln(J) - 1", "exp(ln(J)²)", "Jᵐ(I1 - 3)", "Jᵐ(I2 - 3)", "λiᵐ - m ln(λi) - 1", "λiᵐ - m ln(λi) - 1"]
    
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


def modelArchitecture(Psi_model):
    """
    Complete model architecture definition
    
    Args:
        Psi_model: The strain energy model
        
    Returns:
        Combined model with uniaxial tension and simple shear outputs
    """
    ## Uniaxial tension model

    # Compute principal stretches
    Stretch_ut = keras.layers.Input(shape=(1,), name='Stretch_ut')
    lambda_2_ut = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Stretch_ut])
    lambda_3_ut = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Stretch_ut])
    # Compute strain energy
    Psi_ut = Psi_model([Stretch_ut ** 2, lambda_2_ut, lambda_3_ut])
    # Compute derivatives of the strain energy with respect to the principal stretches
    compute_derivatives_layer = ComputeDerivativesLayer(Psi_model, name='compute_derivatives_ut')
    dWdl1, dWdl2, dWdl3 = compute_derivatives_layer([Stretch_ut ** 2, lambda_2_ut, lambda_3_ut])
    # Compute stress
    Stress_axial = dWdl1 * 2 * Stretch_ut
    Stress_trans = dWdl2 * 2
    model_ut = keras.models.Model(inputs=[Stretch_ut], outputs=[Stress_axial, Stress_trans, Psi_ut])

    ## Simple shear model
    # Define inputs
    Stretch_ss = keras.layers.Input(shape=(1,), name='Stretch_ss')
    Gamma_ss = keras.layers.Input(shape=(1,), name='Gamma_ss')
    # Compute principal stretches
    lambda_1_sq_ss = keras.layers.Lambda(lambda x: (1 + x[0]**2 + x[1] ** 2) / 2 + ((1 + x[0]**2 + x[1] ** 2) ** 2 / 4 - x[0] ** 2) ** 0.5)([Stretch_ss, Gamma_ss])
    lambda_2_sq_ss = keras.layers.Lambda(lambda x: (1 + x[0]**2 + x[1] ** 2) / 2 - ((1 + x[0]**2 + x[1] ** 2) ** 2 / 4 - x[0] ** 2) ** 0.5)([Stretch_ss, Gamma_ss])
    lambda_3_sq_ss = keras.layers.Lambda(lambda x: 1.0 + x[0] * 0)([Gamma_ss])
    # Compute strain energy
    Psi_ss = Psi_model([lambda_1_sq_ss, lambda_2_sq_ss, lambda_3_sq_ss])
    # Compute derivatives of the strain energy with respect to the principal stretches
    compute_derivatives_layer = ComputeDerivativesLayer(Psi_model, name='compute_derivatives_ss')
    dWdl1, dWdl2, dWdl3 = compute_derivatives_layer([lambda_1_sq_ss, lambda_2_sq_ss, lambda_3_sq_ss])
    # Compute stress
    Stress_shear = keras.layers.Lambda(function=Stress_calc_shear_principalStretch,
                                       name='Stress_shear')([dWdl1, dWdl2, dWdl3, Stretch_ss, Gamma_ss])
    model_ss = keras.models.Model(inputs=[Stretch_ss, Gamma_ss], outputs=[Stress_shear, Psi_ss])

    # Combine the models
    model_combined = keras.models.Model(inputs=[Stretch_ut, Stretch_ss, Gamma_ss], outputs=[Stress_axial, Stress_trans, Psi_ut, Stress_shear, Psi_ss])
    return model_combined

