import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Layer
from keras import Model
import keras

from src.utils import *
import sympy
from sympy import Symbol, lambdify, diff, exp

from src.CANN.models import SingleInvNet_symbolic, SingleInvNetI4_symbolic
import re

from src.CANN.models import I4_theta

from src.CANN.util_functions import reshape_input_output_mesh, traindata, makeDIR, Compile_and_fit
from src.CANN.models import get_max_inv_mesh, calculate_I4theta_max


symb_wstar = Symbol("wstar") # symbol for the inner weight w*
symb_inv = Symbol("inv") # symbol for the input invariant (already shifted to be zero in the reference configuration)
initializer_1 = tf.keras.initializers.RandomUniform(minval=0., maxval=1)



def get_model_weights():
    model_type = "correlated"
    alpha_in = 0.1
    stretches, stresses, n_samples = load_data()
    model_id = get_model_id(model_type, alpha_in)

    # Reshape stretch and stress inputs
    stretches = np.float64(stretches)
    stresses = np.float64(stresses)
    lam_ut_all = [[stretches.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]

    #### Compute scale factor
    terms = 17  # 4 x I1, 4 x I2, 3 x I4w, 3 x I4s, 3 x I4theta
    # Compute scale_factor (number to multiply all outputs by to help training stability)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))  # Reshape stress
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))  # Reshape stretch
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1))
    scale_factor = (np.sum(scale_factors) / terms * 2)


    model = ortho_cann_3ff_gcann(lam_ut_all, P_ut_all, n_samples, alpha_in,
                                 True, 0.5, not (model_type == "correlated"), True)
    modelFit_mode = "02356789bcde"
    path2saveResults = '../Results/' + modelFit_mode
    Save_weights = path2saveResults + f'/weights_{model_id}'
    model.load_weights(Save_weights)
    model_weights = model.get_weights()
    disp_equation_weights_gcann(model_weights, lam_ut_all, P_ut_all, "0123456789")
    return model_weights, scale_factor

def generate_synth_data(model_weights, n_lambdas=20, n_samples=1000, scale_factor=1.0):
    n_terms = len(model_weights) // 2

    pct_complete = np.linspace(0, 1, n_lambdas)
    lambda_max = 0.1
    all_stretches = [1 + lambda_max * pct_complete, 1 + lambda_max * pct_complete, 1 + lambda_max * pct_complete, 1 + lambda_max * pct_complete / 2, 1 + pct_complete * 0]
    x_stretches = np.concatenate(all_stretches)
    y_stretches = np.concatenate(all_stretches[::-1])
    stretches = np.stack([x_stretches, y_stretches], axis=1)
    stretches = np.stack([stretches] * n_samples, axis=0)
    stretches = np.stack([stretches, stretches])

    # Sample model weights
    noise = np.random.standard_normal((n_terms, n_samples))
    weights_inner = np.array(model_weights[0:-1:2])
    weights_outer_mean = np.array(model_weights[1:-1:2]) # to account for Lp
    weights_L = model_weights[-1]# Change variance scale here
    weights_outer_sampled = weights_outer_mean ** 2 * (1 + weights_L @ noise) ## Force to be positive
    weights_inner_sampled = weights_inner + 0 * weights_outer_sampled

    # residual = weights_outer_sampled[0, :] / weights_outer_mean[0] - 1
    weights_sampled = np.stack([weights_inner_sampled, weights_outer_sampled], axis=1).reshape((-1, n_samples))

    ## Build model
    stretches = np.float64(stretches)
    lam_ut_all = [[stretches.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    lam_ut_single_sample = [[stretches[i, 0, :, k].flatten() for k in range(2)] for i in range(2)]
    # model = ortho_cann_3ff_gcann(lam_ut_all, None, n_samples, 0.0,
    #                              True, 0.5, True, False, scale_factor=1.0)
    model = ortho_cann_3ff_gcann(lam_ut_all, None, n_samples, 0.0,
                                 True, 1.0, True, False, scale_factor=scale_factor)
    stresses = np.zeros_like(stretches)
    for i in range(n_samples):
        # temp_weights = [np.array([weights_sampled[j, i]]) for j in range(n_terms * 2)] + [np.zeros((n_terms, n_terms))]
        temp_weights = [np.array([weights_sampled[j, i]]) for j in range(n_terms * 2)] + [np.zeros((n_terms, n_terms))]

        # model.set_weights(temp_weights)
        model.set_weights(temp_weights)

        P_ut_single_sample = model.predict(lam_ut_single_sample).reshape((-1, 2, 2, 2))[:, :, :, 0]  # N x 2 x 2
        # P_ut_single_sample_l1 = model_l1.predict(lam_ut_single_sample).reshape((-1, 2, 2, 2))[:, :, :, 0]  # N x 2 x 2
        # print("************HERE******")
        # print(np.max(np.abs(P_ut_single_sample_l1 - P_ut_single_sample)))
        print(P_ut_single_sample.shape)
        stresses[:, i, :, :] = P_ut_single_sample.transpose(1, 0, 2) # reshape to 2 x 1 x 100 x 2

    print(stresses.shape)

    return stretches.reshape((2, -1, 2)), stresses, n_samples


# Function for converting symbolic expression of Psi to its derivative (with normalization)
# Psi_expr is a symbolic expression that expresses one term of the strain energy in terms of the
# inner weight symb_wstar and the input invariant symb_inv
# invs_max is an array of the maximum value of the input invariant for each loading mode (length 15)
# returns derivative of input expression as lambda expression, normalized if desired
def get_stress_expression(Psi_expr, invs_max, should_normalize=True):
    invs_max = tf.cast(invs_max, tf.float32)
    # Compute max value of invariant across all loading modes
    invs_max_max = np.max(invs_max) if should_normalize else 1.0
    # Differentiate strain energy
    dPsi_expr = diff(Psi_expr, symb_inv)
    # Convert strain energy and its derivative to tf lambda expressions
    Psi_func = lambdify([symb_inv, symb_wstar], Psi_expr, "tensorflow")
    dPsi_func = lambdify([symb_inv, symb_wstar], dPsi_expr, "tensorflow")

    # Compute normalized stress lambda expression
    epsilon = 1e-8

    # first we divide the invariant by its maximum value and pass this into our expression for deriv of strain energy
    # then we divide this quantity by the integral of the stress across all loading modes, or equivalently the sum of the maximum strain energy across all loading modes
    # This process ensures that the scaling of all w*s and ws is consistent and helps both with stability of training,
    # and ensures that a regularization term penalizes all terms fairly
    dPsi_out = lambda inv, wstar : 1 / invs_max_max * dPsi_func(inv / invs_max_max, wstar) / (tf.reduce_sum(Psi_func(invs_max / invs_max_max, wstar)) + epsilon)
    out_func = dPsi_out if should_normalize else dPsi_func
    return out_func

# Custom layer that maps invariant I to derivative dPsi / dI, where Psi is specified by a sympy expression
class SingleTermStress(keras.layers.Layer):
    # kernel initializer determines how wstar is initialized
    # Psi_expr is the expression for this term of the strain energy
    # p and alpha are hyperparameters for lp regularization
    # invs_max is an array of the maximum value of the input invariant for each loading mode (length 15)
    def __init__(self, kernel_initializer, Psi_expr, weight_name, should_normalize, p, alpha, invs_max):
        super().__init__()
        self.dPsi_func = get_stress_expression(Psi_expr, invs_max, should_normalize) # Compute function for deriv of strain energy from symbolic expression
        self.weight_name = weight_name
        self.p = p
        self.kernel_initializer = kernel_initializer
        self.alpha = alpha
        self.I1maxmax = np.max(invs_max)

    # Necessary for saving the model to work
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "P_func": self.dPsi_func,
            "weight_name": self.weight_name,
            "p": self.p,
            "alpha": self.alpha
        })
        return config

    # Define weights
    def build(self, input_shape):
        # Wstar is nonnegative and initialized as specified in constructior
        self.wstar = self.add_weight(
            shape=(1, ),
            initializer=self.kernel_initializer,
            constraint=keras.constraints.NonNeg(),
            trainable=True,
            name=self.weight_name + '1'
        )
        # W is nonnegative and has l1 regularization applied with weight alpha, initialized with uniform distribution
        self.w_mu = self.add_weight(
            shape=(1,),
            initializer=initializer_1,
            constraint= keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(self.alpha),
            trainable=True,
            name=self.weight_name + '2'
        )
    def call(self, inputs):
        # In order to apply Lp regularization, we apply L1 regularization on w_mu and raise w_mu to the power of 1/p before using it
        # The output is the contribution to the mean derivative of the strain energy for this term
        mean = self.w_mu ** (1 / self.p) * self.dPsi_func(inputs, self.wstar)
        return mean


# Define activation functions for use in strain energy symbolic expressions
def identity(x):
    return x
def activation_exp(x):
    return exp(x) - 1.0
def activation_exp_I4(x): # This activation is used for I4 terms to ensure a valid strain energy (zero stress at zero deformation)
    return exp(x) - x - 1.0


# Compute terms for I1 or I2
# inv specifies value of the input invariant (shifted so it is zero in reference configuration)
# term_idx is the index of the first term corresponding to this invariant, used so all weights get distinct numbers
# invs_max is an array of the maximum value of the input invariant for each loading mode (length 15)
# inv_grad is the derivative of the input invariant with respect to the x and y stretch
# p and alpha are hyperparameters for lp regularization
def single_inv_stress_gcann(inv, term_idx, invs_max, inv_grad, should_normalize, p, alpha):
    # Iterate over 2 activations and 2 exponents (1st / 2nd power)
    activations = [identity, activation_exp]
    n_exps = 2
    # Create symbolic expressions for strain energy term
    Psi_funcs = [activations[j](symb_wstar * symb_inv ** (i + 1)) for i in range(n_exps) for j in range(len(activations))]
    initializers = [initializer_1] * 4 # Initialize all inner weights w* with uniform distribution
    # Compute term as product of inv_grad (dI/dlambda) and SingleTermStress (dPsi/dI), +0.0 ensures proper weight order
    terms = [inv_grad * SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, invs_max)(inv) + 0.0 for i in range(len(Psi_funcs))]
    return terms

# Compute terms for I4w or I4s, same inputs as single_inv_stress_gcann
def single_inv_I4_stress_gcann(inv, term_idx, invs_max, inv_grad, should_normalize, p, alpha):
    # Iterate over 3 activations
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    # Create symbolic expressions for strain energy
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    initializers = [initializer_1] * 3# Initialize all inner weights with uniform distribution
    # Compute term as product of inv_grad (dI/dlambda) and SingleTermStress (dPsi/dI), +0.0 ensures proper weight order
    terms = [inv_grad * SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, invs_max)(inv) + 0.0 for i in range(len(Psi_funcs))]
    return terms

# Compute terms for I4theta
# I4_plus and I4_minus are the values of the I4 invariants corresponding to the +/- theta directions
# inv_grad_plus and inv_grad_minus are the derivatives of the I4_plus and I4_minus with respect to the x and y stretch
# Remaining inputs are the same as single_inv_stress_gcann
def single_inv_I4_theta_stress_gcann(I4_plus, I4_minus, term_idx, invs_max, inv_grad_plus, inv_grad_minus, should_normalize, p, alpha):
    # Iterate over 3 activations
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    # Create symbolic expressions for strain energy
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    initializers = [initializer_1] * 3 # Initialize all inner weights with uniform distribution
    # Compute SingleTermStresses (dPsi/dI4theta+ and dPsi/dI4theta-)
    layers = [SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, invs_max) for i in range(len(Psi_funcs))]
    # Compute term as product of inv_grad (dI/dlambda) and SingleTermStress (dPsi/dI)
    # Then sum contribution from I4theta+ and I4theta-
    terms = [layer(I4_plus) * inv_grad_plus + layer(I4_minus) * inv_grad_minus for layer in layers]
    return terms

# Create GCANN model
# lam_ut_all, P_ut_all, and n_samples - stretch and stress data
# alpha, should_normalize, and p - regularization hyperparameters
# independent - if true all weights are independent, if false they are correlated
# constrained - if true, std dev of each weight is upper bounded by the mean of that weight. Setting constrained to false with regularization is not advised as regularization may not promote sparsity.
def ortho_cann_3ff_gcann(lam_ut_all, P_ut_all, n_samples, alpha, should_normalize, p, independent, constrained, scale_factor=None):

    terms = 17 # 4 x I1, 4 x I2, 3 x I4w, 3 x I4s, 3 x I4theta
    # Compute invs_max (maximum value of each invariant, incorporated into strain energy to ensure comparable initial stability)
    lam_ut_norepeats = reshape_input_output_mesh([[x[0:int(len(x) / n_samples)] for x in y] for y in lam_ut_all]) # stretch values for first sample
    invs_max = get_max_inv_mesh(lam_ut_norepeats, "0123456789") # compute maximum invariants

    # Divide stress integral by number of terms and multiply by 2 so if each term has an integral of 0.5 in expectation then the stress will be the correct magnitude
    I4theta_max = calculate_I4theta_max(invs_max)

    # Compute scale_factor (number to multiply all outputs by to help training stability)
    if scale_factor is None:
        P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))  # Reshape stress
        lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))  # Reshape stretch
        scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped,
                                                                                                     axis=-1))  # compute the integral of the measured stress
        scale_factor = (np.sum(scale_factors) / terms * 2)


    # Create model with inputs as invariants and derivative of invariants wrt x and y stretches
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')
    dI1_in = tf.keras.Input(shape=(2,), name='dI1')
    dI2_in = tf.keras.Input(shape=(2,), name='dI2')
    dI4f_in = tf.keras.Input(shape=(2,), name='dI4f')
    dI4n_in = tf.keras.Input(shape=(2,), name='dI4n')
    dI8fn_in = tf.keras.Input(shape=(2,), name='dI8fn')

    # Put invariants in the reference configuration
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4n_in)

    # Compute I4 theta
    I4theta, I4negtheta = I4_theta()([I4f_in, I4n_in, I8fn_in])
    I4theta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4theta)
    I4negtheta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4negtheta)

    # Compute derivative of I4theta wrt x and y stretches
    theta = np.pi / 3
    dI4theta = dI4f_in * (np.cos(theta)) ** 2 + dI4n_in * (np.sin(theta)) ** 2  + dI8fn_in * np.sin(2 * theta)
    dI4negtheta = dI4f_in * (np.cos(theta)) ** 2 + dI4n_in * (np.sin(theta)) ** 2 - dI8fn_in * np.sin(2 * theta)

    # Compute stress terms corresponding to each invariant
    I1_terms = single_inv_stress_gcann(I1_ref, 0, invs_max[:, 0], dI1_in, should_normalize, p, alpha)
    I2_terms = single_inv_stress_gcann(I2_ref, 4, invs_max[:, 1], dI2_in, should_normalize, p, alpha)
    I4f_terms = single_inv_I4_stress_gcann(I4f_ref, 8, invs_max[:, 2], dI4f_in, should_normalize, p, alpha)
    I4n_terms = single_inv_I4_stress_gcann(I4n_ref, 11, invs_max[:, 4], dI4n_in, should_normalize, p, alpha)
    I4theta_terms = single_inv_I4_theta_stress_gcann(I4theta_ref, I4negtheta_ref, 14, I4theta_max, dI4theta, dI4negtheta, should_normalize, p, alpha)

    # Concatenate (+ is concatenation not addition)
    all_terms = tf.stack(I1_terms + I2_terms + I4f_terms + I4n_terms + I4theta_terms, axis=-1) * scale_factor

    # Mean is sum of all terms
    mean_out = tf.reduce_sum(all_terms, axis=-1)
    # Multiply the terms by L (lower triangular) then take sum of squares, where Sigma = LL^T is the covariance matrix
    # If terms are independent then L must be diagonal
    var_constraint = (DiagonalNonnegativeLessThanOne() if constrained else DiagonalNonnegative()) if independent else ValidCovariance()
    var_out = tf.reduce_sum(Dense(all_terms.shape[2], kernel_initializer=initializer_1, kernel_constraint=var_constraint, use_bias=False)(all_terms) ** 2, axis=-1)
    # Create model to map from invariants and invariant derivatives to stress means and variances
    P_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in, dI1_in, dI2_in, dI4f_in, dI4n_in, dI8fn_in],
                                   outputs=[mean_out[:, 0], var_out[:, 0], mean_out[:, 1], var_out[:, 1]], name='P_model')


    # Create 0-90 model
    # Inputs are warp and shute stretch
    Stretch_w = keras.layers.Input(shape=(1,),
                                   name='Stretch_w')
    Stretch_s = keras.layers.Input(shape=(1,),
                                   name='Stretch_s')
    # Compute invariants
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_w, Stretch_s])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_w, Stretch_s])
    I4w = keras.layers.Lambda(lambda x: x ** 2)(Stretch_w)
    I4s = keras.layers.Lambda(lambda x: x ** 2)(Stretch_s)
    I8ws = keras.layers.Lambda(lambda x: x ** 0 - 1)(Stretch_s)

    # Compute derivatives of invariants wrt warp and shute stretch
    Stretch_z = keras.layers.Lambda(lambda x: 1 / (x[0] * x[1]))([Stretch_w, Stretch_s])
    dI1 = tf.keras.layers.concatenate([2 * (Stretch_w - Stretch_z * Stretch_z / Stretch_w), 2 * (Stretch_s - Stretch_z*Stretch_z / Stretch_s)], axis=-1)
    dI2 = tf.keras.layers.concatenate([2 * (Stretch_w*Stretch_s*Stretch_s - 1/(Stretch_w*Stretch_w*Stretch_w)), 2 * (Stretch_w*Stretch_w*Stretch_s - 1/(Stretch_s*Stretch_s*Stretch_s))], axis=-1)
    dI4w = tf.keras.layers.concatenate([2 * Stretch_w, 0 * Stretch_w], axis=-1)
    dI4s = tf.keras.layers.concatenate([0 * Stretch_s, 2 * Stretch_s], axis=-1)
    dI8ws = tf.keras.layers.concatenate([0 * Stretch_w, 0 * Stretch_w], axis=-1)

    # Apply P_model to invariants and derivatives
    outputs = P_model([I1, I2, I4w, I4s, I8ws, dI1, dI2, dI4w, dI4s, dI8ws])
    model_90 = keras.models.Model(inputs=[Stretch_w, Stretch_s], outputs=outputs)

    # Create 0-90 model
    # Inputs are x and y stretch
    Stretch_x = keras.layers.Input(shape=(1,),
                                   name='Stretch_x')
    Stretch_y = keras.layers.Input(shape=(1,),
                                   name='Stretch_y')
    Stretch_z = keras.layers.Lambda(lambda x: 1 / (x[0] * x[1]))([Stretch_x, Stretch_y])

    # Compute invariants
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_x, Stretch_y])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_x, Stretch_y])
    I4w = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I4s = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I8ws = keras.layers.Lambda(lambda x: (x[0] ** 2 - x[1] ** 2) / 2)([Stretch_x, Stretch_y])

    # Compute derivatives of invariants wrt warp and shute stretch
    dI1 = tf.keras.layers.concatenate(
        [2 * (Stretch_x - Stretch_z*Stretch_z / Stretch_x), 2 * (Stretch_y - Stretch_z * Stretch_z / Stretch_y)],
        axis=-1)
    dI2 = tf.keras.layers.concatenate([2 * (Stretch_x * Stretch_y * Stretch_y - 1 / (Stretch_x * Stretch_x * Stretch_x)),
                                 2 * (Stretch_x * Stretch_x * Stretch_y - 1 / (Stretch_y * Stretch_y * Stretch_y))],
                                axis=-1)
    dI4w = tf.keras.layers.concatenate([Stretch_x, Stretch_y], axis=-1)
    dI4s = tf.keras.layers.concatenate([Stretch_x, Stretch_y], axis=-1)
    dI8ws = tf.keras.layers.concatenate([Stretch_x, -Stretch_y], axis=-1)

    # Apply P_model to invariants and derivatives
    outputs = P_model([I1, I2, I4w, I4s, I8ws, dI1, dI2, dI4w, dI4s, dI8ws])
    model_45 = keras.models.Model(inputs=[Stretch_x, Stretch_y], outputs=outputs)

    # Combine two models
    models = [model_90, model_45]
    inputs = [model.inputs for model in models]
    outputs = [model.outputs for model in models]
    outputs = tf_stack(flatten(outputs),axis=1) # make output a single tensor so we can apply our custom loss
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# Train GCANNs based on stretch and stress data provided
# Returns trained model as well as reshaped stretch and stress inputs
def train_gcanns(stretches, stresses, n_samples, modelFit_mode ="0123456789abcde", should_train=False, model_type="independent", alpha_in=0, base_path="../Results"):
    model_id = get_model_id(model_type, alpha_in)

    # Reshape stretch and stress inputs
    stretches = np.float64(stretches)
    stresses = np.float64(stresses)
    lam_ut_all = [[stretches.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]

    # Define hyperparameters
    alphas =  [0] if model_type == "unregularized" else [0, alpha_in]
    ps = [1.0] if model_type == "unregularized" else [1.0 , 0.5]
    epochs = 2000
    batch_size = 1024
    gamma_ss = []
    P_ss = []
    last_weights = []

    # Iterate over regularization values
    for i in range(len(alphas)):
        path2saveResults = base_path + "/" + modelFit_mode
        makeDIR(path2saveResults)
        Save_path = path2saveResults + f'/model_{model_id}.h5'
        Save_weights = path2saveResults + f'/weights_{model_id}'
        path_checkpoint = path2saveResults + f'/best_weights_{model_id}'

        if i < len(alphas) - 1 and not should_train: # If not training, skip to last alpha and load model
            continue

        # Build model
        model = ortho_cann_3ff_gcann(lam_ut_all, P_ut_all, n_samples, alphas[i],
                                     True, ps[i], not (model_type == "correlated"), i > 0)


        # If not the first iteration, set initial weights to be final weights from previous iteration
        if i > 0 and should_train:
            model.set_weights(last_weights)

        # Load training data
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model, lam_ut_all, P_ut_all,
                                                                           model, gamma_ss, P_ss, model, 0)
        # print(sample_weights)
        # print(output_train)
        # output_train = [[output_train[i][j] * (sample_weights[i][j] > 0) for j in range(len(output_train[i]))] for i in range(len(output_train))]
        # print(output_train)
        if should_train:
            # Train model
            model_given, history, weight_hist_arr = Compile_and_fit_gcann(model_given, input_train, output_train, epochs,
                                                                          path_checkpoint,
                                                                          sample_weights, batch_size)

            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
            tf.keras.models.save_model(model, Save_path, overwrite=True)
            model_given.save_weights(Save_weights, overwrite=True)
            last_weights = model_given.get_weights()

            if i < len(ps) - 1: # If not last iteration, update last_weights so it is correct for the next value of p
                p_ratio = ps[i + 1] / ps[i]
                last_weights = [last_weights[i] ** (p_ratio if (i % 2 == 1) else 1.0) for i in range(len(last_weights))]

        else:
            model.load_weights(Save_weights)

    # Uncomment this to display names for each weight
    # names = [weight.name for layer in model_given.layers for weight in layer.weights]
    # print(names)


    model_weights = model.get_weights()
    nonzero_terms = sum([x > 0 for x in model_weights[1::2]])
    print(f"Nonzero Terms: {nonzero_terms}") # Print number of nonzero terms to determine effect of regularization

    return model_given, lam_ut_all, P_ut_all



# Display strain energy equation and weights (with variances)
def disp_equation_weights_gcann(weights, lam_ut_all, P_ut_all, modelFit_mode, scale_factor=None):
    sf, terms, weights_var = ortho_gcann_4ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode, scale_factor)

    n_mus = 0
    n_abs = 0
    eqn = ""
    # Iterate over terms
    for term, w_var in zip(terms, weights_var):
        term = term.item()
        if term == 0: # Skip any zero terms
            continue
        gain = term.args[0] * sf * 2 ## * 2 accounts for 1/2 we add later
        if term.args[1].func == sympy.core.add.Add: # Print exponential weights
            try:
                exponent = term.args[1].args[-1].args[0][0].args[0]
            except:
                # print(term)
                continue
            n_abs += 1
            gain *= exponent
            std_dev = float(gain * np.sqrt(w_var))
            print(gain)
            print(std_dev)
            print(exponent)
            print(f"$a_{n_abs} = {format_2sigfigs(gain)}\\pm{format_2sigfigs(std_dev)}" + "\\text{ kPa}$\\newline")
            print(f"$b_{n_abs} = {format_2sigfigs(exponent)}$\\newline")
            term_str = f"a_{n_abs}(" + str(term.args[1]) + f") / b_{n_abs}"
            term_str = re.sub(r'\d+\.\d\d\d+', f"b_{n_abs}", term_str)

        else: # Print non-exponential weights
            n_mus += 1
            term_str = f"\mu_{n_mus}" + str(term.args[1])
            std_dev = float(gain * np.sqrt(w_var))
            print(gain)
            print(std_dev)
            print(f"$\mu_{n_mus} = {format_2sigfigs(gain)}\\pm{format_2sigfigs(std_dev)}" +  "\\text{ kPa}$\\newline")


        # Perform substitutions so it renders correctly
        term_str = term_str.replace("**", "^")
        term_str = term_str.replace("I1", "(I_1 - 3)")
        term_str = term_str.replace("I2", "(I_2 - 3)")
        term_str = term_str.replace("I4w", "(I_{4w} - 1)")
        term_str = term_str.replace("I4s", "(I_{4s} - 1)")
        term_str = term_str.replace("1.0", "1")
        term_str = term_str.replace("exp", "\exp")
        term_str = term_str.replace("*", "")
        term_str = term_str.replace("[", "")
        term_str = term_str.replace("]", "")
        term_str = "&+&\\frac{1}{2}" + term_str
        term_str += "\n"
        # Append term to equation
        if "I4theta" in term_str:
            eqn += term_str.replace("I4theta", "(I_{4s_I} - 1)")
            eqn += term_str.replace("I4theta", "(I_{4s_{II}} - 1)")

        else:
            eqn += term_str

    # Print equation
    print("\psi &=& " + eqn[3:])

# Construct symbolic expression for GCANN
def ortho_gcann_4ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode, scale_factor=None):
    terms = len(weights) // 2

    # Compute scale factor and Imaxes
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    if scale_factor is None:
        P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
        scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
        scale_factor = np.sum(scale_factors) / terms * 2
    I4theta_max = calculate_I4theta_max(Is_max)

    # Separate weights into CANN weights and variances
    weights_mean = weights[:-1]# get only wstar and w
    weights_var = tf.reduce_sum(tf.square(weights[-1]), axis=-1).numpy().tolist() # get only wsigma
    print(weights)
    # Create list of all terms
    output = SingleInvNet_symbolic(Symbol("I1"), weights_mean[0:8],  I1s_max=(Is_max[:, 0]))
    output += SingleInvNet_symbolic(Symbol("I2"), weights_mean[8:16],  I1s_max=(Is_max[:, 1]))
    output += SingleInvNetI4_symbolic(Symbol("I4w"), weights_mean[16:22],  I1s_max=(Is_max[:, 2]))
    output += SingleInvNetI4_symbolic(Symbol("I4s"), weights_mean[22:28],  I1s_max=(Is_max[:, 4]))
    output += SingleInvNetI4_symbolic(Symbol("I4theta"), weights_mean[28:],  I1s_max=(I4theta_max))

    # Output scale factor, list of terms, and list of wsigmas
    return scale_factor, output, weights_var



# Perform training of model, return fit model, training history, and weight history
def Compile_and_fit_gcann(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, batch_size):

    opti1 = tf.optimizers.Adam(learning_rate=0.001)
    # Note custom loss (negative log likelihood) is used
    model_given.compile(loss=NLL,
                        optimizer=opti1,
                        metrics=[NLL])
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

    # Reshape output_train to be a single tf array
    output_temp = tf.keras.backend.stack(flatten(output_train) + flatten(sample_weights),axis=1) #
    output_temp = tf.cast(output_temp, tf.float32)

    history = model_given.fit(input_train,
                              output_temp,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              callbacks=[es_callback, modelckpt_callback, weight_hist_callback],
                              shuffle=True,
                              verbose=1)

    return model_given, history, weight_hist_arr

# Compute negative log likelihood
def NLL(y_true, y_pred):
    eps = 1e-6
    # Separate mean and variance from prediction
    means = y_pred[:, 0::2]
    vars = y_pred[:, 1::2]

    # separate sample weights from measured stresses for true data
    y_wgts = y_true[:, 4:]
    y_true = y_true[:, 0:4]
    # Compute negative log likelihood for a normal distribution
    errors = 0.5 * (tf.math.log(2 * np.pi * (vars + eps)) + tf.math.square(y_true - means) / (vars + eps)) * y_wgts


    # errors_reshape = tf.reshape(errors, shape=(5, 5, 100, 2, 2))
    # print("Error1: ")
    # print(tf.reduce_mean(errors_reshape, axis=[0, 2])) ### test, orient, axis
    return tf.reduce_sum(errors, axis=1)


class DiagonalNonnegativeLessThanOne(keras.constraints.Constraint):
    """Constrains the weights to be diagonal, nonnegative, and at most 1.0
    """
    def __call__(self, w):
        N = tf.keras.backend.int_shape(w)[-1]
        m = tf.eye(N)
        return m * tf.clip_by_value(w, 0, 1.0)

class DiagonalNonnegative(keras.constraints.Constraint):
    """Constrains the weights to be diagonal and nonnegative
    """
    def __call__(self, w):
        N = tf.keras.backend.int_shape(w)[-1]
        m = tf.eye(N)
        return m * tf.clip_by_value(w, 0, np.inf)

# Constrains the matrix w to be lower diagonal and the norm of each row less than or equal to 1.0 such that ww^T is a valid covariance matrix
# with diagonal entries less than or equal to 1
class ValidCovariance(keras.constraints.Constraint):
    def __call__(self, w):
        lower = tf.linalg.band_part(w, -1, 0)
        norms = tf.math.sqrt(
            tf.reduce_sum(tf.square(lower), axis=-1, keepdims=True)
        )
        desired_norms = tf.clip_by_value(norms, 0, 1.0)
        return lower * (desired_norms / (tf.keras.backend.epsilon() + norms))


