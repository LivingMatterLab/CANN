#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:40:23 2022

@author: kevinlinka
"""
import sympy.core.add

# All the models that can be used for CANN training.
from src.CANN.util_functions import *
import numpy as np
from tensorflow import keras
from src.CANN.cont_mech import *
import sympy as sp
import re

initializer_1 = tf.keras.initializers.RandomUniform(minval=0., maxval=1)
initializer_zero = 'zeros'
initializer_log = tf.keras.initializers.RandomUniform(minval=0.,
                                                      maxval=0.001)
initializer_exp = tf.keras.initializers.RandomUniform(minval=0.,
                                                      maxval=1)  # worked off and on, starts with huge residual


# Self defined activation functions for exp
def activation_Exp(x):
    return 1.0 * (tf.math.exp(x) - 1.0)
def activation_Exp_I4(x):
    return 1.0 * (tf.math.exp(x) - x - 1)

def activation_Exp_sp(x):
    return 1.0 * (sp.exp(x) - 1.0)
def activation_Exp_I4_sp(x):
    return 1.0 * (sp.exp(x) - x - 1)


# Define network block
def SingleInvNet(I1_ref, idi, should_normalize, p, alpha, I1s_max):
    # Layer 1. order
    I_1_w11 = 1 * NormalizedDense(kernel_initializer=initializer_1, activation=None, weight_name='w' + str(1 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max, is_exp=False)(I1_ref)
    I_1_w21 = 1 * NormalizedDense(kernel_initializer=initializer_exp, activation=activation_Exp, weight_name='w' + str(2 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max)(I1_ref)

    # Layer 2. order
    I_1_w41 = 1 * NormalizedDense(kernel_initializer=initializer_1, activation=None, weight_name='w' + str(3 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, is_exp=False, I1s_max=I1s_max ** 2)(tf.math.square(I1_ref))
    I_1_w51 = 1 * NormalizedDense(kernel_initializer=initializer_exp, activation=activation_Exp, weight_name='w' + str(4 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max ** 2)(tf.math.square(I1_ref))


    collect = [I_1_w11, I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out

def SingleInvNet_I4(I1_ref, idi, should_normalize, p, alpha, I1s_max):
    I_1_w21 = 1 * NormalizedDense(kernel_initializer=initializer_exp, activation=activation_Exp_I4, weight_name='w' + str(1 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max)(I1_ref)
    I_1_w41 = 1 * NormalizedDense(kernel_initializer=initializer_1, activation=None, weight_name='w' + str(2 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, is_exp=False, I1s_max=I1s_max ** 2)(tf.math.square(I1_ref))
    I_1_w51 = 1 * NormalizedDense(kernel_initializer=initializer_exp, activation=activation_Exp, weight_name='w' + str(3 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max ** 2)(tf.math.square(I1_ref))
    collect = [I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out


def SingleInvNet_I4theta(I4theta, I4negtheta, idi, should_normalize, p, alpha, I1s_max):
    exp_layer = NormalizedDense(kernel_initializer=initializer_exp, activation=activation_Exp_I4, weight_name='w' + str(1 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max)
    exp_term = keras.layers.Lambda(lambda x: x[0] + x[1])([exp_layer(I4theta), exp_layer(I4negtheta)])

    quad_layer = NormalizedDense(kernel_initializer=initializer_1, activation=None, weight_name='w' + str(2 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, is_exp=False, I1s_max=I1s_max ** 2)
    quad_term = keras.layers.Lambda(lambda x: x[0] + x[1])(
        [quad_layer(tf.math.square(I4theta)), quad_layer(tf.math.square(I4negtheta))])

    exp_quad_layer = NormalizedDense(kernel_initializer=initializer_exp, activation=activation_Exp, weight_name='w' + str(3 + idi),
                              should_normalize=should_normalize, p=p, alpha=alpha, I1s_max=I1s_max ** 2)
    exp_quad_term = keras.layers.Lambda(lambda x: x[0] + x[1])(
        [exp_quad_layer((tf.math.square(I4theta))), exp_quad_layer((tf.math.square(I4negtheta)))])

    collect = [exp_term, quad_term, exp_quad_term]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out

def ortho_cann_2ff_2term(terms):
    return lambda lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p: \
        ortho_cann_2ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, two_term=True, terms=terms)

def ortho_cann_3ff_2term(terms):
    return lambda lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p: \
          ortho_cann_3ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, two_term=True, terms=terms)


# Orthotropic CANN with fibers in the Warp and Shute directions
def ortho_cann_2ff_noiso(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, two_term=False, terms=[]):
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors)

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4n_in)
    I8fn_ref = keras.layers.Lambda(lambda x: x ** 2)(I8fn_in)

    # I1_out = SingleInvNet(I1_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 0]))
    # I2_out = SingleInvNet(I2_ref, 6, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 1]))
    I4f_out = SingleInvNet_I4(I4f_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 2]))
    I4n_out = SingleInvNet_I4(I4n_ref, 6, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 4]))
    # I8fn_out = SingleInvNet(I8fn_ref, 24, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 6] ** 2))


    ALL_I_out = [I4f_out,I4n_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    terms = ALL_I_out.get_shape().as_list()[1]
    ALL_I_out_scaled = keras.layers.Lambda(lambda x: (x / terms * 2 * scale_factor))(ALL_I_out)


    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in], outputs=[ALL_I_out_scaled], name='Psi')



    return Psi_model, terms  # 32 terms

def ortho_cann_2ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, two_term=False, terms=[]):
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors)

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4n_in)
    I8fn_ref = keras.layers.Lambda(lambda x: x ** 2)(I8fn_in)

    I1_out = SingleInvNet(I1_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 0]))
    I2_out = SingleInvNet(I2_ref, 6, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 1]))
    I4f_out = SingleInvNet_I4(I4f_ref, 12, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 2]))
    I4n_out = SingleInvNet_I4(I4n_ref, 18, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 4]))
    # I8fn_out = SingleInvNet(I8fn_ref, 24, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 6] ** 2))


    ALL_I_out = [I1_out,I2_out,I4f_out,I4n_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    if two_term:
        ALL_I_out = tf.keras.layers.concatenate([ALL_I_out[:, term:(term+1)] for term in terms], axis=1)

    terms = ALL_I_out.get_shape().as_list()[1]
    ALL_I_out_scaled = keras.layers.Lambda(lambda x: (x / terms * 2 * scale_factor))(ALL_I_out)


    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in], outputs=[ALL_I_out_scaled], name='Psi')



    return Psi_model, terms  # 32 terms


# Orthotropic CANN with fibers in the Warp, theta, and negative theta directions
def ortho_cann_3ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, two_term=False, terms=[]):
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))


    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')
    I4theta, I4negtheta = I4_theta()([I4f_in, I4n_in, I8fn_in])

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4n_in)
    I4theta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4theta)
    I4negtheta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4negtheta)


    I1_out = SingleInvNet(I1_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 0]))
    I2_out = SingleInvNet(I2_ref, 6, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 1]))
    I4f_out = SingleInvNet_I4(I4f_ref, 12, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 2]))
    I4n_out = SingleInvNet_I4(I4n_ref, 18, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 4]))

    I4theta_max = calculate_I4theta_max(Is_max)
    I4theta_out = SingleInvNet_I4theta(I4theta_ref, I4negtheta_ref, 24, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=I4theta_max)
    # I8fn_out = SingleInvNet(I8fn_ref, 24, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 6] ** 2))


    ALL_I_out = [I1_out, I2_out, I4f_out,I4n_out, I4theta_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    if two_term:
        ALL_I_out = tf.keras.layers.concatenate([ALL_I_out[:, term:(term+1)] for term in terms], axis=1)
    terms = ALL_I_out.get_shape().as_list()[1]

    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped,
                                                                                                 axis=-1))  # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors) / terms * 2
    ALL_I_out_scaled = keras.layers.Lambda(lambda x: (x * scale_factor))(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in], outputs=[ALL_I_out_scaled], name='Psi')

    return Psi_model, terms  # 32 terms

def ortho_cann_3ff_noiso(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p):
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors)

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')
    I4theta, I4negtheta = I4_theta()([I4f_in, I4n_in, I8fn_in])

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4theta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4theta)
    I4negtheta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4negtheta)


    I4f_out = SingleInvNet_I4(I4f_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 2]))
    I4theta_max = calculate_I4theta_max(Is_max)
    I4n_out = SingleInvNet_I4theta(I4theta_ref, I4negtheta_ref, 6, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=I4theta_max)
    # I8fn_out = SingleInvNet(I8fn_ref, 24, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 6] ** 2))


    ALL_I_out = [I4f_out,I4n_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    terms = ALL_I_out.get_shape().as_list()[1]
    ALL_I_out_scaled = keras.layers.Lambda(lambda x: (x / terms * 2 * scale_factor))(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in], outputs=[ALL_I_out_scaled], name='Psi')

    return Psi_model, terms  # 32 terms

class NormalizedDense(keras.layers.Layer):
    def __init__(self, kernel_initializer, activation, weight_name, should_normalize, p, alpha, I1s_max, is_exp=True, is_ln=False, **kwargs):
        super().__init__(**kwargs)
        # Save inputs as class variables
        self.kernel_initializer=kernel_initializer
        self.activation=identity if activation is None else activation
        self.weight_name=weight_name
        self.p = p
        self.alpha = alpha
        self.should_normalize = should_normalize
        self.I1s_max = I1s_max
    # Required function for reading / writing layer to file
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "activation": self.activation,
            "weight_name": self.weight_name,
            "p": self.p,
            "alpha": self.alpha,
            "should_normalize": self.should_normalize,
            "I1s_max": self.I1s_max
        })
        return config

    # Create relevant weights
    def build(self, input_shape):
        # Create weight for power
        self.w1 = self.add_weight(
            shape=(1, ),
            initializer=self.kernel_initializer,
            constraint=keras.constraints.NonNeg(),
            trainable=True,
            name=self.weight_name + '1'
        )
        self.w2 = self.add_weight(
            shape=(1,),
            initializer=initializer_1,
            constraint= keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(self.alpha),
            trainable=True,
            name=self.weight_name + '2'
        )

    # Compute and return output of layer given input
    def call(self, inputs):
        epsilon = 1e-8  # add to normalizing factor to prevent divide by zero
        I1maxmax = np.max(self.I1s_max) if self.should_normalize else 1.0
        normalizing_factor = tf.reduce_sum(self.activation(self.w1 * self.I1s_max / I1maxmax)) + epsilon if self.should_normalize else 1.0 # compute normalizing power
        # We raise the gain to the power of 1/p to simulate Lp regularization
        # This works because we are actually using L1 regularization and the gains are all nonnegative
        return self.w2 ** (1 / self.p) * self.activation(inputs * self.w1 / I1maxmax) / normalizing_factor

class I4_theta(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # Required function for reading / writing layer to file
    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config

    # Create relevant weights
    def build(self, input_shape):
        # Create weight for power
        self.theta = np.pi / 3

    # Compute and return output of layer given input
    def call(self, inputs):
        I4f, I4n, I8 = inputs
        I4theta = I4f * (tf.cos(self.theta)) ** 2 + I4n * (tf.sin(self.theta)) ** 2 + I8 * tf.sin(2 * self.theta)
        I4negtheta = I4f * (tf.cos(self.theta)) ** 2 + I4n * (tf.sin(self.theta)) ** 2 - I8 * tf.sin(2 * self.theta)
        return [I4theta, I4negtheta]

    def compute_output_shape(self, inputShape):
        return [(None, 1), (None, 1)]




###### Displaying models
def display_equation(path2saveResults, dfs, Region, modelFit_mode):


    # Load data from file
    with open(f'{path2saveResults}/training.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    full_weight_hist = input_data["weight_hist"]
    weights = full_weight_hist[1][-1]
    disp_equation_weights(weights, dfs, Region, modelFit_mode, path2saveResults)
def disp_equation_weights(weights, dfs, Region, modelFit_mode, path2saveResults):
    print(f"Model for path {path2saveResults}: ")

    if "noiso" in path2saveResults:
        weights = [np.array([0])] * 16 + weights

    P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss, midpoint = getStressStrain(dfs, Region)
    if "ws" in path2saveResults:
        sf, terms = ortho_cann_2ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode)
    else:
        sf, terms = ortho_cann_3ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode)

    n_mus = 0
    n_abs = 0
    eqn = ""
    for x in terms:
        term = x.item()
        if term == 0:
            continue
        gain = term.args[0] * sf * 2 ## * 2 accounts for 1/2 we add later
        if term.args[1].func == sympy.core.add.Add:
            n_abs += 1
            exponent = term.args[1].args[-1].args[0][0].args[0]
            gain *= exponent
            print(f"a_{n_abs} = {gain:.4g}"+  "\\text{ kPa}")
            print(f"b_{n_abs} = {exponent:.4g}")
            term_str = f"a_{n_abs}(" + str(term.args[1]) + f") / b_{n_abs}"
            term_str = re.sub(r'\d+\.\d\d\d+', f"b_{n_abs}", term_str)

        else:
            n_mus += 1
            term_str = f"\mu_{n_mus}" + str(term.args[1])
            print(f"\mu_{n_mus} = {gain:.4g}" +  "\\text{ kPa}")
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



        if "I4theta" in term_str:
            eqn += term_str.replace("I4theta", "(I_{4s_I} - 1)")
            eqn += term_str.replace("I4theta", "(I_{4s_{II}} - 1)")

        else:
            eqn += term_str

    print("\psi &=& " + eqn[3:])






# Orthotropic CANN with fibers in the Warp and Shute directions

def ortho_cann_3ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode):
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors)
    I4theta_max = calculate_I4theta_max(Is_max)


    # Inputs defined
    output = SingleInvNet_symbolic(sp.Symbol("I1"), weights[0:8],  I1s_max=(Is_max[:, 0]))
    output += SingleInvNet_symbolic(sp.Symbol("I2"), weights[8:16],  I1s_max=(Is_max[:, 1]))
    output += SingleInvNetI4_symbolic(sp.Symbol("I4w"), weights[16:22],  I1s_max=(Is_max[:, 2]))
    output += SingleInvNetI4_symbolic(sp.Symbol("I4theta"), weights[22:28],  I1s_max=(I4theta_max))

    terms = 14

    return scale_factor / terms * 2, output
def ortho_cann_2ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode):
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors)


    output = SingleInvNet_symbolic(sp.Symbol("I1"), weights[0:8],  I1s_max=(Is_max[:, 0]))
    output += SingleInvNet_symbolic(sp.Symbol("I2"), weights[8:16],  I1s_max=(Is_max[:, 1]))
    output += SingleInvNetI4_symbolic(sp.Symbol("I4w"), weights[16:22],  I1s_max=(Is_max[:, 2]))
    output += SingleInvNetI4_symbolic(sp.Symbol("I4s"), weights[22:28], I1s_max=(Is_max[:, 4]))

    terms = 14

    return scale_factor / terms * 2, output

def SingleInvNet_symbolic(I1_ref, weights, I1s_max):
    # Layer 1. order
    I_1_w11 = 1 * NormalizedDense_symbolic(I1_ref, weights[0:2], None, None, I1s_max=I1s_max)
    I_1_w21 = 1 * NormalizedDense_symbolic(I1_ref, weights[2:4], activation_Exp, activation_Exp_sp, I1s_max=I1s_max)
    I_1_w41 = 1 * NormalizedDense_symbolic(I1_ref ** 2, weights[4:6], None, None, I1s_max=I1s_max ** 2)
    I_1_w51 = 1 * NormalizedDense_symbolic(I1_ref ** 2, weights[6:8], activation_Exp, activation_Exp_sp, I1s_max=I1s_max ** 2)
    return [I_1_w11, I_1_w21, I_1_w41, I_1_w51]

def SingleInvNetI4_symbolic(I1_ref, weights, I1s_max):
    I_1_w21 = 1 * NormalizedDense_symbolic(I1_ref, weights[0:2], activation_Exp_I4, activation_Exp_I4_sp, I1s_max=I1s_max)
    I_1_w41 = 1 * NormalizedDense_symbolic(I1_ref ** 2, weights[2:4], None, None, I1s_max=I1s_max ** 2)
    I_1_w51 = 1 * NormalizedDense_symbolic(I1_ref ** 2, weights[4:6], activation_Exp, activation_Exp_sp, I1s_max=I1s_max ** 2)
    return [I_1_w21, I_1_w41, I_1_w51]


def NormalizedDense_symbolic(I1_ref, weights, activation, activation_sp, I1s_max):
    if activation is None:
        activation = identity
        activation_sp = identity
    w1 = weights[0]
    w2 = weights[1]
    p = 0.5
    epsilon = 1e-8  # add to normalizing factor to prevent divide by zero
    I1maxmax = np.max(I1s_max)
    normalizing_factor = np.sum(activation(w1 * I1s_max / I1maxmax)) + epsilon
    return w2 ** (1/p) / normalizing_factor * activation_sp(w1 / I1maxmax * I1_ref)