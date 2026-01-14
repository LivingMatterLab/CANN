#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
from tensorflow.keras import regularizers

# from keras import regularizers 
#%%

# L1 regularization strength will be passed as parameter

##
# Initialize (o), (o)^2, exp, log
##

initializer_1 = 'ones'
initializer_zero = 'zeros'
initializer_outer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
initializer_log = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1)
initializer_J = tf.keras.initializers.RandomUniform(minval=1., maxval=4) # worked off and on, starts with huge residual
initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=1.0) # worked off and on, starts with huge residual
initializer_odgen = tf.keras.initializers.RandomUniform(minval=0.5, maxval=1.0) # worked off and on, starts with huge residual

Square_Layer = tf.keras.layers.Lambda(lambda t: K.square(t))
Cube_Layer = tf.keras.layers.Lambda(lambda t: K.pow(K.abs(t), 3))
#%% Activations

def flatten(l):
    return [item for sublist in l for item in sublist]

# Self defined activation functions for exp 
def Exp(x):
    return 1.0*tf.math.exp(x)

def activation_Exp(x):
    return 1.0*(tf.math.exp(x) -1.0)

def activation_Exp_minus_x(x):
    return 1.0*(tf.math.exp(x)-x-1.0)

def activation_ln(x):
    return -1.0*tf.math.log(1.0 - (x))


#%% Constraints

class BoundedNonNeg(keras.constraints.Constraint):
    """
    Custom constraint that bounds weights between 0 and an upper bound.
    
    Args:
        max_value: Upper bound for the weights (default: 1.0)
    """
    def __init__(self, max_value=1.0):
        self.max_value = max_value
    
    def __call__(self, w):
        # Clip weights to be between 0 and max_value
        return tf.clip_by_value(w, 0.0, self.max_value)
    
    def get_config(self):
        return {'max_value': self.max_value}


class SingleOutputDenseLayer(keras.layers.Layer):
    """
    Custom dense layer with a single output unit.
    This is equivalent to Dense(1) but implemented as a custom layer.
    
    Args:
        kernel_initializer: Initializer for the kernel weights
        kernel_constraint: Constraint for the kernel weights
        kernel_regularizer: Regularizer for the kernel weights
        use_bias: Whether to use bias (default: False)
        activation: Activation function (default: None)
        name: Name of the layer
    """
    def __init__(self, kernel_initializer='glorot_uniform', 
                 kernel_constraint=None, 
                 kernel_regularizer=None,
                 use_bias=False,
                 activation=None,
                 name=None,
                 **kwargs):
        super(SingleOutputDenseLayer, self).__init__(name=name, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.activation = activation
        
    def build(self, input_shape):
        # input_shape is (batch_size, input_dim)
        input_dim = input_shape[-1]
        
        # Create kernel weight: shape (input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, 1),
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.bias = None
            
        super(SingleOutputDenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # kernel shape: (input_dim, 1)
        # output shape: (batch_size, 1)
        output = tf.matmul(inputs, self.kernel ** 2)
        
        if self.bias is not None:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def get_config(self):
        config = super(SingleOutputDenseLayer, self).get_config()
        config.update({
            'kernel_initializer': self.kernel_initializer,
            'kernel_constraint': self.kernel_constraint,
            'kernel_regularizer': self.kernel_regularizer,
            'use_bias': self.use_bias,
            'activation': self.activation
        })
        return config


class InvNetDenseLayer(keras.layers.Layer):
    """
    Custom dense layer with a single output unit for use in SingleInvNet.
    This is equivalent to Dense(1) but implemented as a custom layer.
    Separate from SingleOutputDenseLayer for different use cases.
    
    Args:
        kernel_initializer: Initializer for the kernel weights
        kernel_constraint: Constraint for the kernel weights
        kernel_regularizer: Regularizer for the kernel weights
        use_bias: Whether to use bias (default: False)
        activation: Activation function (default: None)
        name: Name of the layer
    """
    def __init__(self, kernel_initializer='glorot_uniform', 
                 kernel_constraint=None, 
                 use_bias=False,
                 activation=None,
                 name=None,
                 **kwargs):
        super(InvNetDenseLayer, self).__init__(name=name, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.activation = activation
        
    def build(self, input_shape):
        # input_shape is (batch_size, input_dim)
        input_dim = input_shape[-1]
        
        # Create kernel weight: shape (input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, 1),
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            trainable=True
        )
            
        super(InvNetDenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # kernel shape: (input_dim, 1)
        # output shape: (batch_size, 1)
        output = tf.matmul(inputs, self.kernel)
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output / (self.kernel + 1e-2)
    
    def get_config(self):
        config = super(InvNetDenseLayer, self).get_config()
        config.update({
            'kernel_initializer': self.kernel_initializer,
            'kernel_constraint': self.kernel_constraint,
            'use_bias': self.use_bias,
            'activation': self.activation
        })
        return config


#%% PI-CANN
# Define network block
def SingleInvNet(I1_ref, idi, L1):
    
    # Layer 1. order
    I_1_w11 = I1_ref
    I_1_w21 = InvNetDenseLayer(kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=activation_Exp,name='w'+str(2+idi)+'1')(I1_ref)
    
    # I_1_w31 = InvNetDenseLayer(kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)


    # Layer 2. order
    I_1_w41 = Square_Layer(I1_ref) # no need for an extra weight here
    I_1_w51 = InvNetDenseLayer(kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=activation_Exp,name='w'+str(4+idi)+'1')(Square_Layer(I1_ref))
    # I_1_w61 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(6+idi)+'1')(Square_Layer(I1_ref))

    collect = [I_1_w11, I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out


def BulkNet(J, idi, L1, include_mixed=False):
    logJ = keras.layers.Lambda(lambda x: tf.math.log(x))(J)

    # term_log_J_squared = Square_Layer(logJ)
    term_exp_log_J_squared = InvNetDenseLayer(kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=activation_Exp, name='w' + str(3 + idi) + '1')(
        Square_Layer(logJ))
    collect = [ term_exp_log_J_squared]

    term_exp_log_J = InvNetDenseLayer(kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                use_bias=False, activation=activation_Exp_minus_x, name='w' + str(1 + idi) + '1')(logJ)
    collect.append(term_exp_log_J)
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out

def SinglePrincipalStretchNet(lambda_in, idi, L1):
    logLambda = keras.layers.Lambda(lambda x: tf.math.log(x))(lambda_in)

    # Layer 1. order
    # I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                              use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(I1_ref)
    layer1 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(1 + idi) + '1')(logLambda)
    # layer2 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
    #                                 kernel_regularizer=regularizers.l1(L1),
    #                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(2 + idi) + '1')(logLambda)
    # layer3 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
    #                                 kernel_regularizer=regularizers.l1(L1),
    #                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(3 + idi) + '1')(logLambda)
    # layer4 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
    #                                 kernel_regularizer=regularizers.l1(L1),
    #                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(4 + idi) + '1')(logLambda)
    # layer5 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
    #                                 kernel_regularizer=regularizers.l1(L1),
    #                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(5 + idi) + '1')(logLambda)

    # I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)

    # Layer 2. order
    # I_1_w41 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=regularizers.l1(L1),
    #                              use_bias=False, activation=None, name='w' + str(2 + idi) + '1')(Square_Layer(logLambda))
    # I_1_w51 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=regularizers.l1(L1),
    #                              use_bias=False, activation=activation_Exp, name='w' + str(3 + idi) + '1')(
    #     Square_Layer(logLambda))

    # # Layer 3. order
    # I_1_w61 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=regularizers.l1(L1),
    #                              use_bias=False, activation=None, name='w' + str(4 + idi) + '1')(Cube_Layer(logLambda))
    # I_1_w71 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=regularizers.l1(L1),
    #                              use_bias=False, activation=activation_Exp, name='w' + str(5 + idi) + '1')(
    #     Cube_Layer(logLambda))
    # I_1_w61 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(6+idi)+'1')(Square_Layer(I1_ref))

    # collect = [layer1, layer2, layer3, layer4, layer5]
    collect = [layer1]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out


class MixedNetLayer(keras.layers.Layer):
    """
    Custom layer that computes mixed terms combining I1 invariant with J (volume change).
    Creates terms of the form: I1_ref * J^power
    Also includes the I_1_w21 term from BulkNet: exp(logJ * weight) - logJ * weight - 1
    Uses explicit custom weights instead of built-in Dense layers.
    """
    def __init__(self, idi, L1, **kwargs):
        super(MixedNetLayer, self).__init__(**kwargs)
        self.L1 = L1        
        # Create explicit weight for I_1_w21 term (moved from BulkNet)
        # This term uses activation_Exp_minus_x: exp(x) - x - 1
        # self.m_minus_one = self.add_weight(
        #     name='w' + str(1 + idi) + '1',  
        #     shape=(1, 1),
        #     initializer=initializer_exp,
        #     constraint=keras.constraints.NonNeg(),
        #     regularizer=regularizers.l1(L1),
        #     trainable=True
        # )
        # self.mu_1 = self.add_weight(
        #     name='w' + str(2 + idi) + '1',  
        #     shape=(1, 1),
        #     initializer=initializer_exp,
        #     constraint=keras.constraints.NonNeg(),
        #     regularizer=regularizers.l1(L1),
        #     trainable=True
        # )

        self.a = self.add_weight(
            name='w' + str(1 + idi) + '1',  
            shape=(1, 1),
            initializer=initializer_exp,
            constraint=keras.constraints.NonNeg(),
            regularizer=regularizers.l1(L1),
            trainable=True
        )
        self.b = self.add_weight(
            name='w' + str(2 + idi) + '1',  
            shape=(1, 1),
            initializer=initializer_exp,
            constraint=keras.constraints.NonNeg(),
            regularizer=regularizers.l1(L1),
            trainable=True
        )
        
    
    def call(self, inputs):
        I1, J = inputs
        I1_ref = I1 / J ** (2/3) - 3.0
        return [tf.math.exp(self.a * I1_ref + self.b * J) - self.b * J]
        # mu_1 = self.mu_1
        # m = 1 + self.m_minus_one
        # I1_ref = I1/J ** (2/3)  - 3.0
        # mu_2 = 1 / (4 * m - 2) / (self.mu_1 + 1e-6)
        # k = 2 * m * mu_2
        # mu_1 = 0
        # mu_2 = 0
        # k = 0
        # return [(J ** m -1)* I1_ref + mu_1 * I1_ref ** 2 + mu_2 * (J ** (2 * m) - 1) - k * J]
        # return [J ** m * (I1 - 3)]
    def get_config(self):
        config = super(MixedNetLayer, self).get_config()
        config.update({
            'L1': self.L1
        })
        return config



def SingleInvNetMix(I1_ref):
    
    # Layer 1. order
    I_1_w11 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=None)(I1_ref)
    I_1_w21 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=None)(I1_ref)
    I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
                              use_bias=False, activation=activation_ln)(I1_ref)
    # I_1_w31 = logLayer('w'+str(3+idi)+'1')(I1_ref)


    collect = [I_1_w11, I_1_w21]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out


def StrainEnergyCANN(l1_reg=0.01, include_mixed=False, no_I2_flag=False):

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    J_in = tf.keras.Input(shape=(1,), name='J')

    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x[0] / K.pow(x[1], 2/3)-3.0))([I1_in, J_in])
    I2_ref = keras.layers.Lambda(lambda x: (x[0] / K.pow(x[1], 4/3)-3.0))([I2_in, J_in])
    
    I1_out = SingleInvNet(I1_ref, 0, l1_reg)
    terms = I1_out.shape[1]
    I2_out = SingleInvNet(I2_ref, terms, l1_reg)
    terms += I2_out.shape[1]
    J_out = BulkNet(J_in, terms, l1_reg, include_mixed)
    terms += J_out.shape[1]
    if include_mixed:
        mixed_layer = MixedNetLayer(terms, l1_reg, name='mixed_net')
        Mixed_out = mixed_layer([I1_in, J_in])
        terms += len(Mixed_out)
    else:
        Mixed_out = []

    ALL_I_out = [I1_out,I2_out,J_out] + Mixed_out
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer=initializer_outer,kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, J_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms


class SinglePrincipalStretchLayer(keras.layers.Layer):
    """
    Custom layer that computes strain energy for a single principal stretch.
    Contains 5 inner weights (one per term) and 5 outer weights (final combination).
    Total: 10 trainable weights.
    
    This replaces the previous single_principal_stretch_model which had:
    - 5 Dense layers (inner weights) applying activation_Exp_minus_x to logLambda
    - 1 final Dense layer (outer weights) combining the 5 terms
    """
    def __init__(self, l1_reg=0.01, p=1.0, **kwargs):
        super(SinglePrincipalStretchLayer, self).__init__(**kwargs)
        self.l1_reg = l1_reg
        self.p = p
        self.n_layers = 1
        
    def build(self, input_shape):
        # 5 inner weights (one for each term) - shape (5, 1)
        self.inner_weights = self.add_weight(
            name='inner_weights',
            shape=(1, 1),
            initializer=initializer_J,
            constraint=keras.constraints.NonNeg(),
            regularizer=None,
            trainable=True
        )
        
        # # 5 outer weights (for final combination) - shape (5, 1)
        # self.outer_weights = self.add_weight(
        #     name='outer_weights',
        #     shape=(self.n_layers, 1),
        #     initializer=initializer_outer,
        #     constraint=keras.constraints.NonNeg(),
        #     regularizer=regularizers.l1(self.l1_reg),
        #     trainable=True
        # )
        
        super(SinglePrincipalStretchLayer, self).build(input_shape)
    
    def call(self, inputs):
        lambda_in = inputs  # Shape: (batch, 1)
        
        # Compute log(lambda)
        logLambda = tf.math.log(lambda_in)  # Shape: (batch, 1)
        
        # Apply activation to each term
        term = activation_Exp_minus_x(self.inner_weights * logLambda) / (self.inner_weights ** 2 + 1e-2) # Shape: (batch, 5)
        
        # Combine with outer weights: sum(outer_weight_i * term_i)
        # outer_weights shape: (5, 1), terms shape: (batch, 5)
        # Multiply element-wise and sum
        
        return term
    
    def get_config(self):
        config = super(SinglePrincipalStretchLayer, self).get_config()
        config.update({'l1_reg': self.l1_reg})
        return config

def StrainEnergyPrincipalStretch(l1_reg=0.01, p=1.0):

    # Inputs defined
    lambda_1 = tf.keras.Input(shape=(1,), name='lambda_1_psi')
    lambda_2 = tf.keras.Input(shape=(1,), name='lambda_2_psi')
    lambda_3 = tf.keras.Input(shape=(1,), name='lambda_3_psi')

    # Create a single layer instance that will be shared across all three lambda inputs
    # This layer has 10 weights total (5 inner + 5 outer)
    single_principal_stretch_layer = SinglePrincipalStretchLayer(l1_reg=l1_reg, name='single_principal_stretch', p=p)

    # Apply the same layer to all three principal stretches
    lambda_out_1 = single_principal_stretch_layer(lambda_1)
    lambda_out_2 = single_principal_stretch_layer(lambda_2)
    lambda_out_3 = single_principal_stretch_layer(lambda_3)
    lambda_out_combined = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined')([lambda_out_1, lambda_out_2, lambda_out_3])

    # Create a single layer instance that will be shared across all three lambda inputs
    # This layer has 10 weights total (5 inner + 5 outer)
    single_principal_stretch_layer2 = SinglePrincipalStretchLayer(l1_reg=l1_reg, name='single_principal_stretch2', p=p)

    # Apply the same layer to all three principal stretches
    lambda_out_1_2 = single_principal_stretch_layer2(lambda_1)
    lambda_out_2_2 = single_principal_stretch_layer2(lambda_2)
    lambda_out_3_2 = single_principal_stretch_layer2(lambda_3)
    lambda_out_combined_2 = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined_2')([lambda_out_1_2, lambda_out_2_2, lambda_out_3_2])

    single_principal_stretch_layer3 = SinglePrincipalStretchLayer(l1_reg=l1_reg, name='single_principal_stretch3', p=p)

    # Apply the same layer to all three principal stretches
    lambda_out_1_3 = single_principal_stretch_layer3(lambda_1)
    lambda_out_2_3 = single_principal_stretch_layer3(lambda_2)
    lambda_out_3_3 = single_principal_stretch_layer3(lambda_3)
    lambda_out_combined_3 = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined_3')([lambda_out_1_3, lambda_out_2_3, lambda_out_3_3])

    
    ALL_I_out = [lambda_out_combined, lambda_out_combined_2, lambda_out_combined_3]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer - using custom single output dense layer
    psi_out = SingleOutputDenseLayer(
        kernel_initializer=initializer_outer,
        kernel_constraint=keras.constraints.NonNeg(),
        kernel_regularizer=regularizers.l1(l1_reg),
        use_bias=False,
        activation=None,
        name='wx2'
    )(ALL_I_out)
    
    
    Psi_model = keras.models.Model(inputs=[lambda_1, lambda_2, lambda_3], outputs=[psi_out], name='Psi')
    
    # The number of terms is 5 (from the 5 inner/outer weight pairs)
    
    return Psi_model, 2


def StrainEnergyPrincipalStretchMixed(l1_reg=0.01, p=1.0):

    # Inputs defined
    lambda_1 = tf.keras.Input(shape=(1,), name='lambda_1_psi')
    lambda_2 = tf.keras.Input(shape=(1,), name='lambda_2_psi')
    lambda_3 = tf.keras.Input(shape=(1,), name='lambda_3_psi')

    # Create a single layer instance that will be shared across all three lambda inputs
    # This layer has 10 weights total (5 inner + 5 outer)
    single_principal_stretch_layer = SinglePrincipalStretchLayer(l1_reg=l1_reg, name='single_principal_stretch', p=p)

    # Apply the same layer to all three principal stretches
    lambda_out_1 = single_principal_stretch_layer(lambda_1)
    lambda_out_2 = single_principal_stretch_layer(lambda_2)
    lambda_out_3 = single_principal_stretch_layer(lambda_3)
    lambda_out_combined = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined')([lambda_out_1, lambda_out_2, lambda_out_3])

    # Create a single layer instance that will be shared across all three lambda inputs
    # This layer has 10 weights total (5 inner + 5 outer)
    single_principal_stretch_layer2 = SinglePrincipalStretchLayer(l1_reg=l1_reg, name='single_principal_stretch2', p=p)

    # Apply the same layer to all three principal stretches
    lambda_out_1_2 = single_principal_stretch_layer2(lambda_1)
    lambda_out_2_2 = single_principal_stretch_layer2(lambda_2)
    lambda_out_3_2 = single_principal_stretch_layer2(lambda_3)
    lambda_out_combined_2 = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined_2')([lambda_out_1_2, lambda_out_2_2, lambda_out_3_2])

    ## Invariant-based terms
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2)([lambda_1, lambda_2, lambda_3])
    I2 = keras.layers.Lambda(lambda x: (x[0] * x[1]) ** 2 + (x[0] * x[2]) ** 2 + (x[1] * x[2]) ** 2)([lambda_1, lambda_2, lambda_3])
    J = keras.layers.Lambda(lambda x: x[0] * x[1] * x[2])([lambda_1, lambda_2, lambda_3])

    I1_ref = keras.layers.Lambda(lambda x: (x[0] / K.pow(x[1], 2/3)-3.0))([I1, J])
    I2_ref = keras.layers.Lambda(lambda x: (K.pow(x[0], 1.5) / K.pow(x[1], 2.0)- K.pow(3.0, 1.5)))([I2, J])

    
    I1_out = SingleInvNet(I1_ref, 0, l1_reg)
    terms = I1_out.shape[1]
    I2_out = SingleInvNet(I2_ref, terms, l1_reg)
    terms += I2_out.shape[1]
    J_out = BulkNet(J, terms, l1_reg)
    terms += J_out.shape[1]

    ALL_I_out = [I1_out, I2_out, J_out, lambda_out_combined, lambda_out_combined_2]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer - using custom single output dense layer
    psi_out = SingleOutputDenseLayer(
        kernel_initializer=initializer_outer,
        kernel_constraint=keras.constraints.NonNeg(),
        kernel_regularizer=regularizers.l1(l1_reg),
        use_bias=False,
        activation=None,
        name='wx2'
    )(ALL_I_out)
    
    
    Psi_model = keras.models.Model(inputs=[lambda_1, lambda_2, lambda_3], outputs=[psi_out], name='Psi')
    
    # The number of terms is 5 (from the 5 inner/outer weight pairs)
    
    return Psi_model, terms + 1
