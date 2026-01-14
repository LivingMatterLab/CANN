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

initializer_outer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
initializer_J = tf.keras.initializers.RandomUniform(minval=1., maxval=4) # worked off and on, starts with huge residual
initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=1.0) # worked off and on, starts with huge residual

Square_Layer = tf.keras.layers.Lambda(lambda t: K.square(t))


def activation_Exp(x):
    return 1.0*(tf.math.exp(x) -1.0)

def activation_Exp_minus_x(x):
    return 1.0*(tf.math.exp(x)-x-1.0)



class OuterLayer(keras.layers.Layer):
    """
    Custom dense layer with a single output unit to be used as the final layer of the network. 
    Multiplies the input by a weight and raises it to the power of 1/p in order to enable lp regularization.
    
    Args:
        kernel_initializer: Initializer for the kernel weights
        kernel_constraint: Constraint for the kernel weights
        kernel_regularizer: Regularizer for the kernel weights
        use_bias: Whether to use bias (default: False)
        activation: Activation function (default: None)
        name: Name of the layer
    """
    def __init__(self, kernel_initializer='glorot_uniform', lp_reg=0.0, p=1.0,
                 name=None,
                 **kwargs):
        super(OuterLayer, self).__init__(name=name, **kwargs)
        self.lp_reg = lp_reg
        self.p = p
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        # input_shape is (batch_size, input_dim)
        input_dim = input_shape[-1]
        
        # Create outer weight: shape (input_dim, 1)
        self.outer_weights = self.add_weight(
            name='outer_weights',
            shape=(input_dim, 1),
            initializer=self.kernel_initializer,
            constraint=keras.constraints.NonNeg(),
            regularizer=regularizers.l1(self.lp_reg),
            trainable=True
        )
        super(OuterLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # kernel shape: (input_dim, 1)
        # output shape: (batch_size, 1)
        output = tf.matmul(inputs, self.outer_weights ** (1.0 / self.p))
            
        return output
    
    def get_config(self):
        config = super(OuterLayer, self).get_config()
        config.update({
            'l1_reg': self.lp_reg,
            'p': self.p
        })
        return config


class InnerLayer(keras.layers.Layer):
    """
    Custom dense layer with a single output unit to be used in the inner layers of the network.
    Applies the function f(x; w) = activation(w * x) / (w + 1e-2) to the input.
    
    Args:
        kernel_initializer: Initializer for the weight
        activation: Activation function (default: None)
        name: Name of the layer
    """
    def __init__(self, kernel_initializer='glorot_uniform', 
                 activation=None,
                 name=None,
                 **kwargs):
        super(InnerLayer, self).__init__(name=name, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        
    def build(self, input_shape):
        # input_shape is (batch_size, input_dim)
        input_dim = input_shape[-1]
        
        # Create kernel weight: shape (input_dim, 1)
        self.inner_weight = self.add_weight(
            name='inner_weight',
            shape=(1, 1),
            initializer=self.kernel_initializer,
            constraint=keras.constraints.NonNeg(),
            trainable=True
        )
            
        super(InnerLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # kernel shape: (input_dim, 1)
        # output shape: (batch_size, 1)
        output = tf.matmul(inputs, self.inner_weight)
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output / (self.inner_weight + 1e-2)
    
    def get_config(self):
        config = super(InnerLayer, self).get_config()
        config.update({
            'kernel_initializer': self.kernel_initializer,
            'activation': self.activation
        })
        return config



class MixedNetLayer(keras.layers.Layer):
    """
    Custom layer that computes mixed terms combining I1 invariant with J (volume change).
    Creates terms of the form: I1_ref * J^power
    """
    def __init__(self, idi, name=None, **kwargs):
        super(MixedNetLayer, self).__init__(name=name, **kwargs)

        self.idi = idi
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.alpha = self.add_weight(
            name='alpha',  
            shape=(1, 1),
            initializer=initializer_exp,
            constraint=keras.constraints.NonNeg(),
            trainable=True
        )
    
    def call(self, inputs):
        I1_ref, J = inputs
        return I1_ref * J ** self.alpha

    def get_config(self):
        config = super(MixedNetLayer, self).get_config()
        config.update({
            'idi': self.idi
        })
        return config


class SinglePrincipalStretchLayer(keras.layers.Layer):
    """
    Custom layer that computes a single term in the principal stretch-based hyperfoam model
    """
    def __init__(self, **kwargs):
        super(SinglePrincipalStretchLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.inner_weight = self.add_weight(
            name='inner_weights',
            shape=(1, 1),
            initializer=initializer_J,
            constraint=keras.constraints.NonNeg(),
            regularizer=None,
            trainable=True
        )
        
        
        super(SinglePrincipalStretchLayer, self).build(input_shape)
    
    def call(self, inputs):
        lambda_in = inputs  # Shape: (batch, 1)
        
        # Compute log(lambda)
        logLambda = tf.math.log(lambda_in)  # Shape: (batch, 1)
        
        # Apply activation
        term = activation_Exp_minus_x(self.inner_weight * logLambda) / (self.inner_weight ** 2 + 1e-2) # Shape: (batch, 5)
        
        return term
    
    def get_config(self):
        config = super(SinglePrincipalStretchLayer, self).get_config()
        config.update({})
        return config


#%% PI-CANN
# All terms that are a function of either I1_bar or I2_bar
def SingleInvNet(I_ref, idi):
    
    # I ^ 1 terms
    I_1_w11 = I_ref
    I_1_w21 = InnerLayer(kernel_initializer=initializer_exp, activation=activation_Exp, name='w'+str(2+idi)+'1')(I_ref)

    # I ^ 2 terms
    I_1_w41 = Square_Layer(I_ref) # no need for an extra weight here
    I_1_w51 = InnerLayer(kernel_initializer=initializer_exp, activation=activation_Exp,name='w'+str(4+idi)+'1')(Square_Layer(I_ref))
    

    collect = [I_1_w11, I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out

## All terms that are a function of J
def BulkNet(J, idi):
    logJ = keras.layers.Lambda(lambda x: tf.math.log(x))(J)

    term_exp_log_J = InnerLayer(kernel_initializer=initializer_J, activation=activation_Exp_minus_x, name='w' + str(1 + idi) + '1')(logJ)

    term_exp_log_J_squared = InnerLayer(kernel_initializer=initializer_J, activation=activation_Exp, name='w' + str(2 + idi) + '1')(
        Square_Layer(logJ))

    collect = [ term_exp_log_J_squared, term_exp_log_J]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out

def StrainEnergyCANN(lp_reg, p, include_invariant_terms, include_mixed_terms, include_principal_stretch_terms):

    # Inputs defined as principal stretches
    lambda_1 = tf.keras.Input(shape=(1,), name='lambda_1_psi')
    lambda_2 = tf.keras.Input(shape=(1,), name='lambda_2_psi')
    lambda_3 = tf.keras.Input(shape=(1,), name='lambda_3_psi')

    if include_invariant_terms:
        ## Compute principal invariants from principal stretches
        I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2)([lambda_1, lambda_2, lambda_3])
        I2 = keras.layers.Lambda(lambda x: (x[0] * x[1]) ** 2 + (x[0] * x[2]) ** 2 + (x[1] * x[2]) ** 2)([lambda_1, lambda_2, lambda_3])
        J = keras.layers.Lambda(lambda x: x[0] * x[1] * x[2])([lambda_1, lambda_2, lambda_3])

        I1_ref = keras.layers.Lambda(lambda x: (x[0] / K.pow(x[1], 2/3)-3.0))([I1, J])
        I2_ref = keras.layers.Lambda(lambda x: (K.pow(x[0], 1.5) / K.pow(x[1], 2.0)- K.pow(3.0, 1.5)))([I2, J])

        ## Compute invariant-based terms
        I1_out = SingleInvNet(I1_ref, 0)
        terms = I1_out.shape[1]
        I2_out = SingleInvNet(I2_ref, terms)
        terms += I2_out.shape[1]
        J_out = BulkNet(J, terms)
        terms += J_out.shape[1]

        ## Compute terms based on multiple invariants
        if include_mixed_terms:
            mixed_layer = MixedNetLayer(terms, name='mixed_net')
            Mixed_out = mixed_layer([I1_ref, J_in])
            terms += len(Mixed_out)
        else:
            Mixed_out = []

        ALL_I_out = [I1_out, I2_out, J_out] + Mixed_out
    else:
        ALL_I_out = []
    
    if include_principal_stretch_terms:
        # Create a single layer instance that will be shared across all three lambda inputs
        # This layer has 10 weights total (5 inner + 5 outer)
        single_principal_stretch_layer = SinglePrincipalStretchLayer(l1_reg=lp_reg, name='single_principal_stretch', p=p)

        # Apply the same layer to all three principal stretches
        lambda_out_1 = single_principal_stretch_layer(lambda_1)
        lambda_out_2 = single_principal_stretch_layer(lambda_2)
        lambda_out_3 = single_principal_stretch_layer(lambda_3)
        lambda_out_combined = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined')([lambda_out_1, lambda_out_2, lambda_out_3])

        # Create a single layer instance that will be shared across all three lambda inputs
        # This layer has 10 weights total (5 inner + 5 outer)
        single_principal_stretch_layer2 = SinglePrincipalStretchLayer(l1_reg=lp_reg, name='single_principal_stretch2', p=p)

        # Apply the same layer to all three principal stretches
        lambda_out_1_2 = single_principal_stretch_layer2(lambda_1)
        lambda_out_2_2 = single_principal_stretch_layer2(lambda_2)
        lambda_out_3_2 = single_principal_stretch_layer2(lambda_3)
        lambda_out_combined_2 = keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='lambda_out_combined_2')([lambda_out_1_2, lambda_out_2_2, lambda_out_3_2])

        ALL_I_out += [lambda_out_combined, lambda_out_combined_2]
        terms += 2

    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    # second layer - using custom single output dense layer
    psi_out = OuterLayer(
        kernel_initializer=initializer_outer,
        kernel_constraint=keras.constraints.NonNeg(),
        kernel_regularizer=regularizers.l1(lp_reg),
        use_bias=False,
        activation=None,
        name='wx2'
    )(ALL_I_out)
    
    
    Psi_model = keras.models.Model(inputs=[lambda_1, lambda_2, lambda_3], outputs=[psi_out], name='Psi')
    
    # The number of terms is 5 (from the 5 inner/outer weight pairs)
    
    return Psi_model, terms
