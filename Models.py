#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:40:23 2022

@author: kevinlinka
"""


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
# Initialize (o), (o)^2, exp, log
##
initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=0.00001) # worked off and on, starts with huge residual
initializer_1 = 'glorot_normal'

#%% Activations

def flatten(l):
    return [item for sublist in l for item in sublist]

# Self defined activation functions for exp 
def activation_Exp(x):
    return 1.0*(tf.math.exp(x) -1.0) 

def activation_ln(x):
    return -1.0*tf.math.log(1.0 - (x))




#%% PI-CANN
# Define network block
def SingleInvNet6(I1_ref,idi,L1):
    
    # Layer 1. order

    I_1_w11 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=None,name='w'+str(1+idi)+'1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=activation_Exp,name='w'+str(2+idi)+'1')(I1_ref)
    I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                              use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)


    # Layer 2. order
    I_1_w41 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=None,name='w'+str(4+idi)+'1')(tf.math.square(I1_ref))
    I_1_w51 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=activation_Exp,name='w'+str(5+idi)+'1')(tf.math.square(I1_ref))
    I_1_w61 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                              use_bias=False, activation=activation_ln,name='w'+str(6+idi)+'1')(tf.math.square(I1_ref))


    collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41, I_1_w51, I_1_w61]        
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out

  

def SingleInvNet4(I1_ref,idi,L2):
    
    # Layer 1. order
    I_1_w11 = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=None,name='w'+str(1+idi)+'1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=activation_Exp,name='w'+str(2+idi)+'1')(I1_ref)

    # Layer 2. order
    I_1_w31 = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=None,name='w'+str(3+idi)+'1')(tf.math.square(I1_ref))
    I_1_w41 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=activation_Exp,name='w'+str(4+idi)+'1')(tf.math.square(I1_ref))


    collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41]        
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out


def StrainEnergyCANN():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    L2 = 0.001
    I1_out = SingleInvNet4(I1_ref, 0, L2)
    terms = I1_out.get_shape().as_list()[1] 
    I2_out = SingleInvNet4(I2_ref, terms, L2)
    
    ALL_I_out = [I1_out,I2_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=keras.regularizers.l2(L2),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms*2


