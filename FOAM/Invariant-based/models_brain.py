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
initializer_log = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1)
# initializer_log = 'glorot_normal'
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



#%% Neo Hookean


# Define network block
def Net_NeoHook(I1_ref, idi, l1_reg):
    
    # Layer 1. order
    I_1_w11 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(l1_reg),
                                 use_bias=False, activation=None,name='w'+str(1+idi)+'1')(I1_ref)

    collect_out = I_1_w11
    
    return collect_out

def StrainEnergyNeoHook():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    
    I1_out = Net_NeoHook(I1_ref, 0, l1_reg)
    terms = I1_out.get_shape().as_list()[1] 
    
    
    ALL_I_out = I1_out
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms





#%% Blatz_Ko


def StrainEnergyBlatz_Ko():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    
    I2_out = Net_NeoHook(I2_ref, 6, l1_reg)
    terms = I2_out.get_shape().as_list()[1] 
    
    
    ALL_I_out = I2_out
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms



#%% Mooney Rivilin

# Define network block
def StrainEnergyMoonRiv():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    
    I1_out = Net_NeoHook(I1_ref, 0, l1_reg)
    terms = I1_out.get_shape().as_list()[1] 
    
    I2_out = Net_NeoHook(I2_ref, 6, l1_reg)
    
    ALL_I_out = [I1_out,I2_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms*2


#%% Isihara model

# Define network block
def Net_Isihara(I1_ref, idi, l1_reg):
    
    # Layer 1. order
    I_1_w11 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(l1_reg),
                                 use_bias=False, activation=None,name='w'+str(1+idi)+'1')(I1_ref)

    # Layer 2. order
    I_1_w41 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=None,name='w'+str(4+idi)+'1')(Square_Layer(I1_ref))
    
    collect = [I_1_w11, I_1_w41]        
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    return collect_out

# Define network block
def StrainEnergyIsihara():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    
    I1_out = Net_Isihara(I1_ref, 0, l1_reg)
    terms = I1_out.get_shape().as_list()[1] 
    
    I2_out = Net_NeoHook(I2_ref, 6, l1_reg)
    
    ALL_I_out = [I1_out,I2_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms*2


#%% Demiray

# Define network block
def Net_Demiray(I1_ref, idi, l1_reg):
    
    # Layer 1. order
    I_1_w21 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=activation_Exp,name='w'+str(2+idi)+'1')(I1_ref)

      
    collect_out = I_1_w21
    
    return collect_out

def StrainEnergyDemiray():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    
    I1_out = Net_Demiray(I1_ref, 0, l1_reg)
    terms = I1_out.get_shape().as_list()[1] 
    
    
    ALL_I_out = I1_out
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms


#%% Gent


def Net_Gent(I1_ref, idi, l1_reg):
    
    # Layer 1. order
    I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
                              use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)

    # collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41, I_1_w51, I_1_w61]        
    collect_out = I_1_w31
    
    return collect_out

def StrainEnergyGent():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    # I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)

    # I1_out = logLayer('w31')(I1_ref)
    I1_out = Net_Gent(I1_ref, 0, l1_reg)
    terms = I1_out.get_shape().as_list()[1] 
    # I2_out = tf.zeros_like(I1_out)
    
    
    ALL_I_out = I1_out
    # ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms


#%% Holzapfel model

# Define network block
def Net_Holzapfel(I1_ref, idi, l1_reg):
    

    I_1_w51 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=activation_Exp,name='w'+str(5+idi)+'1')(Square_Layer(I1_ref))
    
    
    return I_1_w51

# Define network block

def StrainEnergyHolzapfel():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    # I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    
    
    I1_out = Net_Holzapfel(I1_ref, 0, l1_reg)
    terms = I1_out.get_shape().as_list()[1] 
    
    ALL_I_out = I1_out

    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms*2



#%% PI-CANN
# Define network block
def SingleInvNet(I1_ref, idi, L1):
    
    # Layer 1. order
    I_1_w11 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=None,name='w'+str(1+idi)+'1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp,name='w'+str(2+idi)+'1')(I1_ref)
    
    # I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)


    # Layer 2. order
    I_1_w41 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=None,name='w'+str(3+idi)+'1')(Square_Layer(I1_ref))
    I_1_w51 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp,name='w'+str(4+idi)+'1')(Square_Layer(I1_ref))
    # I_1_w61 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(6+idi)+'1')(Square_Layer(I1_ref))

    collect = [I_1_w11, I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out


def BulkNet(J, idi, L1):
    logJ = keras.layers.Lambda(lambda x: tf.math.log(x))(J)

    # Layer 1. order
    # I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                              use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(1 + idi) + '1')(logJ)

    # I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)

    # Layer 2. order
    I_1_w41 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=None, name='w' + str(2 + idi) + '1')(Square_Layer(logJ))
    I_1_w51 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp, name='w' + str(3 + idi) + '1')(
        Square_Layer(logJ))
    # I_1_w61 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(6+idi)+'1')(Square_Layer(I1_ref))

    collect = [I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out

def SinglePrincipalStretchNet(lambda_in, idi, L1):
    logLambda = keras.layers.Lambda(lambda x: tf.math.log(x))(lambda_in)

    # Layer 1. order
    # I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                              use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp_minus_x, name='w' + str(1 + idi) + '1')(logLambda)

    # I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(3+idi)+'1')(I1_ref)

    # Layer 2. order
    I_1_w41 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=None, name='w' + str(2 + idi) + '1')(Square_Layer(logLambda))
    I_1_w51 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp, name='w' + str(3 + idi) + '1')(
        Square_Layer(logLambda))

    # Layer 3. order
    I_1_w61 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=None, name='w' + str(4 + idi) + '1')(Cube_Layer(logLambda))
    I_1_w71 = keras.layers.Dense(1, kernel_initializer=initializer_J, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=activation_Exp, name='w' + str(5 + idi) + '1')(
        Cube_Layer(logLambda))
    # I_1_w61 = keras.layers.Dense(1,kernel_initializer=initializer_log,kernel_constraint=keras.constraints.NonNeg(),
    #                              kernel_regularizer=keras.regularizers.l2(L1),
    #                           use_bias=False, activation=activation_ln,name='w'+str(6+idi)+'1')(Square_Layer(I1_ref))

    collect = [I_1_w21, I_1_w41, I_1_w51, I_1_w61, I_1_w71]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out


def MixedNet(I1_ref, I2_ref, J_in, idi, L1):
    logJ = keras.layers.Lambda(lambda x: tf.math.log(x))(J_in)

    # Layer 1. order
    I1_coefficient = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=Exp, name='w' + str(1 + idi) + '1')(logJ) # Raising to a power
    I1_term = keras.layers.Lambda(lambda x: x[0] * x[1])([I1_ref, I1_coefficient])

    I2_coefficient = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularizers.l1(L1),
                                 use_bias=False, activation=Exp, name='w' + str(2 + idi) + '1')(logJ) # Raising to a power
    I2_term = keras.layers.Lambda(lambda x: x[0] * x[1])([I2_ref, I2_coefficient])

    collect = [I1_term, I2_term]
    # collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect

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


def StrainEnergyCANN(l1_reg=0.01, include_mixed=False):

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
    J_out = BulkNet(J_in, terms, l1_reg)
    terms += J_out.shape[1]
    if include_mixed:
        Mixed_out = MixedNet(I1_ref, I2_ref, J_in, terms, l1_reg) # This output is a list of 2 tensors instead of a single tensor that was concatenated
        terms += len(Mixed_out)
    else:
        Mixed_out = []

    ALL_I_out = [I1_out,I2_out,J_out] + Mixed_out
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, J_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms


def StrainEnergyPrincipalStretch(l1_reg=0.01):

    # Inputs defined
    lambda_1 = tf.keras.Input(shape=(1,), name='lambda_1_psi')
    lambda_2 = tf.keras.Input(shape=(1,), name='lambda_2_psi')
    lambda_3 = tf.keras.Input(shape=(1,), name='lambda_3_psi')

    lambda_in = tf.keras.Input(shape=(1,), name='lambda_in')
    All_out = SinglePrincipalStretchNet(lambda_in, 0, l1_reg)
    All_out = tf.keras.layers.concatenate([All_out],axis=1)

    single_lambda_out = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularizers.l1(l1_reg),
                           use_bias=False, activation=None,name='wx2_psi')(All_out)
    terms = All_out.shape[1]
    single_principal_stretch_model = keras.models.Model(inputs=[lambda_in], outputs=[single_lambda_out], name='SinglePrincipalStretch')

    lambda_out_1 = single_principal_stretch_model([lambda_1])
    lambda_out_2 = single_principal_stretch_model([lambda_2])
    lambda_out_3 = single_principal_stretch_model([lambda_3])
    psi_out = keras.layers.Lambda(lambda x: tf.concat([x[0] + x[1] + x[2]], axis=1), name='lambda_out_combined')([lambda_out_1, lambda_out_2, lambda_out_3])

    # print(terms)
    # print(psi_out)
    # assert False
    
    Psi_model = keras.models.Model(inputs=[lambda_1, lambda_2, lambda_3], outputs=[psi_out], name='Psi')
    
    return Psi_model, terms

