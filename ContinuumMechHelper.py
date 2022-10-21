#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:48:53 2022

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


#%% Contiuum Mechanics helper


# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

# Continuum mechanics stress definition for uniaxial tension only
def Stress_calc_UT(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    minus  = two * ( dPsidI1 *             one/ K.square(Stretch)  + dPsidI2 *      one/K.pow(Stretch,3)   ) 
    stress = two * ( dPsidI1 *  Stretch                          + dPsidI2 *  one                      ) - minus

    return stress

# Continuum mechanics stress definition for pure shear only
def Stress_calc_PS(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    minus  = two * ( dPsidI1 *             one/K.pow(Stretch,3)  + dPsidI2 *          one/K.pow(Stretch,3)   ) 
    stress = two * ( dPsidI1 *  Stretch                        + dPsidI2 *  Stretch                      ) - minus

    return stress

# Continuum mechanics stress definition for biaxial tension only
def Stress_calc_ET(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    # minus  = two * ( dPsidI1 *            one/K.pow(Stretch,5)  + dPsidI2 *                   one/K.pow(Stretch,3)   ) 
    # stress = two * ( dPsidI1 *  Stretch                       + dPsidI2 *  K.pow(Stretch,3)                      ) - minus

    stress = two*(dPsidI1 + K.pow(Stretch,2)* dPsidI2)*(Stretch - one/K.pow(Stretch,5))
    
    return stress


# Complte model architecture definition
def ContinuumMechanicsFramework(Psi_model):
    # Stretch and Gamma as input
    Stretch_UT = keras.layers.Input(shape = (1,), name = 'Stretch_UT')
    Stretch_PS = keras.layers.Input(shape = (1,), name = 'Stretch_PS')
    Stretch_ET = keras.layers.Input(shape = (1,), name = 'Stretch_ET')

    # specific Invariants UT
    I1_UT = keras.layers.Lambda(lambda x: x**2   + 2.0/x  )(Stretch_UT)
    I2_UT = keras.layers.Lambda(lambda x: 2.0*x  + 1/x**2 )(Stretch_UT)
    # specific Invariants PS
    I1_PS = keras.layers.Lambda(lambda x: x**2 + 1.0 + 1.0/x**2 )(Stretch_PS)
    I2_PS = keras.layers.Lambda(lambda x: x**2 + 1.0 + 1.0/x**2 )(Stretch_PS)
    # specific Invariants ET
    I1_ET = keras.layers.Lambda(lambda x: 2.0*x**2 + 1.0/x**4 )(Stretch_ET)
    I2_ET = keras.layers.Lambda(lambda x:     x**4 + 2.0/x**2 )(Stretch_ET)
    
    #% load specific models
    Psi_UT = Psi_model([I1_UT, I2_UT])
    Psi_PS = Psi_model([I1_PS, I2_PS])
    Psi_ET = Psi_model([I1_ET, I2_ET])
    
    # derivative UT
    dWdI1_UT  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I1_UT])
    dWdI2_UT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I2_UT])
    # derivative PS
    dWdI1_PS  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_PS, I1_PS])
    dWdI2_PS = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_PS, I2_PS])
    # derivative ET
    dWdI1_ET  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_ET, I1_ET])
    dWdI2_ET = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_ET, I2_ET])
    
    
    # Stress UT
    Stress_UT = keras.layers.Lambda(function = Stress_calc_UT,
                                name = 'Stress_UT')([dWdI1_UT,dWdI2_UT,Stretch_UT])
    # Stress PS
    Stress_PS = keras.layers.Lambda(function = Stress_calc_PS,
                                name = 'Stress_PS')([dWdI1_PS,dWdI2_PS,Stretch_PS])
    # Stress ET
    Stress_ET = keras.layers.Lambda(function = Stress_calc_ET,
                                name = 'Stress_ET')([dWdI1_ET,dWdI2_ET,Stretch_ET])    
    
    # Define model for different load case
    model_UT = keras.models.Model(inputs=Stretch_UT, outputs= Stress_UT)
    model_PS = keras.models.Model(inputs=Stretch_PS, outputs= Stress_PS)
    model_ET = keras.models.Model(inputs=Stretch_ET, outputs= Stress_ET)
    # Combined model
    model = keras.models.Model(inputs=[model_UT.inputs, model_PS.inputs, model_ET.inputs],
                               outputs=[model_UT.outputs, model_PS.outputs, model_ET.outputs])
    
    return model_UT, model_PS, model_ET, model



