#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras import regularizers

#%%
initializer_1 = 'glorot_normal'
initializer_zero = 'zeros'
initializer_log = 'glorot_normal'
initializer_exp = tf.keras.initializers.RandomUniform(minval=0.5, maxval=2.5) # worked off and on, starts with huge residual
initializer_exp2 = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.1)


#%% energy
# Self defined activation functions for exp 
def activation_Exp(x):
    return 1.0*(tf.math.exp(x) -1.0) 

def activation_ln(x):
    return -1.0*tf.math.log(1.0 - (x))


# Define network block
def Net_Holzapfel(I1_ref,idi):
    
    # Layer 2. order
    I_1_w51 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=activation_Exp,name='w'+str(12)+'1')(tf.math.square(I1_ref))

    
    return I_1_w51

# Define Holzapfel energy
def StrainEnergyHolzapfel():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4_in = tf.keras.Input(shape=(1,), name='I4')
    I5_in = tf.keras.Input(shape=(1,), name='I5')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4_in)
    I5_ref = keras.layers.Lambda(lambda x: (x-1.0))(I5_in)
    
    L1 = 0.000
    I1_out = Net_NeoHook(I1_ref,0)
    terms = I1_out.get_shape().as_list()[1] 

    I4_out = Net_Holzapfel(I4_ref,3*terms)


    
    ALL_I_out = [I1_out,I4_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=keras.regularizers.l2(L1),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4_in, I5_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, 16



#%%

# Define Invariant building-blocks
def SingleInvNet16_i5(I1_ref,idi,L1):
    
    # Layer 1. order
    I_1_w11 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=None,name='w'+str(1+idi)+'1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=activation_Exp,name='w'+str(2+idi)+'1')(I1_ref)


    # Layer 2. order
    I_1_w31 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=None,name='w'+str(3+idi)+'1')(tf.math.square(I1_ref))
    I_1_w41 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L1),
                                 use_bias=False, activation=activation_Exp,name='w'+str(4+idi)+'1')(tf.math.square(I1_ref))


    collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41]        
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    
    
    return collect_out



# Define CANN Strain energy
def StrainEnergy_i5():

    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4_in = tf.keras.Input(shape=(1,), name='I4')
    I5_in = tf.keras.Input(shape=(1,), name='I5')
            
    
    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4_in)
    I5_ref = keras.layers.Lambda(lambda x: (x-1.0))(I5_in)
    
    L1 = 0.000
    I1_out = SingleInvNet16_i5(I1_ref,0,L1)
    terms = I1_out.get_shape().as_list()[1] 
    I2_out = SingleInvNet16_i5(I2_ref,terms,L1)
    I4_out = SingleInvNet16_i5(I4_ref,2*terms,L1)
    I5_out = SingleInvNet16_i5(I5_ref,3*terms,L1)
    
    
    ALL_I_out = [I1_out,I2_out,I4_out,I5_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)
    
    # second layer
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=keras.regularizers.l2(L1),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4_in, I5_in], outputs=[W_ANN], name='Psi')
    
    return Psi_model, terms*4



#%% modeling


# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

# Definition of stress
def Stress_xx_I5_BT(inputs):

    (dPsidI1, dPsidI2, dWdI4, dWdI5, Stretch, Stretch_z, I1, h11, h11_i5) = inputs

#   calculate cauchy stress sigma
    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     
    four = tf.constant(4.0,dtype='float32')     

    stress_1 = two * ( dPsidI1 +         I1 *dPsidI2 ) * (Stretch**two -Stretch_z**two)
    stress_2 = two*dPsidI2 *(Stretch_z**four  - Stretch**four)
    stress_3 = two*dWdI4*h11 
    stress_4 = four*dWdI5*h11_i5 
  
    return stress_1 + stress_2 + stress_3 + stress_4


# Define H-layer
class H_Layer_FungBiax_I4I5(keras.layers.Layer):
    
    def __init__(self, nameU, setAl, init):
        super(H_Layer_FungBiax_I4I5, self).__init__()
        self.nameU = nameU
        self.setAl = setAl
        self.init =  init

        
        self.alpha =  tf.Variable(initial_value=self.init, name=self.nameU, constraint=keras.constraints.NonNeg(), dtype=tf.float32, trainable=self.setAl)
        
    def get_config(self):
    
        config = super().get_config().copy()

        return config
        
    def call(self, lam):
        
        
        lamx, lamy = tf.split(lam, num_or_size_splits=2, axis=1) 

        al = tf.nn.relu(self.alpha)

        h11_i4  =  tf.math.multiply( tf.math.pow(lamx,2), tf.math.square(tf.math.cos(al) ) )
        h22_i4  =  tf.math.multiply( tf.math.pow(lamy,2), tf.math.square(tf.math.sin(al) ) )
        
        h11_i5  =  tf.math.multiply( tf.math.pow(lamx,4), tf.math.square(tf.math.cos(al) ) )
        h22_i5  =  tf.math.multiply( tf.math.pow(lamy,4), tf.math.square(tf.math.sin(al) ) )
                 
        return [h11_i4, h22_i4, h11_i5, h22_i5]




# Complte model architecture definition
def modelArchitecture_I5(Psi_model,setAl, init):
    # Stretch and Gamma as input
    Stretch_x = keras.layers.Input(shape = (1,),
                                  name = 'Stretch_x')
    Stretch_y = keras.layers.Input(shape = (1,),
                                  name = 'Stretch_y')

    # specific Invariants BT
    Stretch_z = tf.keras.layers.Lambda(lambda x: 1/(x[0] * x[1]), name = 'lam_z')([Stretch_x, Stretch_y])
    I1_BT = tf.keras.layers.Lambda(lambda x: x[0]**2 + x[1]**2 +x[2]**2, name = 'I1')([Stretch_x, Stretch_y, Stretch_z])
    I2_BT = tf.keras.layers.Lambda(lambda x: (x[0]**2)*(x[1]**2) + 1/x[0]**2 + 1/x[1]**2, name = 'I2')([Stretch_x, Stretch_y])

    # H-Tensor in-plane komponents
    Stretches_in = tf.keras.layers.concatenate([Stretch_x,Stretch_y], axis=1)
    
    h11, h22, h11_i5, h22_i5 = H_Layer_FungBiax_I4I5('alpha',setAl, init)(Stretches_in)
    # Define I4 in terms of H
    I4_BT = tf.keras.layers.Lambda(lambda x: x[0] + x[1], name = 'I4')([h11, h22])
    I5_BT = tf.keras.layers.Lambda(lambda x: x[0] + x[1], name = 'I5')([h11_i5, h22_i5])

    
    #% Define Strain Energy
    Psi_BT = Psi_model([I1_BT, I2_BT, I4_BT, I5_BT])
    
    # derivative DWdX
    dWI1_BT  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_BT, I1_BT])
    dWdI2_BT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_BT, I2_BT])
    dWdI4_BT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_BT, I4_BT])
    dWdI5_BT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_BT, I5_BT])
    
    # Stress XX
    Stress_xx_BT = keras.layers.Lambda(function = Stress_xx_I5_BT,
                                name = 'Stress_xx_BT')([dWI1_BT, dWdI2_BT, dWdI4_BT, dWdI5_BT,
                                                        Stretch_x, Stretch_z, I1_BT, h11, h11_i5])
    # Stress YY
    Stress_yy_BT = keras.layers.Lambda(function = Stress_xx_I5_BT,
                                name = 'Stress_yy_BT')([dWI1_BT, dWdI2_BT, dWdI4_BT, dWdI5_BT,
                                                        Stretch_y, Stretch_z, I1_BT,  h22, h22_i5])
    
    # Define model 
    model_BT = keras.models.Model(inputs=[Stretch_x, Stretch_y],  outputs= [Stress_xx_BT, Stress_yy_BT])

    
    return model_BT






