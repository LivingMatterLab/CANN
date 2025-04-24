"""
Last modified April 2024

@author: Kevin Linka, Skyler St. Pierre
"""


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# Initializers
initializer_1 = 'glorot_normal'
initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1)


# Regularizers
def regularize(reg, pen):
    if reg == 'L2':
        return keras.regularizers.l2(pen)
    if reg == 'L1':
        return keras.regularizers.l1(pen)


# Activation functions for built-in Layers
def activation_Exp(x):
    return 1.0 * (tf.math.exp(x) - 1.0)


def activation_ln(x):
    return -1.0 * tf.math.log(1.0 - x)


def SingleInvNet(I1_ref, idi, reg, pen):
    # Invariant
    I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(2 + idi) + '1')(I1_ref)
    # Invariant squared
    I_1_w31 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(3 + idi) + '1')(tf.math.square(I1_ref))
    I_1_w41 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(4 + idi) + '1')(
        tf.math.square(I1_ref))

    collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)

    return collect_out


def StrainEnergy_invariant(reg, pen):
    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')

    # Invariants reference confi
    I1_ref = keras.layers.Lambda(lambda x: (x - 3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x - 3.0))(I2_in)

    I1_out = SingleInvNet(I1_ref, 0, reg, pen)
    terms = I1_out.get_shape().as_list()[1]
    I2_out = SingleInvNet(I2_ref, terms, reg, pen)

    ALL_I_out = [I1_out, I2_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out, axis=1)

    # second layer
    W_ANN = keras.layers.Dense(1, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.NonNeg(),
                               kernel_regularizer=regularize(reg, pen),
                               use_bias=False, activation=None, name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')

    return Psi_model, terms * 2
