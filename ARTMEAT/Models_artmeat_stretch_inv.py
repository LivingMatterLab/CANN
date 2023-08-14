"""
Last modified August 2023 - Corrected Version

@author: Kevin Linka, Skyler St. Pierre
"""


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# Initializers
initializer_1 = 'glorot_normal'
initializer_log = 'glorot_normal'
initializer_exp = tf.keras.initializers.RandomUniform(minval=0.,
                                                      maxval=0.1)


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


# Custom Layer activation functions
def Exp_fun(arg, a, b):
    # Note that in Table 3: w(1,?) = b and w(2,?) = a/b*wx2 for consistency w/ Eq 16
    # wx2 refers to the weight from W_ANN
    return (a / b) * (tf.math.exp(b * (arg - 1.0)) - 1.0)


def Fun_ln(x, a, b):
    # Note that in Table 3: w(1,?) = b and w(2,?) = 1/2*a/b*wx2 for consistency w/ Eq 16
    # wx2 refers to the weight from W_ANN
    return -0.5 * a / b * tf.math.log(1.0 - b * x)


# Valanis–Landel model layer (7 Terms)
class VL_Layer(keras.layers.Layer):

    def __init__(self, nameU, init, sign):  # created in order of Variable initialization
        super(VL_Layer, self).__init__()
        self.nameU = nameU
        self.init = init  # initialization for trainable power
        self.sign = sign  # positive or negative
        init2 = np.random.uniform(0.05, 0.1, 1)[0]

        self.w1 = tf.Variable(initial_value=init2, name='O1' + self.nameU, constraint=keras.constraints.NonNeg(),
                              dtype=tf.float32, trainable=True)
        self.w2 = tf.Variable(initial_value=init2, name='O2' + self.nameU, constraint=keras.constraints.NonNeg(),
                              dtype=tf.float32, trainable=True)
        self.w3a = tf.Variable(initial_value=init2, name='O3a' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w3b = tf.Variable(initial_value=init2, name='O3b' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w4a = tf.Variable(initial_value=init2, name='O4a' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w4b = tf.Variable(initial_value=init2, name='O4b' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w5 = tf.Variable(initial_value=init2, name='O5' + self.nameU, constraint=keras.constraints.NonNeg(),
                              dtype=tf.float32, trainable=True)
        self.w6a = tf.Variable(initial_value=init2, name='O6a' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w6b = tf.Variable(initial_value=init2, name='O6b' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w7a = tf.Variable(initial_value=init2, name='O7a' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.w7b = tf.Variable(initial_value=init2, name='O7b' + self.nameU, constraint=keras.constraints.NonNeg(),
                               dtype=tf.float32, trainable=True)
        self.x = tf.Variable(initial_value=init, name='alpha' + self.nameU, dtype=tf.float32, trainable=True)


    def get_config(self):
        config = super().get_config().copy()

        return config

    def call(self, I_d):  # forward computation
        I1_in, I2_in = tf.split(I_d, num_or_size_splits=2, axis=1)

        Q = (tf.math.pow(I1_in, 2) - 3.0 * I2_in) + 0.001
        R = ((-9.0 * I1_in * I2_in) + 27.0 + (2.0 * tf.math.pow(I1_in, 3)))
        Theta = tf.math.acos(R / (2.0 * tf.math.pow(Q, 3 / 2)))

        # Computes stretches^2
        Stretch_1 = 1.0 / 3.0 * (I1_in + 2.0 * tf.math.sqrt(Q) * tf.math.cos(
            1.0 / 3.0 * (Theta + 2.0 * np.pi * (1.0 - 1.0))))
        Stretch_2 = 1.0 / 3.0 * (I1_in + 2.0 * tf.math.sqrt(Q) * tf.math.cos(
            1.0 / 3.0 * (Theta + 2.0 * np.pi * (2.0 - 1.0))))
        Stretch_3 = 1.0 / 3.0 * (I1_in + 2.0 * tf.math.sqrt(Q) * tf.math.cos(
            1.0 / 3.0 * (Theta + 2.0 * np.pi * (3.0 - 1.0))))

        # alpha(neg) and alpha(pos) are w1,2 and w2,2 in Eq 16
        # exponential function activation is used to get self.x to change its value drastically from its initialized value
        # the derivative of the exponential fn has a large slope, so the gradient will be large during backpropagation
        # this means we can "discover" exponents that are very small (<5) or very large (>40) with the same initial value
        alpha = self.sign * activation_Exp(self.x)  # trainable weight activated with exponential fn

        O1_2 = tf.math.pow(tf.math.sqrt(Stretch_1), self.sign*2)  # stretch^2
        O2_2 = tf.math.pow(tf.math.sqrt(Stretch_2), self.sign*2)
        O3_2 = tf.math.pow(tf.math.sqrt(Stretch_3), self.sign*2)

        O1_4 = tf.math.pow(tf.math.sqrt(Stretch_1), self.sign*4)  # stretch^4
        O2_4 = tf.math.pow(tf.math.sqrt(Stretch_2), self.sign*4)
        O3_4 = tf.math.pow(tf.math.sqrt(Stretch_3), self.sign*4)

        O1_al = tf.math.pow(tf.math.sqrt(Stretch_1), alpha)  # stretch^alpha
        O2_al = tf.math.pow(tf.math.sqrt(Stretch_2), alpha)
        O3_al = tf.math.pow(tf.math.sqrt(Stretch_3), alpha)

        T1 = self.w1 * (O1_2 + O2_2 + O3_2 - 3.0)  # stretch^2 term
        T2 = self.w2 * (O1_al + O2_al + O3_al - 3.0)  # stretch^alpha term; w2,2 & w2,9 in Eq 16 are calculated as self.w2*wx2 (rather than reporting both separately/redundantly)
        T3 = Exp_fun(O1_2, self.w3a, self.w3b) + Exp_fun(O2_2, self.w3a, self.w3b) + Exp_fun(O3_2, self.w3a, self.w3b)  # exp(stretch^2) term
        T4 = Fun_ln(O1_2, self.w4a, self.w4b) + Fun_ln(O2_2, self.w4a, self.w4b) + Fun_ln(O3_2, self.w4a, self.w4b)  # ln(stretch^2) term
        #
        T5 = self.w5 * (O1_4 + O2_4 + O3_4 - 3.0)  # stretch^4 term
        T6 = Exp_fun(O1_4, self.w6a, self.w6b) + Exp_fun(O2_4, self.w6a, self.w6b) + Exp_fun(O3_4, self.w6a, self.w6b)  # exp(stretch^4) term
        T7 = Fun_ln(O1_4, self.w7a, self.w7b) + Fun_ln(O2_4, self.w7a, self.w7b) + Fun_ln(O3_4, self.w7a, self.w7b)  # ln(stretch^4) term

        collect = [T1, T2, T3, T4, T5, T6, T7]
        # collect_out = T1  # T2 if you want only trainable power term, T1 for Mooney Rivlin (+/-2)
        collect_out = tf.keras.layers.concatenate(collect, axis=1)

        return collect_out


def StrainEnergy_stretch_VL(reg, pen):  # Valanis-Landel type stretch-based model
    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')

    I_in = tf.keras.layers.concatenate([I1_in, I2_in], axis=1)

    Oneg = VL_Layer('neg', 2., -1.0)(I_in)  # negative powers
    Opos = VL_Layer('pos', 2., 1.0)(I_in)  # positive powers

    ALL_I_out_arr = [Oneg, Opos]  # build negative then positive

    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out_arr, axis=1)
    # ALL_I_out = Oneg  # if only want NH or BK just take the correct pos/neg layer

    # second layer
    W_ANN = keras.layers.Dense(1, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.NonNeg(),
                               use_bias=False, kernel_regularizer=regularize(reg, pen), activation=None,
                               name='wx2')(ALL_I_out)

    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[W_ANN], name='Psi')

    amount_terms = 14  # 7 positive + 7 negative terms

    return Psi_model, amount_terms


def SingleInvNetStretch(I1_ref, reg, pen):
    I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 use_bias=False, activation=None, kernel_regularizer=regularize(reg, pen))(I1_ref)
    return I_1_w11


def princStretch(inputs):
    (I1_in, I2_in) = inputs

    Q = (tf.math.pow(I1_in, 2) - 3.0 * I2_in) + 0.001
    R = ((-9.0 * I1_in * I2_in) + 27.0 + (2.0 * tf.math.pow(I1_in, 3)))
    Theta = tf.math.acos(R / (2.0 * tf.math.pow(Q, 3 / 2)))

    Stretch_1 = 1.0 / 3.0 * (
                I1_in + 2.0 * tf.math.sqrt(Q) * tf.math.cos(1.0 / 3.0 * (Theta + 2.0 * np.pi * (1.0 - 1.0))))
    Stretch_2 = 1.0 / 3.0 * (
                I1_in + 2.0 * tf.math.sqrt(Q) * tf.math.cos(1.0 / 3.0 * (Theta + 2.0 * np.pi * (2.0 - 1.0))))
    Stretch_3 = 1.0 / 3.0 * (
                I1_in + 2.0 * tf.math.sqrt(Q) * tf.math.cos(1.0 / 3.0 * (Theta + 2.0 * np.pi * (3.0 - 1.0))))

    return tf.math.sqrt(Stretch_1), tf.math.sqrt(Stretch_2), tf.math.sqrt(Stretch_3), Q


def StrainEnergy_stretch(reg, pen):
    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')

    Stretch_1, Stretch_2, Stretch_3, Q = keras.layers.Lambda(function=princStretch,
                                                             name='P_stretch')([I1_in, I2_in])
    ALL_I_out_arr = []
    for i in range(-30, 33, 3):  # define range of fixed powers
        if i != 0:
            stretch_out = SingleInvNetStretch(Stretch_1 ** i + Stretch_2 ** i + Stretch_3 ** i - 3.0, reg, pen)
            ALL_I_out_arr.append(stretch_out)
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out_arr, axis=1)
    amount_terms = len(ALL_I_out_arr)  # only stretches

    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=[ALL_I_out], name='Psi')

    return Psi_model, amount_terms


def SingleInvNet(I1_ref, idi, reg, pen):
    # Invariant
    I_1_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(1 + idi) + '1')(I1_ref)
    I_1_w21 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(2 + idi) + '1')(I1_ref)

    I_1_w31 = keras.layers.Dense(1, kernel_initializer=initializer_log, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_ln, name='w' + str(3 + idi) + '1')(I1_ref)
    # Invariant squared
    I_1_w41 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None, name='w' + str(4 + idi) + '1')(tf.math.square(I1_ref))
    I_1_w51 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp, name='w' + str(5 + idi) + '1')(
        tf.math.square(I1_ref))
    I_1_w61 = keras.layers.Dense(1, kernel_initializer=initializer_log, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_ln, name='w' + str(6 + idi) + '1')(
        tf.math.square(I1_ref))

    collect = [I_1_w11, I_1_w21, I_1_w31, I_1_w41, I_1_w51, I_1_w61]
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
