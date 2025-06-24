# File containing all continuum mechanics formulas for computing invariants and stress from the constitutive model

from src.CANN.util_functions import *
import keras.backend as K
import tensorflow as tf
from tensorflow import keras

""""
Computes maximum reference values of I4theta and I4negtheta for each loading mode given the maximum values of the other invariants (I1, I2, I4w, I4z, I4s, etc)
:param Is_max: 10 x 8 numpy array where Is_max[i,j] is the maximum reference value of the jth invariant in experiment i (output of get_max_inv_mesh)
:return: Is_max - length 20 numpy array that contains the maximum reference values of the I4theta and I4negtheta for each experiment 
"""
def calculate_I4theta_max(Is_max):
    theta = np.pi / 3
    I4w = Is_max[:, 2] + 1
    I4s = Is_max[:, 4] + 1
    I8ws = Is_max[:, 6]
    I4theta = I4w * np.cos(theta) ** 2 + I4s * np.sin(theta) ** 2 + I8ws * np.sin(2 * theta)
    I4negtheta = I4w * np.cos(theta) ** 2 + I4s * np.sin(theta) ** 2 - I8ws * np.sin(2 * theta)
    return np.concatenate([I4theta, I4negtheta], axis=0) - 1

def get_max_inv_mesh(lam_ut_all, modelFitMode):
    """"
    Computes maximum reference values of each invariant for each loading mode given the stretch values for each loading mode
    :param lam_ut_all: 2 x 5 x 2 list of 1d numpy arrays that contain x and y stretch values for each loading mode. First index corresponds to mounting direction (90 vs 45), second index corresponds to experiment type (strip x, off x, etc), third index corresponds to strain direction (x vs y)
    :param modelFitMode: String corresponding to which loading directions are used for training
    :return: Is_max - 10 x 8 numpy array where Is_max[i,j] is the maximum value of the jth invariant in experiment i
    """
    # Compute invariant values for 90 and 45 degree orientation separately
    Is_90 = np.array([get_inv_value_biax_90(x) for x in lam_ut_all[0]])
    Is_45 = np.array([get_inv_value_biax_45(x) for x in lam_ut_all[1]])
    Is = np.stack((Is_90, Is_45))

    # Compute maximum within each experiment and reshape so first index corresponds to experiment index and second to which invariant
    Is_max = np.max(Is, 2)
    Is_max = Is_max.reshape((-1, 8))

    # Iterate through modelFitMode and zero out any experiments not used for training
    if modelFitMode.isnumeric():
        modes = [int(x) for x in modelFitMode]
        for i in range(Is_max.shape[0]):
            if i not in modes:
                Is_max[i, :] = 0
    return Is_max

def get_inv_value_biax_90(stretch):
    """
    Compute invariant reference values given stretch in 0-90 orientation
    :param stretch: list of 2 numpy arrays of length n that contain the warp and shute stretch values, respectively
    :return: n x 8 numpy array containing the invariant values corresponding to each stretch value
    """
    Is = np.zeros((stretch[0].shape[0], 8))
    # Extract stretch in w, s, and z directions
    Stretch_w = stretch[0]
    Stretch_s = stretch[1]
    Stretch_z = 1. / (Stretch_w * Stretch_s)

    # Compute invariants using well known continuum mechanics formulas
    # Order of invariants is I1, I2, I4w, I4z, I4s, I8wz, I8ws, I8sz
    Is[:, 0] = Stretch_w ** 2 + Stretch_s ** 2 + Stretch_z ** 2 - 3
    Is[:, 1] = Stretch_w ** (-2) + Stretch_s ** (-2) + Stretch_z ** (-2) - 3
    Is[:, 2] = Stretch_w ** 2 - 1
    Is[:, 3] = Stretch_z ** 2 - 1
    Is[:, 4] = Stretch_s ** 2 - 1
    return Is

def get_inv_value_biax_45(stretch):
    """
    Compute invariant reference values given stretch in 45-135 orientation
    :param stretch: list of 2 numpy arrays of length n that contain the x and y stretch values, respectively
    :return: n x 8 numpy array containing the invariant values corresponding to each stretch value
    """
    Is = np.zeros((stretch[0].shape[0], 8))
    # Extract stretch in x, y, and z directions
    Stretch_x = stretch[0]
    Stretch_y = stretch[1]
    Stretch_z = 1. / (Stretch_x * Stretch_y)

    # Compute invariants using well known continuum mechanics formulas
    # Order of invariants is I1, I2, I4w, I4z, I4s, I8wz, I8ws, I8sz
    Is[:, 0] = Stretch_x ** 2 + Stretch_y ** 2 + Stretch_z ** 2 - 3
    Is[:, 1] = Stretch_x ** (-2) + Stretch_y ** (-2) + Stretch_z ** (-2) - 3
    Is[:, 2] = (Stretch_x ** 2 + Stretch_y ** 2) / 2 - 1
    Is[:, 3] = Stretch_z ** 2 - 1
    Is[:, 4] = (Stretch_x ** 2 + Stretch_y ** 2) / 2 - 1
    Is[:, 6] = np.abs(Stretch_x ** 2 - Stretch_y ** 2) / 2
    return Is



# Continuum mechanics stress definition for uniaxial tension
def Stress_calc_TC(inputs):
    (dPsidI1, dPsidI2, Stretch) = inputs

    #   calculate cauchy stress sigma
    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')

    minus = two * (dPsidI1 * 1 / K.square(Stretch) + dPsidI2 * 1 / K.pow(Stretch, 3))
    stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus

    return stress


# Simple stress P12
def Stress_cal_SS(inputs):
    (dPsidI1, dPsidI2, gamma) = inputs

    two = tf.constant(2.0, dtype='float32')

    # Shear stress
    stress = two * gamma * (dPsidI1 + dPsidI2)

    return stress

def Stress_cal_w(inputs):
    """
    Compute stress in warp direction given stretch and strain energy partial derivatives (in 0-90 orientation)
    :param inputs: tuple containing strain energy derivatives with respect to I1, I2, and I4w, and stretch in w and s
    :return: Stress in warp direction
    """
    (dWI1,dWI2,dWI4_w,stretch_w, stretch_s) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    stretch_z = one/(stretch_w*stretch_s)
    stress_w = two * (dWI1 * (stretch_w - stretch_z*stretch_z / stretch_w) +
                      dWI2 * (stretch_w*stretch_s*stretch_s - 1/(stretch_w*stretch_w*stretch_w)) +
                              dWI4_w * stretch_w)
    return stress_w

def Stress_cal_w_sq(inputs):
    """
    Compute stress in warp direction given stretch and strain energy partial derivatives (in 0-90 orientation)
    :param inputs: tuple containing strain energy derivatives with respect to I1, I2, and I4w, and stretch in w and s
    :return: Stress in warp direction
    """
    (dWI1, dWI2, dWI4_w, stretch_w, stretch_s) = inputs
    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')

    stretch_z = one / (stretch_w * stretch_s)
    stress_w = 4.0 * (dWI1 * (stretch_w - stretch_z * stretch_z / stretch_w) ** 2 +
                      dWI2 * (stretch_w * stretch_s * stretch_s - 1 / (stretch_w * stretch_w * stretch_w)) ** 2 +
                      dWI4_w * stretch_w ** 2)
    return stress_w

    """
    Compute stress in shute direction given stretch and strain energy partial derivatives (in 0-90 orientation)
    :param inputs: tuple containing strain energy derivatives with respect to I1, I2, and I4s, and stretch in w and s
    :return: Stress in shute direction
    """
def Stress_cal_s(inputs):
    (dWI1,dWI2,dWI4_s,stretch_w,stretch_s) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    #according to Holzapfel 2009
    stretch_z = one/(stretch_w*stretch_s)
    stress_s = two * (dWI1 * (stretch_s - stretch_z*stretch_z / stretch_s) +
                      dWI2 * (stretch_w*stretch_w*stretch_s - 1/(stretch_s*stretch_s*stretch_s)) +
                              dWI4_s * stretch_s)
    return stress_s

def Stress_cal_s_sq(inputs):
    (dWI1,dWI2,dWI4_s,stretch_w,stretch_s) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    #according to Holzapfel 2009
    stretch_z = one/(stretch_w*stretch_s)
    stress_s = 4.0 * (dWI1 * (stretch_s - stretch_z*stretch_z / stretch_s) ** 2 +
                      dWI2 * (stretch_w*stretch_w*stretch_s - 1/(stretch_s*stretch_s*stretch_s)) ** 2 +
                              dWI4_s * stretch_s ** 2)
    return stress_s

"""
    Compute stress in x direction given stretch and strain energy partial derivatives (in 45-135 orientation)
    :param inputs: tuple containing strain energy derivatives with respect to I1, I2, I4w, I4s, and I8ws , and stretch in x and y
    :return: Stress in x direction
"""
def Stress_cal_x_45(inputs):
    (dWI1,dWI2,dWI4w, dWI4s, dWI8ws, stretch_x, stretch_y) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    stretch_z = one/(stretch_x*stretch_y)
    stress_x = two * (dWI1 * (stretch_x - stretch_z*stretch_z / stretch_x) +
                      dWI2 * (stretch_x*stretch_y*stretch_y - one/(stretch_x*stretch_x*stretch_x))) + \
               (dWI4w + dWI4s + dWI8ws) * stretch_x


    return stress_x

def Stress_cal_x_45_sq(inputs):
    (dWI1,dWI2,dWI4w, dWI4s, dWI8ws, stretch_x, stretch_y) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    stretch_z = one/(stretch_x*stretch_y)
    stress_x = 4.0 * (dWI1 * (stretch_x - stretch_z*stretch_z / stretch_x) ** 2 +
                      dWI2 * (stretch_x*stretch_y*stretch_y - one/(stretch_x*stretch_x*stretch_x)) ** 2) + \
               dWI4w * stretch_x ** 2 + dWI4s * stretch_x ** 2


    return stress_x

"""
    Compute stress in y direction given stretch and strain energy partial derivatives (in 45-135 orientation)
    :param inputs: tuple containing strain energy derivatives with respect to I1, I2, I4w, I4s, and I8ws , and stretch in x and y
    :return: Stress in y direction
"""
def Stress_cal_y_45(inputs):
    (dWI1,dWI2,dWI4f, dWI4n, dWI8fn, stretch_x, stretch_y) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    stretch_z = one/(stretch_x*stretch_y)
    stress_y = two * (dWI1 * (stretch_y - stretch_z * stretch_z / stretch_y) +
                      dWI2 * (stretch_x * stretch_x * stretch_y - one / (stretch_y * stretch_y * stretch_y))) + \
               (dWI4f + dWI4n - dWI8fn) * stretch_y

    return stress_y

def Stress_cal_y_45_sq(inputs):
    (dWI1,dWI2,dWI4w, dWI4s, dWI8fn, stretch_x, stretch_y) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')

    stretch_z = one/(stretch_x*stretch_y)
    stress_y = 4.0 * (dWI1 * (stretch_y - stretch_z * stretch_z / stretch_y) ** 2 +
                      dWI2 * (stretch_x * stretch_x * stretch_y - one / (stretch_y * stretch_y * stretch_y)) ** 2) + \
               dWI4w * stretch_y ** 2 + dWI4s * stretch_y ** 2

    return stress_y

# Complete model architecture definition given strain energy model
def modelArchitecture(Region, Psi_model):
    if Region == 'mesh':
        Stretch_w = keras.layers.Input(shape=(1,),
                                         name='Stretch_w')
        Stretch_s = keras.layers.Input(shape=(1,),
                                         name='Stretch_s')
        I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_w, Stretch_s])
        I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
            [Stretch_w, Stretch_s])
        I4w = keras.layers.Lambda(lambda x: x ** 2)(Stretch_w)
        I4s = keras.layers.Lambda(lambda x: x ** 2)(Stretch_s)
        I8ws = keras.layers.Lambda(lambda x: x ** 0 - 1)(Stretch_s)
        Psi = Psi_model([I1, I2, I4w, I4s, I8ws])
        dWI1 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I1])
        dWdI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I2])
        dWdI4w = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4w])
        dWdI4s = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4s])
        Stress_w = keras.layers.Lambda(function=Stress_cal_w,
                                       name='Stress_w')([dWI1, dWdI2, dWdI4w, Stretch_w, Stretch_s])
        Stress_s = keras.layers.Lambda(function=Stress_cal_s,
                                       name='Stress_s')([dWI1, dWdI2, dWdI4s, Stretch_w, Stretch_s])

        model_90 = keras.models.Model(inputs=[Stretch_w, Stretch_s], outputs=[Stress_w, Stress_s])

        # 45 degree offset

        Stretch_x = keras.layers.Input(shape=(1,),
                                       name='Stretch_x')
        Stretch_y = keras.layers.Input(shape=(1,),
                                       name='Stretch_y')
        I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_x, Stretch_y])
        I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
            [Stretch_x, Stretch_y])
        I4w = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
        I4s = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
        I8ws = keras.layers.Lambda(lambda x: (x[0] ** 2 - x[1] ** 2) / 2)([Stretch_x, Stretch_y])
        Psi = Psi_model([I1, I2, I4w, I4s, I8ws])

        dWI1 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I1])
        dWdI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I2])
        dWdI4w = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4w])
        dWdI4s = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4s])
        dWdI8ws = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I8ws])

        Stress_x = keras.layers.Lambda(function=Stress_cal_x_45,
                                       name='Stress_x')([dWI1, dWdI2, dWdI4w, dWdI4s, dWdI8ws, Stretch_x, Stretch_y])
        Stress_y = keras.layers.Lambda(function=Stress_cal_y_45,
                                       name='Stress_y')([dWI1, dWdI2, dWdI4w, dWdI4s, dWdI8ws, Stretch_x, Stretch_y])

        model_45 = keras.models.Model(inputs=[Stretch_x, Stretch_y], outputs=[Stress_x, Stress_y])

        models = [model_90, model_45]
        inputs = [model.inputs for model in models]
        outputs = [model.outputs for model in models]
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        # print(model.inputs)
        #
        return model, model, Psi_model, model

    else:
        # Stretch and Gamma as input
        Stretch = keras.layers.Input(shape=(1,),
                                     name='Stretch')
        Gamma = keras.layers.Input(shape=(1,),
                                   name='gamma')

        # specific Invariants UT
        I1_UT = keras.layers.Lambda(lambda x: x ** 2 + 2.0 / x)(Stretch)
        I2_UT = keras.layers.Lambda(lambda x: 2.0 * x + 1 / x ** 2)(Stretch)
        # specific Invariants SS
        I1_SS = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma)
        I2_SS = keras.layers.Lambda(lambda x: x ** 2 + 3.0)(Gamma)

        # % load specific models
        Psi_UT = Psi_model([I1_UT, I2_UT])
        Psi_SS = Psi_model([I1_SS, I2_SS])

        Psi_UT_out = keras.layers.Lambda(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=1), -1))(Psi_UT)
        Psi_SS_out = keras.layers.Lambda(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=1), -1))(Psi_SS)

        # derivative UT
        dWI1_UT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I1_UT])
        dWdI2_UT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I2_UT])
        # derivative SS
        dWI1_SS = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS, I1_SS])
        dWdI2_SS = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS, I2_SS])

        # Stress UT
        Stress_UT = keras.layers.Lambda(function=Stress_calc_TC,
                                        name='Stress_UT')([dWI1_UT, dWdI2_UT, Stretch])
        # Stress SS
        Stress_SS = keras.layers.Lambda(function=Stress_cal_SS,
                                        name='Stress_SS')([dWI1_SS, dWdI2_SS, Gamma])

        # Define models for computing stress
        model_UT = keras.models.Model(inputs=Stretch, outputs=Stress_UT)
        model_SS = keras.models.Model(inputs=Gamma, outputs=Stress_SS)
        # Combined stress model
        model = keras.models.Model(inputs=[model_UT.inputs, model_SS.inputs], outputs=[model_UT.outputs, model_SS.outputs])

        return model_UT, model_SS, Psi_model, model
