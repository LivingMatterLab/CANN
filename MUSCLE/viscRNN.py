import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# custom RNN cell to learn Prony series parameters
class ViscRNNCellGen(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.state_size = [tf.TensorShape([1]), tf.TensorShape([units])]
        self.units = units
        super(ViscRNNCellGen, self).__init__(**kwargs)

    def build(self, input_shape):
        self.tau = self.add_weight(shape=(1, self.units),
                                   initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1.), name='tau')
        self.built = True

    def call(self, inputs, states):
        scaleFactor = tf.constant([1000.])  # scale factor can be adjusted to see if training improves

        (sig0_prev, h_prev) = states  # stored from previous time step

        dt, sig0 = tf.split(inputs, num_or_size_splits=2, axis=1)

        tauPos = tf.nn.relu(self.tau) + tf.constant([1e-5])  # constrain parameters to be positive

        a = tf.math.exp(tf.math.divide(tf.math.multiply(tf.constant([-1.]), dt), tauPos * scaleFactor))

        b = tf.math.exp(tf.math.divide(tf.math.multiply(tf.constant([-1.]), dt),
                                       tf.math.multiply(tf.constant([2.]), tauPos * scaleFactor)))

        dsig = tf.math.subtract(sig0, sig0_prev)

        h = tf.math.add(tf.math.multiply(a, h_prev), tf.math.multiply(b, dsig))

        output = tf.concat([sig0, h], 1)

        return output, (sig0, h)


# custom constraint to have all weights in a layer be positive and sum to 1
class SumToOne(tf.keras.constraints.Constraint):

    def __call__(self, w):
        w2 = tf.nn.relu(w) + 1e-5
        return tf.math.divide(w2, tf.math.reduce_sum(w2))


# constrains weights to be positive
class Positive(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.nn.relu(w)+1e-5


# %%%% Principal-stretch-based model %%%%

# calculate powers of the principal stretches
#  units: number of terms to include
#  inputs: the principal stretches
def pStrEnergyTerms(units, inputs):
    # terms for powers of 1 and -1
    posExps = tf.ones((tf.shape(inputs)[-1],)) * (1.0)
    posPow = tf.math.pow(inputs, posExps)
    posTerm = tf.math.reduce_sum(posPow, axis=-1) - 3.0

    negExps = -1.0 * posExps
    negPow = tf.math.pow(inputs, negExps)
    negTerm = tf.math.reduce_sum(negPow, axis=-1) - 3.0
    terms = tf.stack([negTerm, posTerm], -1)

    # terms for powers of 2 through units and -2 through -units
    for i in range(1, units):
        posExps = tf.ones((tf.shape(inputs)[-1],)) * (i + 1.0)
        posPow = tf.math.pow(inputs, posExps)
        posTerm = tf.math.reduce_sum(posPow, axis=-1) - 3.0

        negExps = -1.0 * posExps
        negPow = tf.math.pow(inputs, negExps)
        negTerm = tf.math.reduce_sum(negPow, axis=-1) - 3.0

        newTerms = tf.stack([negTerm, posTerm], -1)
        terms = tf.concat([terms, newTerms], -1)

    return terms


# build the principal stress-based RNN model
#  numUnits: number of terms to include in the initial stored energy function
#  numHistoryVars: number of terms to include in the relaxation function
def build_pStr(numUnits, numHistoryVars):

    cell = ViscRNNCellGen(numHistoryVars)  # viscoelastic model RNN cell

    stretch = keras.layers.Input(shape=(None, 1), name='input_stretch')  # input: axial stretch

    with tf.GradientTape() as g:
        g.watch(stretch)
        lam1 = keras.layers.Lambda(lambda x: x, name='lam1')(stretch)  # 1st principal stretch
        lam2 = keras.layers.Lambda(lambda x: tf.math.pow(x, -0.5), name='lam2')(stretch)  # 2nd principal stretch
        lam3 = keras.layers.Lambda(lambda x: tf.math.pow(x, -0.5), name='lam3')(stretch)  # 3rd principal stretch

        pStretches = keras.layers.Concatenate(name='principal_stretches')([lam1, lam2, lam3])

        eTerms = keras.layers.Lambda(lambda x: pStrEnergyTerms(numUnits, x),
                                     name='pStrTerms')(pStretches)  # calculate powers of the stretches
        psi = keras.layers.Dense(1, activation='linear', use_bias=False,
                                 kernel_initializer=keras.initializers.RandomUniform(minval=0.01, maxval=2.),
                                 kernel_constraint=Positive(), name='ogdenCoeffs')(eTerms)  # stored energy function

    der = g.gradient(psi, stretch, unconnected_gradients='zero')
    sig0 = keras.layers.Lambda(lambda x: x[0] * x[1],
                               name='initialStress')([stretch, der])  # normal initial Cauchy stress in axial direction

    dt = keras.layers.Input(shape=(None, 1), name='time_step')  # input: time step

    merge = keras.layers.Concatenate(name='rnn_input')([dt, sig0])
    rnnOutput = keras.layers.RNN(cell, return_sequences=True, name='relax_function')(merge)

    out = keras.layers.TimeDistributed(keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomUniform(
                                                              minval=0.01, maxval=1.), use_bias=False,
                                                          kernel_constraint=SumToOne(), name='stress'))(rnnOutput)

    model = keras.models.Model(inputs=[stretch, dt], outputs=out)

    return model


# %%%% Invariant-based model %%%%

# exponential activation function
def activation_Exp(x):
    return 1.0 * (tf.math.exp(x) - 1.0)


# logarithmic activation function
def activation_ln(x):
    return -1.0 * tf.math.log(1.0 - (x))


# gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]


# assembles invariant-based terms for the initial stored energy function
#  I_ref: either I1-3 or I2-3
#  L2: regularization strength
def SingleInvNet6(I_ref, L2):
    initializer_1 = 'glorot_normal'
    initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=0.00001)
    initializer_log = tf.keras.initializers.RandomUniform(minval=0., maxval=0.00001)

    # linear terms
    I_w11 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=None)(I_ref)
    I_w21 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=activation_Exp)(I_ref)
    I_w31 = keras.layers.Dense(1, kernel_initializer=initializer_log, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=activation_ln)(I_ref)

    # quadratic terms
    I_w41 = keras.layers.Dense(1, kernel_initializer=initializer_1, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=None)(tf.math.square(I_ref))
    I_w51 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=activation_Exp)(tf.math.square(I_ref))
    I_w61 = keras.layers.Dense(1, kernel_initializer=initializer_log, kernel_constraint=keras.constraints.NonNeg(),
                                 kernel_regularizer=keras.regularizers.l2(L2),
                                 use_bias=False, activation=activation_ln)(tf.math.square(I_ref))

    collect = [I_w11, I_w21, I_w31, I_w41, I_w51, I_w61]
    collect_out = tf.keras.layers.concatenate(collect)

    return collect_out


# Calculation of Cauchy stress for uniaxial tension/compression only
#  inputs: (dPsidI1, dPsidI2, Stretch, I1)
#   dPsidI1: partial derivative of the initial stored energy function w/ respect to the first invariant I1
#   dPsidI2: partial derivative of the initial stored energy function w/ respect to I2
#   Stretch: the current axial stretch
#   I1: the first invariant
def Stress_calc_UT(inputs):

    (dPsidI1, dPsidI2, Stretch, I1) = inputs

    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')

    minus = two * (dPsidI1 * one / tf.math.square(Stretch) + dPsidI2 * one / tf.math.pow(Stretch, 3))
    P = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus

    sig = P * Stretch

    return sig


# build the invariant-based RNN model
#  n: number of terms in the relaxation function
#  l2: regularization weight for the initial stored energy function
#  rp: regularization weight for the prony series relaxation function
def build_inv(n, l2, rp):
    cell = ViscRNNCellGen(n)  # viscoelastic model RNN cell

    stretch = keras.layers.Input(shape=(None, 1), name='input_stretch')  # input: axial stretch

    # calculate the invariants
    I1 = keras.layers.Lambda(lambda x: x ** 2 + 2.0 / x, name='I1')(stretch)
    I2 = keras.layers.Lambda(lambda x: 2.0 * x + 1 / x ** 2, name='I2')(stretch)

    I1_ref = keras.layers.Lambda(lambda x: (x - 3.0), name='I1_ref')(I1)
    I2_ref = keras.layers.Lambda(lambda x: (x - 3.0), name='I2_ref')(I2)

    # calculate the terms of the initial stored energy function
    I1_out = SingleInvNet6(I1_ref, 0)
    I2_out = SingleInvNet6(I2_ref, 0)

    ALL_I_out = [I1_out, I2_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out)

    # initial stored energy function
    psi = keras.layers.Dense(1, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.NonNeg(),
                             kernel_regularizer=keras.regularizers.l2(l2),
                             use_bias=False, activation=None, name='wx2')(ALL_I_out)

    # taking derivatives to calculate the stress
    dPsidI1 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([psi, I1])
    dPsidI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([psi, I2])

    # normal initial Cauchy stress in axial direction
    sig0 = keras.layers.Lambda(function=Stress_calc_UT, name='init_Stress')([dPsidI1, dPsidI2, stretch, I1])

    dt = keras.layers.Input(shape=(None, 1), name='time_step')  # input: time step

    merge = keras.layers.Concatenate(name='rnn_input')([dt, sig0])
    # the relaxation function RNN
    rnnOutput = keras.layers.RNN(cell, return_sequences=True, name='relax_function')(merge)

    # calculating the total Cauchy stress
    out = keras.layers.TimeDistributed(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(rp),
                                                          kernel_initializer=keras.initializers.RandomUniform(
                                                              minval=0.01, maxval=1.), use_bias=False,
                                                          kernel_constraint=SumToOne(), name='stress'))(rnnOutput)

    model = keras.models.Model(inputs=[stretch, dt], outputs=out)

    return model


# %%%% vanilla RNN model %%%%

# build the vanilla RNN model with 1 LSTM layer of 8 hidden units
def build_rnn():
    visible1 = layers.Input(shape=(None, 2))
    hidden1 = layers.LSTM(8, return_sequences=True)(visible1)
    outputLayer = layers.TimeDistributed(layers.Dense(1))(hidden1)

    model = keras.Model(inputs=visible1, outputs=outputLayer)

    return model