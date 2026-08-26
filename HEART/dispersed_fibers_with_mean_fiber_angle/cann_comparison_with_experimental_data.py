"""Custom Constitutive Neural Network Using TensorFlow.

Research Article Title:
Discovering Fiber Dispersion in Myocardial Tissue:
A Comparison of Constitutive Neural Network  Predictions with Experimental Data
"""
#  Python Version == 3.14.7

import argparse
import copy
import json
from pathlib import Path

import keras  # version == 3.13.2
import numpy as np  # version == 2.4.2
import pandas as pd  # version == 3.0.0
import tensorflow as tf  # version == 2.20.0
from sklearn.metrics import r2_score  # version == 1.8.0

# ==========================================
# Constants for the CANN model
# ==========================================


DEFAULT_CONFIG = {
    "run": {
        "input_file": "input_data_raw.xlsx",
        "sheet_name": "TSL09",
        "output_dir": "results",
    },
    "training": {
        "learning_rate": 0.001,
        "epochs": 10000,
        "batch_size": 32,
        "validation_split": 0.2,
        "loss_type": "normalized",
        "loss_normalization": "rms",
        "early_stop_patience": 1000,
        "early_stop_min_delta": 0.001,
    },
    "model_spec": {
        "l1_penalty": 0.01,
        "l2_penalty": 0.0,
        "is_theta_trainable": False,
        "is_kappa_f_trainable": True,
        "is_kappa_s_trainable": False,
        "is_kappa_n_trainable": False,
    },
    "initialization": {
        "theta": 1.3,
        "kappa_f": 0.0,
        "kappa_s": 0.0,
        "kappa_n": 0.0,
    },
}

PROTOCOLS = [
    "shear_fs",
    "shear_fn",
    "shear_sf",
    "shear_sn",
    "shear_nf",
    "shear_ns",
    "uniaxial_f",
    "uniaxial_s",
    "uniaxial_n",
]

# ==========================================
# Parser for command-line arguments
# ==========================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Constitutive Artificial Neural Network (CANN) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Run arguments ---
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_CONFIG["run"]["input_file"],
        help="Path to the input data file (Excel format)",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default=DEFAULT_CONFIG["run"]["sheet_name"],
        help="Sheet name in the Excel file containing the data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_CONFIG["run"]["output_dir"],
        help="Directory to save output results",
    )

    # --- Training arguments ---
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG["training"]["learning_rate"],
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["training"]["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["training"]["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_CONFIG["training"]["validation_split"],
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["basic", "normalized"],
        default=DEFAULT_CONFIG["training"]["loss_type"],
        help="Type of loss function to use",
    )
    parser.add_argument(
        "--loss-normalization",
        type=str,
        choices=["rms", "max"],
        default=DEFAULT_CONFIG["training"]["loss_normalization"],
        help="Normalization method for normalized loss",
    )

    # --- Model Specification ---
    parser.add_argument(
        "--l1-penalty",
        type=float,
        default=DEFAULT_CONFIG["model_spec"]["l1_penalty"],
        help="L1 regularization penalty",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=DEFAULT_CONFIG["model_spec"]["l2_penalty"],
        help="L2 regularization penalty",
    )
    parser.add_argument(
        "--train-theta",
        default=DEFAULT_CONFIG["model_spec"]["is_theta_trainable"],
        action="store_true",
        help="Whether theta parameter is trainable",
    )
    parser.add_argument(
        "--train-kappa-f",
        default=DEFAULT_CONFIG["model_spec"]["is_kappa_f_trainable"],
        action="store_true",
        help="Whether kappa_f parameter is trainable",
    )
    parser.add_argument(
        "--train-kappa-s",
        default=DEFAULT_CONFIG["model_spec"]["is_kappa_s_trainable"],
        action="store_true",
        help="Whether kappa_s parameter is trainable",
    )
    parser.add_argument(
        "--train-kappa-n",
        default=DEFAULT_CONFIG["model_spec"]["is_kappa_n_trainable"],
        action="store_true",
        help="Whether kappa_n parameter is trainable",
    )

    # --- Initial parameters ---
    parser.add_argument(
        "--theta",
        type=float,
        default=DEFAULT_CONFIG["initialization"]["theta"],
        help="Initial value for theta parameter",
    )
    parser.add_argument(
        "--kappa-f",
        type=float,
        default=DEFAULT_CONFIG["initialization"]["kappa_f"],
        help="Initial value for kappa_f parameter",
    )
    parser.add_argument(
        "--kappa-s",
        type=float,
        default=DEFAULT_CONFIG["initialization"]["kappa_s"],
        help="Initial value for kappa_s parameter",
    )
    parser.add_argument(
        "--kappa-n",
        type=float,
        default=DEFAULT_CONFIG["initialization"]["kappa_n"],
        help="Initial value for kappa_n parameter",
    )

    # --- Verbosity ---
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during training",
    )

    return parser.parse_args()


def get_config_from_args(args):
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Run
    config["run"]["input_file"] = args.input_file
    config["run"]["sheet_name"] = args.sheet_name

    # Training
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["validation_split"] = args.validation_split
    config["training"]["loss_type"] = args.loss_type
    config["training"]["loss_normalization"] = args.loss_normalization

    # Model spec
    config["model_spec"]["l1_penalty"] = args.l1_penalty
    config["model_spec"]["l2_penalty"] = args.l2_penalty
    config["model_spec"]["is_theta_trainable"] = args.train_theta
    config["model_spec"]["is_kappa_f_trainable"] = args.train_kappa_f
    config["model_spec"]["is_kappa_s_trainable"] = args.train_kappa_s
    config["model_spec"]["is_kappa_n_trainable"] = args.train_kappa_n

    # Initialization
    config["initialization"]["theta"] = args.theta
    config["initialization"]["kappa_f"] = args.kappa_f
    config["initialization"]["kappa_s"] = args.kappa_s
    config["initialization"]["kappa_n"] = args.kappa_n

    # Output
    kappa_f_str = f"kf_{args.kappa_f:.2f}".replace(".", "_")
    config["run"]["output_dir"] = f"{args.output_dir}/{args.sheet_name}/{kappa_f_str}"

    return config


# ==========================================
# Data Loading and Preprocessing
# ==========================================


def load_data(input_file, sheet_name, verbose=False):
    if verbose:
        print(f"Loading data from {input_file}, sheet: {sheet_name}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Extract protocols from column names (format: strain_XX, stress_XX)
    strain_cols = [col for col in df.columns if col.startswith("strain_")]
    stress_cols = [col for col in df.columns if col.startswith("stress_")]

    # Prepare input data
    strains_and_gammas = df[strain_cols].values
    stretches = (
        strains_and_gammas.copy()
    )
    is_uniaxial = np.array(["uniaxial" in p for p in PROTOCOLS])
    # Add 1.0 only to uniaxial columns to convert strain to stretch (λ = 1 + ε)
    stretches[:, is_uniaxial] += 1.0
    stresses = df[stress_cols].values  # Shape: (n_samples, 9)

    # Convert to float32
    stretches = stretches.astype("float32")
    stresses = stresses.astype("float32")

    if verbose:
        print(
            f"Loaded {stretches.shape[0]} samples with {stretches.shape[1]} protocols."
        )

    strain_dict = {
        protocol: stretches[:, i : i + 1] for i, protocol in enumerate(PROTOCOLS)
    }
    stress_dict = {
        protocol: stresses[:, i : i + 1] for i, protocol in enumerate(PROTOCOLS)
    }

    return strain_dict, stress_dict


# ==========================================
# Model Definitions and Neural Network Architecture
# ==========================================

# --- Invariant Layer ---


class InvariantLayer(keras.layers.Layer):
    def __init__(self, protocol_type, **kwargs):
        super().__init__(**kwargs)
        self.protocol_type = protocol_type

    def get_config(self):
        config = super().get_config()
        config.update({"protocol_type": self.protocol_type})
        return config

    @tf.function
    def smooth_fiber_activation(self, I4, sharpness=100.0):
        """
        Smooth tension-only fiber activation.

        Physics: Fibers only bear load in tension (I4 > 1).
        Math: Approximates max(1, I4) with continuous derivatives.
        """
        log2 = tf.math.log(2.0)
        return 1.0 + (tf.nn.softplus(sharpness * (I4 - 1.0)) - log2) / sharpness

    @tf.function  # Add graph compilation
    def get_C(self, lam):
        """Builds Right Cauchy-Green Tensor C based on protocol and stretch λ."""

        one = tf.ones_like(lam)
        zero = tf.zeros_like(lam)

        # Pre-compute common terms
        lam_sq = lam * lam
        one_lam_sq = one + lam_sq
        inv_l = tf.math.reciprocal(lam)

        if "uniaxial_f" in self.protocol_type:
            rows = [
                tf.concat([lam_sq, zero, zero], axis=1),
                tf.concat([zero, inv_l, zero], axis=1),
                tf.concat([zero, zero, inv_l], axis=1),
            ]

        elif "uniaxial_s" in self.protocol_type:
            rows = [
                tf.concat([inv_l, zero, zero], axis=1),
                tf.concat([zero, lam_sq, zero], axis=1),
                tf.concat([zero, zero, inv_l], axis=1),
            ]

        elif "uniaxial_n" in self.protocol_type:
            rows = [
                tf.concat([inv_l, zero, zero], axis=1),
                tf.concat([zero, inv_l, zero], axis=1),
                tf.concat([zero, zero, lam_sq], axis=1),
            ]

        elif "shear_fs" in self.protocol_type:
            rows = [
                tf.concat([one_lam_sq, lam, zero], axis=1),
                tf.concat([lam, one, zero], axis=1),
                tf.concat([zero, zero, one], axis=1),
            ]

        elif "shear_fn" in self.protocol_type:
            rows = [
                tf.concat([one_lam_sq, zero, lam], axis=1),
                tf.concat([zero, one, zero], axis=1),
                tf.concat([lam, zero, one], axis=1),
            ]

        elif "shear_sf" in self.protocol_type:
            rows = [
                tf.concat([one, lam, zero], axis=1),
                tf.concat([lam, one_lam_sq, zero], axis=1),
                tf.concat([zero, zero, one], axis=1),
            ]

        elif "shear_sn" in self.protocol_type:
            rows = [
                tf.concat([one, zero, zero], axis=1),
                tf.concat([zero, one_lam_sq, lam], axis=1),
                tf.concat([zero, lam, one], axis=1),
            ]

        elif "shear_nf" in self.protocol_type:
            rows = [
                tf.concat([one, zero, lam], axis=1),
                tf.concat([zero, one, zero], axis=1),
                tf.concat([lam, zero, one_lam_sq], axis=1),
            ]

        elif "shear_ns" in self.protocol_type:
            rows = [
                tf.concat([one, zero, zero], axis=1),
                tf.concat([zero, one, lam], axis=1),
                tf.concat([zero, lam, one_lam_sq], axis=1),
            ]

        else:
            raise ValueError(f"Unknown protocol type: {self.protocol_type}")

        return tf.stack(rows, axis=1)

    def call(self, lam, theta):
        C = self.get_C(lam)

        # Pre-compute trignometric functions
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)
        zeros = tf.zeros_like(theta)
        ones = tf.ones_like(theta)

        # Basis vectors (pre-computed)
        f = tf.concat([cos_theta, sin_theta, zeros], axis=1)
        s = tf.concat([-sin_theta, cos_theta, zeros], axis=1)
        n = tf.concat([zeros, zeros, ones], axis=1)

        # Invariants
        I1 = tf.linalg.trace(C)[:, tf.newaxis]
        C_sq = tf.matmul(C, C)
        trace_C_sq = tf.linalg.trace(C_sq)[:, tf.newaxis]
        I2 = 0.5 * (tf.square(I1) - trace_C_sq)

        # Fiber Products
        Cf, Cs, Cn = (
            tf.linalg.matvec(C, f),
            tf.linalg.matvec(C, s),
            tf.linalg.matvec(C, n),
        )
        I4f_raw = tf.reduce_sum(f * Cf, axis=1, keepdims=True)
        I4s_raw = tf.reduce_sum(s * Cs, axis=1, keepdims=True)
        I4n_raw = tf.reduce_sum(n * Cn, axis=1, keepdims=True)

        # Smooth tension-only activation (replaces tf.maximum(1, I4))
        I4s = self.smooth_fiber_activation(I4s_raw)
        I4n = self.smooth_fiber_activation(I4n_raw)
        I4f = self.smooth_fiber_activation(I4f_raw)

        # Shear invariants
        I8fs = tf.abs(tf.reduce_sum(f * Cs, axis=1, keepdims=True))
        I8fn = tf.abs(tf.reduce_sum(f * Cn, axis=1, keepdims=True))
        I8sn = tf.abs(tf.reduce_sum(s * Cn, axis=1, keepdims=True))

        return I1, I2, I4f, I4s, I4n, I8fs, I8fn, I8sn


# --- Dispersion Layer ---


class DispersionConstraint(keras.constraints.Constraint):
    def __call__(self, w):
        # Hard clipping to [0, 1/3]
        return tf.clip_by_value(w, 0.0, tf.constant(1.0 / 3.0, dtype=tf.float32))


class DispersionLayer(keras.layers.Layer):
    def __init__(
        self,
        trainable_flags=[True, False, False],
        initial_values=[0.0, 0.0, 0.0],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trainable_flags = trainable_flags
        self.initial_values = initial_values

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "trainable_flags": self.trainable_flags,
                "initial_values": self.initial_values,
            }
        )
        return config

    def build(self, input_shape):
        dispersion_constraint = DispersionConstraint()

        self.kappa_f = self.add_weight(
            name="kappa_f",
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_values[0]),
            constraint=dispersion_constraint,
            trainable=self.trainable_flags[0],
        )
        self.kappa_s = self.add_weight(
            name="kappa_s",
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_values[1]),
            constraint=dispersion_constraint,
            trainable=self.trainable_flags[1],
        )
        self.kappa_n = self.add_weight(
            name="kappa_n",
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_values[2]),
            constraint=dispersion_constraint,
            trainable=self.trainable_flags[2],
        )

    def call(self, I1, I4f, I4s, I4n):
        I4f_star = self.kappa_f * I1 + (1.0 - 3.0 * self.kappa_f) * I4f
        I4s_star = self.kappa_s * I1 + (1.0 - 3.0 * self.kappa_s) * I4s
        I4n_star = self.kappa_n * I1 + (1.0 - 3.0 * self.kappa_n) * I4n
        return I4f_star, I4s_star, I4n_star


# --- Single Invariant Basis Expansion Layer ---


class SingleInvariantBasisLayer(keras.layers.Layer):
    def __init__(
        self, index, use_linear_basis=False, l1_penalty=0.01, l2_penalty=0.00, **kwargs
    ):
        super().__init__(**kwargs)
        self.index = index
        self.use_linear_basis = use_linear_basis

        # Initializers
        self.linear_initializer = keras.initializers.GlorotNormal()
        self.exponential_initializer = keras.initializers.RandomUniform(
            minval=0.0, maxval=0.1
        )

        self.regularizer = keras.regularizers.L1L2(l1=l1_penalty, l2=l2_penalty)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "index": self.index,
                "use_linear_basis": self.use_linear_basis,
                "l1_penalty": self.regularizer.l1,
                "l2_penalty": self.regularizer.l2,
            }
        )
        return config

    @tf.function
    def exponential_activation(self, x):
        return tf.math.exp(x) - 1.0

    def build(self, input_shape):
        if self.use_linear_basis:
            # Linear basis on I
            self.w_linear_I = keras.layers.Dense(
                1,
                kernel_initializer=self.linear_initializer,  # type:ignore
                kernel_constraint=keras.constraints.NonNeg(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=None,
                name=f"w_linear_I_{self.index}",
            )

            # Exponential basis on I
            self.w_exponential_I = keras.layers.Dense(
                1,
                kernel_initializer=self.exponential_initializer,  # type: ignore[arg-type]
                kernel_constraint=keras.constraints.NonNeg(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=self.exponential_activation,
                name=f"w_exponential_I_{self.index}",
            )

        # Linear basis on I²
        self.w_linear_I2 = keras.layers.Dense(
            1,
            kernel_initializer=self.linear_initializer,  # type: ignore[arg-type]
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=self.regularizer,
            use_bias=False,
            activation=None,
            name=f"w_linear_I2_{self.index}",
        )

        # Exponential basis on I²
        self.w_exponential_I2 = keras.layers.Dense(
            1,
            kernel_initializer=self.exponential_initializer,  # type: ignore[arg-type]
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=self.regularizer,
            use_bias=False,
            activation=self.exponential_activation,
            name=f"w_exponential_I2_{self.index}",
        )

        super().build(input_shape)

    def call(self, Invariant):
        outputs = []

        if self.use_linear_basis:
            # Linear and exponential basis expansion on I
            out_linear_I = self.w_linear_I(Invariant)
            out_exponential_I = self.w_exponential_I(Invariant)
            outputs.extend([out_linear_I, out_exponential_I])

        # Linear and exponential basis expansion on I²
        I_squared = tf.math.square(Invariant)
        out_linear_I2 = self.w_linear_I2(I_squared)
        out_exponential_I2 = self.w_exponential_I2(I_squared)
        outputs.extend([out_linear_I2, out_exponential_I2])

        return tf.concat(outputs, axis=1)


# --- Shared Strain Energy Model ---


class SharedStrainEnergyModel(keras.Model):
    """Custom strain energy network with basis function expansion."""

    # Class-level constants
    INVARIANT_NAMES = ["I1", "I2", "I4f", "I4s", "I4n", "I8fs", "I8fn", "I8sn"]
    REF_SHIFTS = [3.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    LINEAR_BASIS_INVARIANTS = ["I1", "I2"]

    def __init__(self, l1_penalty=0.001, l2_penalty=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.ref_shifts = tf.constant(self.REF_SHIFTS, dtype=tf.float32)

        self.invariants_basis_nets = [
            SingleInvariantBasisLayer(
                index=i,
                use_linear_basis=(name in self.LINEAR_BASIS_INVARIANTS),
                l1_penalty=l1_penalty,
                l2_penalty=l2_penalty,
                name=f"{name}_basis",
            )
            for i, name in enumerate(self.INVARIANT_NAMES)
        ]

        self.basis_combination = keras.layers.Dense(
            1,
            use_bias=False,
            kernel_constraint=keras.constraints.NonNeg(),
            kernel_regularizer=keras.regularizers.L1L2(l1=l1_penalty, l2=l2_penalty),
            name="Psi",
        )

    def call(self, invariants):
        shifted = invariants - self.ref_shifts
        shifted_invs = tf.split(shifted, len(self.INVARIANT_NAMES), axis=1)

        all_basis = [
            net(inv) for inv, net in zip(shifted_invs, self.invariants_basis_nets)
        ]

        return self.basis_combination(tf.concat(all_basis, axis=1))

    def get_config(self):
        config = super().get_config()
        config.update({"l1_penalty": self.l1_penalty, "l2_penalty": self.l2_penalty})
        return config


# --- Custom Constitutive Neural Network Model ---


class ThetaConstraint(keras.constraints.Constraint):
    """Constraint theta to be between 0 and 180 degrees (0 to pi radians)."""

    def __call__(self, w):
        return tf.clip_by_value(w, 0.0, tf.constant(np.pi, dtype=tf.float32))


class ConstitutiveNeuralNetworkModel(keras.Model):
    def __init__(
        self,
        l1_penalty=0.01,
        l2_penalty=0.0,
        is_theta_trainable=False,
        is_kappa_f_trainable=True,
        is_kappa_s_trainable=False,
        is_kappa_n_trainable=False,
        theta_init_deg=0.0,
        kappa_f_init=0.0,
        kappa_s_init=0.0,
        kappa_n_init=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.is_theta_trainable = is_theta_trainable
        self.theta_init_deg = theta_init_deg

        self.kappa_trainable_flags = [
            is_kappa_f_trainable,
            is_kappa_s_trainable,
            is_kappa_n_trainable,
        ]
        self.kappa_init_values = [
            kappa_f_init,
            kappa_s_init,
            kappa_n_init,
        ]

        self.invariant_layers = {
            protocol: InvariantLayer(
                protocol_type=protocol, name=f"invariant_layer_{protocol}"
            )
            for protocol in PROTOCOLS
        }

        self.theta = np.radians(self.theta_init_deg)

        self.dispersion_layer = DispersionLayer(
            trainable_flags=self.kappa_trainable_flags,
            initial_values=self.kappa_init_values,
            name="dispersion_layer",
        )

        self.strain_energy_model = SharedStrainEnergyModel(
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            name="shared_strain_energy_model",
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "l1_penalty": self.l1_penalty,
                "l2_penalty": self.l2_penalty,
                "is_theta_trainable": self.is_theta_trainable,
                "is_kappa_f_trainable": self.kappa_trainable_flags[0],
                "is_kappa_s_trainable": self.kappa_trainable_flags[1],
                "is_kappa_n_trainable": self.kappa_trainable_flags[2],
                "theta_init_deg": self.theta_init_deg,
                "kappa_f_init": self.kappa_init_values[0],
                "kappa_s_init": self.kappa_init_values[1],
                "kappa_n_init": self.kappa_init_values[2],
            }
        )
        return config

    def build(self, input_shape):
        self.theta = self.add_weight(
            name="theta",
            shape=(1,),
            initializer=tf.constant_initializer(self.theta),
            constraint=ThetaConstraint(),
            trainable=self.is_theta_trainable,
        )
        super().build(input_shape)

    def call(self, stretches_dict):
        # stretches_dict: {protocol: (batch, 1)} for each protocol
        # Get batch size from the first protocol's input
        first_protocol = PROTOCOLS[0]
        batch_size = tf.shape(stretches_dict[first_protocol])[0]
        theta_expanded = tf.tile(self.theta, [batch_size])[:, tf.newaxis]

        all_stresses = {}

        for protocol in PROTOCOLS:
            lambda_i = stretches_dict[protocol]

            with tf.GradientTape() as tape:
                tape.watch(lambda_i)
                # Calculate unmixed invariants
                I1, I2, I4f, I4s, I4n, I8fs, I8fn, I8sn = self.invariant_layers[
                    protocol
                ](lambda_i, theta_expanded)
                I4f_star, I4s_star, I4n_star = self.dispersion_layer(I1, I4f, I4s, I4n)

                # Concatenate all invariants
                invariants = tf.concat(
                    [I1, I2, I4f_star, I4s_star, I4n_star, I8fs, I8fn, I8sn], axis=1
                )

                # Calculate Energy
                Psi = self.strain_energy_model(invariants)

            # Stress P = dW / d_lambda
            stress = tape.gradient(Psi, lambda_i)
            stress = tf.identity(stress, name=protocol)
            all_stresses[protocol] = stress

        return all_stresses


# ==========================================
# Model Training
# ==========================================


def _shuffle_data(stretches, stresses, seed=None):
    """
    Shuffle the training data while keeping stretches and stresses aligned.
    """
    # Get number of samples from first protocol
    first_protocol = PROTOCOLS[0]
    n_samples = len(stretches[first_protocol])

    # Create shuffled indices
    indices = np.arange(n_samples)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    # Shuffle all protocols with same indices to maintain alignment
    shuffled_stretches = {
        protocol: stretches[protocol][indices] for protocol in PROTOCOLS
    }
    shuffled_stresses = {
        protocol: stresses[protocol][indices] for protocol in PROTOCOLS
    }

    return shuffled_stretches, shuffled_stresses


def _compute_loss_weights(stresses, config, verbose=False):
    """
    Compute loss weights for each protocol based on the configured loss type.
    """
    loss_weights_dict = {}

    if config["training"]["loss_type"] == "basic":
        if verbose:
            print("Using basic MSE loss.")
        loss_weights_dict = {protocol: 1.0 for protocol in PROTOCOLS}
    elif config["training"]["loss_type"] == "normalized":
        if verbose:
            print(
                f"Using normalized MSE loss with {config['training']['loss_normalization']} normalization."
            )
        # Compute normalization factors for each protocol from the stress dictionary
        for protocol in PROTOCOLS:
            stress_vals = stresses[protocol]
            if config["training"]["loss_normalization"] == "rms":
                factor = np.sqrt(np.mean(np.square(stress_vals)))
            elif config["training"]["loss_normalization"] == "max":
                factor = np.max(np.abs(stress_vals))
            else:
                raise ValueError("Invalid loss normalization method.")

            factor = max(factor, 1e-6)  # Avoid division by zero
            weight = 1.0 / (factor**2)
            loss_weights_dict[protocol] = weight
            if verbose:
                print(f"   {protocol:15s}: Scale={factor:.2f}, Weight={weight:.2e}")

    return loss_weights_dict


def train_model(model, stretches, stresses, config, verbose=False):
    # Shuffle data before training (improves convergence and R2 scores)
    stretches, stresses = _shuffle_data(stretches, stresses)
    if verbose:
        print("Data shuffled for training.")

    if verbose:
        print("Compiling model...")

    optimizer = keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
    loss_weights_dict = _compute_loss_weights(stresses, config, verbose)

    model.compile(
        optimizer=optimizer,
        loss={p: "mse" for p in PROTOCOLS},
        loss_weights=loss_weights_dict,
    )

    # --- Callbacks ---
    early_stop_patience = config["training"].get("early_stop_patience", 1000)
    early_stop_min_delta = config["training"].get("early_stop_min_delta", 0.001)
    monitor = "val_loss" if config["training"]["validation_split"] > 0 else "loss"

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        restore_best_weights=True,
        verbose=2 if verbose else 0,
    )

    # Reduce learning rate on plateau - use 1/4 of early stop patience
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=early_stop_patience // 4,
        min_lr=1e-6,
        verbose=2 if verbose else 0,
    )
    callbacks = [early_stopping, reduce_lr]

    if verbose:
        print("Starting training...")

    # Train model
    train_kwargs = {
        "x": stretches,
        "y": stresses,
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
        "callbacks": callbacks,
        "verbose": 1 if verbose else 0,
    }
    if config["training"]["validation_split"] > 0:
        if verbose:
            print(
                "Using validation split of "
                f"{config['training']['validation_split']:.2f} for training."
            )
        train_kwargs["validation_split"] = config["training"]["validation_split"]

    history = model.fit(**train_kwargs)

    return model, history


# ==========================================
# Save and Load Model Utilities
# ==========================================


def _numpy_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_run_config(config):
    """
    Save the complete configuration (including results) to a JSON file.
    """
    output_path = Path(config["run"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "run_config.json"
    with open(config_path, "w") as json_file:
        json.dump(config, json_file, indent=4, default=_numpy_converter)
    print(f"Run configuration saved to {config_path}")


def save_trained_weights(model, config):
    """
    Save trained model weights. Architecture should already be saved
    before training (in run_experiment) so it stays clean of loss_weights.
    Updates config in-place with saved paths.

    Args:
        model: The trained ConstitutiveNeuralNetworkModel instance
        config: Configuration dictionary (uses output_dir and sheet_name, updates in-place)
    """
    model_name = "cann_" + config["run"]["sheet_name"]

    output_path = Path(config["run"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = output_path / f"{model_name}.weights.h5"
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")

    # Update config in-place with saved paths
    config["run"]["saved_weights_path"] = str(weights_path)


def save_history(history, config):

    output_path = Path(config["run"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Save history
    history_path = output_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print(f"Model history saved to {history_path}")

    # Update config in-place with saved paths
    config["run"]["history_path"] = str(history_path)


def test_model(model, stretches, stresses, verbose=False):
    if verbose:
        print("Evaluating model on training data...")

    stress_preds = model.predict(stretches, verbose=1 if verbose else 0)

    # --- R2 Score Calculation ---
    r2_scores = {}
    for protocol in PROTOCOLS:
        y_true = stresses[protocol]
        y_pred = stress_preds[protocol]

        r2_scores[protocol] = r2_score(y_true, y_pred)

    average_r2 = np.mean(list(r2_scores.values()))
    r2_scores["average"] = average_r2

    if verbose:
        print("R2 Scores by Protocol:")
        for protocol, r2 in r2_scores.items():
            print(f"   {protocol:15s}: R2 = {r2:.4f}")
        print(f"Average R2 Score: {average_r2:.4f}")

    return stress_preds, r2_scores


# ==========================================
# Model and Result Analysis
# ==========================================


def analyze_model(model, config):
    """
    Analyze the trained model and update config in-place with results.

    Args:
        model: The trained ConstitutiveNeuralNetworkModel instance
        config: Configuration dictionary to update in-place
    """
    # Extract trained parameters (convert theta to degrees)
    trained_theta_rad = model.theta.numpy().item()
    trained_kappa_f = model.dispersion_layer.kappa_f.numpy().item()

    config["trained_parameters"] = {
        "theta": np.degrees(trained_theta_rad),
        "kappa_f": trained_kappa_f,
        "kappa_s": model.dispersion_layer.kappa_s.numpy().item(),
        "kappa_n": model.dispersion_layer.kappa_n.numpy().item(),
    }

    def _get_active_invariants(model, threshold=1e-4):
        """Analyze which invariants are active based on effective weight contributions."""
        invariant_names = SharedStrainEnergyModel.INVARIANT_NAMES
        active_invariants = []

        # Get combination layer weights
        w_combination = model.strain_energy_model.basis_combination.get_weights()[
            0
        ]  # Shape: (N, 1)

        idx = 0
        for i, (name, inv_net) in enumerate(
            zip(invariant_names, model.strain_energy_model.invariants_basis_nets)
        ):
            total_effective_weight = 0.0

            # Determine number of outputs for this invariant
            n_outputs = 4 if inv_net.use_linear_basis else 2

            # Get combination weights for this invariant's outputs
            comb_weights = w_combination[idx : idx + n_outputs, 0]

            comb_idx = 0
            if inv_net.use_linear_basis:
                # Linear pathway on I
                w_linear_I = float(inv_net.w_linear_I.get_weights()[0][0, 0])
                w_exp_I = float(inv_net.w_exponential_I.get_weights()[0][0, 0])
                total_effective_weight += abs(w_linear_I * comb_weights[comb_idx])
                comb_idx += 1
                total_effective_weight += abs(w_exp_I * comb_weights[comb_idx])
                comb_idx += 1

            # Squared pathway weights (all invariants have these)
            w_linear_I2 = float(inv_net.w_linear_I2.get_weights()[0][0, 0])
            w_exp_I2 = float(inv_net.w_exponential_I2.get_weights()[0][0, 0])
            total_effective_weight += abs(w_linear_I2 * comb_weights[comb_idx])
            comb_idx += 1
            total_effective_weight += abs(w_exp_I2 * comb_weights[comb_idx])

            idx += n_outputs

            if total_effective_weight > threshold:
                active_invariants.append(name)

        return active_invariants

    active_invs = _get_active_invariants(model)
    config["active_invariants"] = active_invs


def build_model_from_config(config):
    return ConstitutiveNeuralNetworkModel(
        l1_penalty=config["model_spec"]["l1_penalty"],
        l2_penalty=config["model_spec"]["l2_penalty"],
        is_theta_trainable=config["model_spec"]["is_theta_trainable"],
        is_kappa_f_trainable=config["model_spec"]["is_kappa_f_trainable"],
        is_kappa_s_trainable=config["model_spec"]["is_kappa_s_trainable"],
        is_kappa_n_trainable=config["model_spec"]["is_kappa_n_trainable"],
        theta_init_deg=config["initialization"]["theta"],
        kappa_f_init=config["initialization"]["kappa_f"],
        kappa_s_init=config["initialization"]["kappa_s"],
        kappa_n_init=config["initialization"]["kappa_n"],
    )


def build_model_from_spec(model_spec):
    model = ConstitutiveNeuralNetworkModel(
        l1_penalty=model_spec["l1_penalty"],
        l2_penalty=model_spec["l2_penalty"],
        is_theta_trainable=model_spec["is_theta_trainable"],
        is_kappa_f_trainable=model_spec["is_kappa_f_trainable"],
        is_kappa_s_trainable=model_spec["is_kappa_s_trainable"],
        is_kappa_n_trainable=model_spec["is_kappa_n_trainable"],
        theta_init_deg=0.0,
        kappa_f_init=0.0,
        kappa_s_init=0.0,
        kappa_n_init=0.0,
    )

    dummy_input = {
        protocol: tf.zeros((1, 1), dtype=tf.float32) for protocol in PROTOCOLS
    }
    _ = model(dummy_input)

    return model


def run_experiment(config, verbose=False):
    """
    Run a complete training experiment with the given configuration.

    Args:
        config: Configuration dictionary with all training parameters
        verbose: Whether to print verbose output

    Returns:
        Completed config dictionary with results (r2_scores, trained_parameters, etc.)
    """
    # Load and preprocess data
    stretches, stresses = load_data(
        config["run"]["input_file"], config["run"]["sheet_name"], verbose=verbose
    )

    # Initialize model
    print("Building Constitutive Neural Network Model...")
    model = build_model_from_config(config)

    # Train model (compiles with data-specific loss_weights inside)
    model, history = train_model(model, stretches, stresses, config, verbose=verbose)

    # Save only the trained weights
    save_trained_weights(model, config)
    save_history(history, config)

    # Evaluate model
    stress_preds, r2_scores = test_model(model, stretches, stresses, verbose=verbose)

    # Add R2 scores to config
    config["r2_scores"] = {
        protocol: float(score) for protocol, score in r2_scores.items()
    }

    analyze_model(model, config)
    print("Model analysis complete. Config updated.")

    # Save complete config to output directory
    save_run_config(config)

    return config


def main():
    args = parse_args()
    config = get_config_from_args(args)

    if args.verbose:
        print("Starting training run with configuration:")
        print(json.dumps(config, indent=4))

    # Run the experiment
    config = run_experiment(config, verbose=args.verbose)

    if args.verbose:
        print("\nFinal configuration with results:")
        print(json.dumps(config, indent=4, default=_numpy_converter))

    return config


if __name__ == "__main__":
    main()
