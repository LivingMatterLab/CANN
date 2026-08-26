

Publications:

1. Automated model discovery for human cardiac tissue: Discovering the best model and parameters

doi: 10.1101/2024.02.27.582427

2. Discovering dispersion: How robust is automated model discovery for human myocardial tissue?

doi: 10.1007/s10237-025-02005-x

3. Fiber dispersion in the right ventricle: A comparison of constitutive neural network predictions with experimental data

doi: 10.1016/j.jbiomech.2026.113532

Data organisation

\input\CANNsHEARTdata_shear05.xlsx: experimental data from biaxial and shear tests

\input\CANNsHEARTdata_shear05_synthetic.xlsx: synthetic data from biaxial and shear tests

\input\CANN_comparison_data_positive_negative_strain.xlsx: raw uniaxial tension-compression and simple shear tests experimental data for 11 right ventricular myocardial tissue samples (positive and negative strains data)

\input\CANN_comparison_data_positive_strain.xlsx: uniaxial tension and simple shear tests experimental data for 11 right ventricular myocardial tissue samples (positive strains data only)

aligned_fibers:

HeartCANN_discovered.ipynb: Jupyter Notebook used to train and evaluate the orthotropic, perfectly incompressible, feed forward constitutive neural network, reads and processes data and plots results

HeartCANN_Guan.ipynb: Jupyter Notebook used to train and evaluate the Guan model, reads and processes data and plots results

HeartCANN_HO.ipynb: Jupyter Notebook used to train and evaluate the Holzapfel-Ogden model, reads and processes data and plots results

HeartCANN_Guan.ipynb: Jupyter Notebook used to train and evaluate the generalized Holzapfel model, reads and processes data and plots results

\results_shear05\HeartCANN: all results used in the publication 1


dispersed_fibers:

HeartCANN_discovered_dispersion.ipynb: Jupyter Notebook used to train and evaluate the orthotropic, perfectly incompressible, feed forward constitutive neural network with fixed fiber, sheet and normal dispersions, reads and processes data and plots results

HeartCANN_discovered_dispersion_train_kappa.ipynb: Jupyter Notebook used to train and evaluate the orthotropic, perfectly incompressible, feed forward constitutive neural network with trainable fiber, sheet and normal dispersion, reads and processes data and plots results


dispersed_fibers_with_mean_fiber_angle:

cann_comparison_with_experimental_data: Python module used to train and evaluate the orthotropic, perfectly incompressible, feed forward constitutive neural network with dispersed fibers and a mean fiber angle. Module allows training for different scenarios based on the configuration, namely:
a. trainable fiber dispersion + fixed mean fiber angle
b. trainable fiber dispersion + trainable mean fiber angle