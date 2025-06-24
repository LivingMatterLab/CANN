from tabulate import tabulate

from src.CANN.util_functions import disp_mat, makeDIR, get_hgo_weights
from src.gcann import disp_equation_weights_gcann, train_gcanns, generate_synth_data, ortho_cann_3ff_gcann, \
    get_model_weights
from src.plotting import plot_gcann
from src.utils import *
import csv

if __name__ == "__main__":


    ## Load data from inputs folder
    # stretches, stresses, n_samples = load_data()
    # base_path = "../Results/"

    ## Uncomment these three lines to generate synthetic data using a 2 term HGO model
    model_weights = get_hgo_weights()
    stretches, stresses, n_samples = generate_synth_data(model_weights)
    base_path = "../Results_synth_hgo"

    ## Uncomment these three lines to generate synthetic data using the discovered weights from the correlated model
    # model_weights, scale_factor = get_model_weights()
    # stretches, stresses, n_samples = generate_synth_data(model_weights, scale_factor=scale_factor)
    # base_path = "../Results_synth_correlated/"


    # Set to true to train model, false to load previously trained model
    should_train = True

    # Create results folder
    makeDIR(base_path)


    # String with list of which tests to use for training (see legend below)
    ### 0-90 Orientation ###
    # strip-w:  0 = w stress, 1 = s stress
    # off-w:    2 = w stress, 3 = s stress
    # equibiax: 4 = w stress, 5 = s stress
    # off-s:    6 = w stress, 7 = s stress
    # strip-s:  8 = w stress, 9 = s stress
    ### +/- 45 Orientation ###
    # strip-x:  a = x stress, e = y stress
    # off-x:    b = x stress, d = y stress
    # equibiax: c = x stress = y stress (by symmetry)

    # modelFit_mode = "02356789bcde"  # 80/20 train test split
    modelFit_mode = "0123456789abcde" # Uncomment this to train using all data
    # modelFit_mode = "0123456789" # Uncomment this to train using only data from 0/90 orientation
    # modelFit_mode = "abcde" # Uncomment this to train using only data from +/- 45 orientation


    # Switch which of these is uncommented to determine which type of model is trained
    table = [["Model Type", "Alpha", "Nonzero Terms", "Train NLL", "Test NLL"]]
    for model_type in ["correlated"]:#["unregularized", "independent", "correlated"]:
        # Change this to set the regularization parameter (if used)
        alphas = [0] if model_type == "unregularized" else [0.1]
        for alpha in alphas:
            print(model_type)


            # Train GCANN based on stretch and stress data provided
            model_given, lam_ut_all, P_ut_all = train_gcanns(stretches, stresses, n_samples, modelFit_mode=modelFit_mode, should_train=should_train, model_type=model_type, alpha_in=alpha, base_path=base_path) # Test if works the same as before

            # Print correlation matrix (identity matrix if using independent model)
            model_weights = model_given.get_weights()
            print(len(model_weights))
            nonzero_weights = [i for i in range(len(model_weights) // 2) if model_weights[2 * i + 1] > 0.0]
            cov_matrix = model_weights[-1] @ model_weights[-1].T
            cov_matrix_subset = cov_matrix[nonzero_weights, :][:, nonzero_weights]
            std_devs = np.sqrt(np.diag(cov_matrix_subset))
            corr_matrix = cov_matrix_subset / (std_devs[:, np.newaxis] @ std_devs[np.newaxis, :])
            disp_mat(corr_matrix)
            print(corr_matrix)

            # Print model equation and weights (with variances)
            disp_equation_weights_gcann(model_given.get_weights(), lam_ut_all, P_ut_all, "0123456789")

            # Create all plots
            model_id = get_model_id(model_type, alpha)
            plot_gcann(stretches, stresses, model_given, lam_ut_all, False, model_id, modelFit_mode, n_samples, blank=True, base_path=base_path) # Blank plot of just data
            plot_gcann(stretches, stresses, model_given, lam_ut_all, True, model_id, modelFit_mode, n_samples, base_path=base_path) # Plot of terms contributing to mean
            train_nll, test_nll, train_nll2 = plot_gcann(stretches, stresses, model_given, lam_ut_all, False, model_id, modelFit_mode, n_samples, plot_dist=True, base_path=base_path) # Plot of mean and standard deviation (plot data distribution vs predicted distribution)
            plot_gcann(stretches, stresses, model_given, lam_ut_all, False, model_id, modelFit_mode, n_samples, plot_dist=False, base_path=base_path) # Plot of mean and standard deviation (plot data vs predicted distribution)

            table += [[model_type, alpha, len(nonzero_weights), f"{train_nll:.3f}", f"{test_nll:.3f}"]]


    # Write table to file
    print(tabulate(table))
    filename = f"{base_path}/{modelFit_mode}/loss_table.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(table)

