from src.frontend import *
from src.plotting import *
from src.models import *
from src.cont_mech import *
from src.util_functions import *
from matplotlib import gridspec
import json
import keras.backend
import os

filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()

# Define relevant flags and training parameters
p = 0.5 # Set to 1 to L1 regularization instead of l0.5 regularization
alphas = [0, 0.001, 0.01, 0.1, 1] # List of regularization parameters (must start with 0)
modelFit_mode_all = ['0123456789', '01234', '56789', '012356789']  # List of which data is used for training (all data, 0-90 orientation only, 45-135 orientation only)
Region = 'mesh' # Material type (specifies which spreadsheet of raw data to load)
cann_names = ["cann_I4ws", "cann_I4w_theta", "cann_I4ws_noiso", "cann_I4w_theta_noiso"] # List of names of folders to save results in
models = [ortho_cann_2ff, ortho_cann_3ff, ortho_cann_2ff_noiso, ortho_cann_3ff_noiso] # List of models to train with
models_2term = [ortho_cann_2ff_2term, ortho_cann_3ff_2term, ortho_cann_2ff_2term, ortho_cann_3ff_2term] # List of models that can take as a parameter which parameters to restrict
is_I4betas = [False, True, False, True]

n_terms = 14

numIter = 1 # Number of times training is repeated (relevant for generating uniqueness graphs)

if __name__ == "__main__":

    # Iterate through brain regions / tissue types
    for id1, cann_name in enumerate(cann_names):

        model = models[id1]
        model_2term = models_2term[id1]

        weight_hist_all_mode = []
        # Iterate through training modes (i.e. tension and shear only)
        for id2, modelFit_mode in enumerate(modelFit_mode_all):


            dfs = load_df(Region)

            # Fit 2 term model
            if modelFit_mode == '0123456789':
                for term1 in range(n_terms):
                    for term2 in range(term1 + 1):
                        cann_name_temp = cann_name + f"/{term2}_{term1}"
                        terms = [term1] if term1 == term2 else [term1, term2]
                        model_temp = model_2term(terms)
                        train_canns(dfs, model_temp, [0], p, Region, modelFit_mode, numIter, cann_name_temp)

                paths2saveResults = [os.path.join(get_path(Region, modelFit_mode), cann_name + f"/{term2}_{term1}", str(0))
                                     for term1 in range(n_terms) for term2 in range(term1 + 1)]
                plot_l0_map(paths2saveResults, n_terms, dfs, Region, is_I4beta=is_I4betas[id1])

            # Repeat training numIter times to assess uniqueness
            train_canns(dfs, model, alphas, p, Region, modelFit_mode, numIter, cann_name)
            for num in range(numIter):
                path2saveResults_curr = os.path.join(get_path(Region, modelFit_mode), cann_name, str(num))
                if modelFit_mode == "0123456789":
                    display_equation(path2saveResults_curr, dfs, Region, modelFit_mode)
                plot_training(dfs, model, modelFit_mode, p, path2saveResults_curr, is_I4beta=is_I4betas[id1])
                keras.backend.clear_session() # Clear keras session to save memory

        if len(modelFit_mode_all) == 4:
            plot_r2_bargraph([os.path.join(get_path(Region, modelFit_mode), cann_name, str(0)) for modelFit_mode in modelFit_mode_all])

    plot_arch_comp_graph([os.path.join(get_path(Region, '0123456789'), cann_name, str(0)) for cann_name in cann_names[0:2]])
    # After training on all modes, create all mode r squared vs number of terms graph
    plot_raw_data()
    export_graphs()