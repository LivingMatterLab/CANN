#!/usr/bin/env python
# coding: utf-8



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

from models_brain import*
from models_brain import SinglePrincipalStretchLayer, OuterLayer  # Import explicitly for type checking
from plotting_brain import*
from util import *

from sklearn.metrics import r2_score


#%% Uts
filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()





#%% Init
train = False
epochs = 10000
batch_size = 64
p = 0.5

# Define lists to iterate over
# model_types_all = ['model_si_mi', 'model_si', 'model_si_ps']
model_types_all = ['model_si_mi', 'model_si_ps']


# lp_reg_values = [0.0, 1.0]
lp_reg_values = [1.0]

# Select loading modes and foam types of interest
# #modelFit_mode_all = ['SS', 'C', 'T', 'TC', "TC_and_SS"]
modelFit_mode_all = ["TC_and_SS"] 
foam_types_all = ['leap', 'turbo']

#Import excel file
file_name = 'input/FoamData.xlsx'
dfs = pd.read_excel(file_name, sheet_name='Sheet1')

# Initialize list to store all results for comprehensive table
results_summary = []

#%%  Training and validation loop 
count = 1
total_combinations = len(model_types_all) * len(lp_reg_values) * len(foam_types_all) * len(modelFit_mode_all)

for model_type in model_types_all:
    # Choose which terms to include in the model architecture based on model_type
    include_invariant_terms = "si" in model_type
    include_mixed_terms = "mi" in model_type
    include_principal_stretch_terms = "ps" in model_type
    
    for lp_reg in lp_reg_values:
        model_name = model_type + '_lp_' + str(lp_reg)
        
        # Create the path to save the results for this model_type and lp_reg combination
        path2saveResults_0 = 'Results/' + filename + '/' + model_name
        makeDIR(path2saveResults_0)
        Model_summary = path2saveResults_0 + '/Model_summary.txt'
        
        # prepare R2 array for this model_type/lp_reg combination
        R2_all = np.zeros([len(foam_types_all), len(modelFit_mode_all) + 2])
        
        # Store results for this model_type/lp_reg combination
        model_results = {
            'model_type': model_type,
            'lp_reg': lp_reg,
            'leap': {'nonzero_terms': None, 'R2_tension': None, 'R2_compression': None, 'R2_shear': None},
            'turbo': {'nonzero_terms': None, 'R2_tension': None, 'R2_compression': None, 'R2_shear': None}
        }
        
        for id1, foam_type in enumerate(foam_types_all):
            
            R2_curr_foam_type = []
            for id2, modelFit_mode in enumerate(modelFit_mode_all):
        
                print(40*'=')
                print("Comp {:d} / {:d}".format(count, total_combinations))
                print(40*'=')
                print("Model type: ", model_type, "| Lp_reg: ", lp_reg)
                print("Foam type: ", foam_type ,"| Fitting Mode: ", modelFit_mode)
                print(40*'=')
                count += 1
                
                # Create new folder to save the results
                path2saveResults = os.path.join(path2saveResults_0, foam_type, modelFit_mode)
                path2saveResults_check = os.path.join(path2saveResults, 'Checkpoints')
                makeDIR(path2saveResults)
                makeDIR(path2saveResults_check)
                
                # Get the stress-strain data from the excel file
                lam_ut, P_ut, P_ut_std, gamma_ss, P_ss, P_ss_std = getStressStrain(foam_type, dfs)
                
                # Create the strain energy (Psi) model
                Psi_model_unreg, terms, _, _, _ = StrainEnergyCANN(0.0, p, include_invariant_terms, include_mixed_terms, include_principal_stretch_terms)
                model_unreg = modelArchitecture(Psi_model_unreg)
                Psi_model, terms, mixed_layer, single_principal_stretch_layer, single_principal_stretch_layer2 = StrainEnergyCANN(lp_reg, p, include_invariant_terms, include_mixed_terms, include_principal_stretch_terms)
                model = modelArchitecture(Psi_model)

                
                with open(Model_summary, 'w') as fh:
                    # Pass the file handle in as a lambda function to make it callable
                    Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
                    
                #%%  Model training


                
                model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model, lam_ut, P_ut, P_ss, gamma_ss)
                    
                Save_path = path2saveResults + '/model.keras'
                Save_weights = path2saveResults + '/weights.weights.h5'
                path_checkpoint = path2saveResults_check + '/best_weights.weights.h5'
                if train:

                    ##
                    
                    model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, batch_size, model_unreg)

                    # model_given.load_weights(path_checkpoint,  skip_mismatch=False)
                    tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
                    Psi_model.save_weights(Save_weights, overwrite=True)
                    
                    
                    # Plot loss function
                    loss_history = history.history['loss']
                    fig, axe = plt.subplots(figsize=[6, 5])  # inches
                    plotLoss(axe, loss_history)
                    plt.savefig(path2saveResults + '/Plot_loss_' + foam_type + '_' + modelFit_mode + '.pdf')
                    plt.close()
                    
                else:
                    # Psi_model = tf.keras.models.load_model(Save_path)
                    try:
                        Psi_model.load_weights(Save_weights, skip_mismatch=False)
                    except ValueError as e:
                        print(f"Warning: Could not load weights for {foam_type} - {modelFit_mode}: {e}")
                        print("Continuing with randomly initialized weights...")
                
                # Display the strain energy expression
                display_strain_energy_expression(Psi_model, terms, mixed_layer, single_principal_stretch_layer, single_principal_stretch_layer2)
                
                # Count nonzero terms after training/loading weights (only once per foam type)
                if modelFit_mode == modelFit_mode_all[0]:  # Only count once per foam type (first mode)
                    try:
                        nonzero_count = count_nonzero_terms(Psi_model, p=p)
                        model_results[foam_type]['nonzero_terms'] = nonzero_count
                        print(f"Nonzero terms for {foam_type}: {nonzero_count}")
                    except Exception as e:
                        print(f"Warning: Could not count nonzero terms for {foam_type}: {e}")
                        model_results[foam_type]['nonzero_terms'] = 0
                    
                
                # PI-CANN  get model respons
                lam_ut_model = np.linspace(np.amin(lam_ut), np.amax(lam_ut), 200)
                gamma_model = np.linspace(np.amin(gamma_ss), np.amax(gamma_ss), 200)



                # PI-CANN get model response at data points
                Stress_predict_axial, Stress_predict_trans, _, Stress_predict_shear, _ = model.predict([lam_ut, lam_ut * 0 + 0.8, gamma_ss])
                # Extract individual term contributions at data points
                stress_contributions, shear_stress_contributions, term_names = extract_term_contributions(model, lam_ut, gamma_ss)

                #%% Plotting - Create both old combined plot and new separate plots
                
                # Original combined plot (keep the old functionality)
                fig = plt.figure(figsize=(10, 3))
                spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
                ax1 = fig.add_subplot(spec[0, 0])

                # R2, R2_c, R2_t = plotTenCom(ax1, lam_ut, P_ut, Stress_predict_axial, Region)
                plotTrans(ax1, lam_ut, P_ut * 0.0, Stress_predict_trans, foam_type)
                # R2_ss = plotShear(ax2, gamma_ss, P_ss, Stress_predict_shear, Region)
                fig.tight_layout()        

                plt.savefig(path2saveResults + '/Plot_Trans_' + foam_type + '_' + modelFit_mode + '.pdf')
                plt.close()
                
                # Combined plot with all contributions (tension, compression, shear) in horizontal subplots
                # Also includes transverse plot centered underneath
                fig_all, R2_c, R2_t, R2_ss = plotAllContributions(
                    lam_ut, P_ut, Stress_predict_axial,
                    gamma_ss, P_ss, Stress_predict_shear,
                    stress_contributions, shear_stress_contributions, term_names, foam_type,
                    Stress_predict_trans=Stress_predict_trans
                )
                plt.savefig(path2saveResults + '/Plot_All_Contributions_' + foam_type + '_' + modelFit_mode + '.pdf', 
                           bbox_inches='tight', dpi=300)
                plt.close()


                if modelFit_mode == 'T':
                    R2_cur = [R2_t]
                elif modelFit_mode == "C":
                    R2_cur = [R2_c]
                elif modelFit_mode == "TC":
                    R2 = r2_score_own(P_ut, Stress_predict_axial)
                    R2_cur = [R2] 
                elif modelFit_mode == "SS":
                    R2_cur = [R2_ss]    
                elif modelFit_mode == "TC_and_SS":
                    R2_cur = [R2_ss, R2_c, R2_t]
                    # Store R2 values for this foam type
                    model_results[foam_type]['R2_shear'] = R2_ss
                    model_results[foam_type]['R2_compression'] = R2_c
                    model_results[foam_type]['R2_tension'] = R2_t
                 
                R2_curr_foam_type.append(R2_cur)
                print("R2: ", R2_cur) 


                

            R2_all[id1, :] = np.array(flatten(R2_curr_foam_type))
        
        
        
        #%% Summarizing results for this model_type/lp_reg combination
        modelFit_mode_all_table = ['SS', 'C', 'T']  # Match the actual R2 outputs
        R2_mean = np.expand_dims(np.mean(R2_all, axis=0), axis=0)
        R2_sd = np.expand_dims(np.std(R2_all, axis=0), axis=0)
        R2_all_mean = np.concatenate((R2_all, R2_mean, R2_sd), axis=0)
        R2_df = pd.DataFrame(R2_all_mean, index=foam_types_all + ['mean', 'SD'], columns=modelFit_mode_all_table)
        R2_df.to_latex(path2saveResults_0 + '/R2_table.tex', index=True)
        R2_df.to_csv(path2saveResults_0 + '/R2_table.csv', index=True)
        
        # Add this model's results to the summary
        results_summary.append(model_results)

#%% Create comprehensive summary table
print("\n" + 80*"=")
print("Creating comprehensive summary table...")
print(80*"=")

# Create DataFrame from results_summary
table_data = []
for result in results_summary:
    row = {
        'Model type': result['model_type'],
        'Regularization parameter': result['lp_reg'],
        '# nonzero terms (leap)': result['leap']['nonzero_terms'] if result['leap']['nonzero_terms'] is not None else 'N/A',
        'R2 tension (leap)': result['leap']['R2_tension'] if result['leap']['R2_tension'] is not None else 'N/A',
        'R2 compression (leap)': result['leap']['R2_compression'] if result['leap']['R2_compression'] is not None else 'N/A',
        'R2 shear (leap)': result['leap']['R2_shear'] if result['leap']['R2_shear'] is not None else 'N/A',
        '# nonzero terms (turbo)': result['turbo']['nonzero_terms'] if result['turbo']['nonzero_terms'] is not None else 'N/A',
        'R2 tension (turbo)': result['turbo']['R2_tension'] if result['turbo']['R2_tension'] is not None else 'N/A',
        'R2 compression (turbo)': result['turbo']['R2_compression'] if result['turbo']['R2_compression'] is not None else 'N/A',
        'R2 shear (turbo)': result['turbo']['R2_shear'] if result['turbo']['R2_shear'] is not None else 'N/A'
    }
    table_data.append(row)

# Create DataFrame
summary_df = pd.DataFrame(table_data)

# Format numeric columns for better LaTeX output
numeric_cols = ['R2 tension (leap)', 'R2 compression (leap)', 'R2 shear (leap)',
                'R2 tension (turbo)', 'R2 compression (turbo)', 'R2 shear (turbo)']
for col in numeric_cols:
    summary_df[col] = summary_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

# Save to LaTeX and CSV
summary_latex_path = 'Results/' + filename + '/comprehensive_summary_table.tex'
summary_csv_path = 'Results/' + filename + '/comprehensive_summary_table.csv'

# Format LaTeX table with better formatting
summary_df.to_latex(summary_latex_path, index=False, float_format="%.3f", escape=False)
summary_df.to_csv(summary_csv_path, index=False)

print(f"Comprehensive summary table saved to:")
print(f"  LaTeX: {summary_latex_path}")
print(f"  CSV: {summary_csv_path}")
print("\nSummary table:")
print(summary_df.to_string())

