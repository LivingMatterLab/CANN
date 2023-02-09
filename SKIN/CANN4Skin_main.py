#!/usr/bin/env python
# coding: utf-8


# Essentials
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
# ML
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from sklearn.metrics import r2_score
# Others
import pickle
import json
import statistics

# Own imports
from PlotSkin_p import*
from ModelsSkin_p import*

#%% Uts
filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()

def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)

def flatten(l):
    return [item for sublist in l for item in sublist]
    
def r2_score_own(Truth, Prediction):
    R2 = r2_score(Truth,Prediction)
    return max(R2,0.0)


#%% Functions

def Compile_and_fit(modelFit_mode, model_given, input_train, output_train, input_val, output_val, epochs, path_checkpoint):
    
    mse_loss = keras.losses.MeanSquaredError()
    metrics  =[keras.metrics.MeanSquaredError()]
    opti1    = tf.optimizers.Adam(learning_rate=0.001)
    
    model_given.compile(loss=mse_loss,
                  optimizer=opti1,
                  metrics=metrics)

        
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=2000, restore_best_weights=True)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="loss",
    filepath=path_checkpoint,
    verbose=0,
    save_weights_only=True,
    save_best_only=True,
    )
    
    history = model_given.fit(input_train,
                        output_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.0,
                        callbacks=[es_callback, modelckpt_callback],
                        shuffle = True,
                        verbose = 0, 
                        # sample_weight = sample_weights
                        )
    
    return model_given, history





#%% Read and process data

if __name__ == "__main__": 
    
    df_MechData = pd.read_csv('data/NODE_porcine_skin_data_1.csv')
    
    
    lambda_x =  df_MechData['lambda_x'].to_numpy()
    sigma_xx =  df_MechData['sigma_xx [MPa]'].to_numpy()
    
    lambda_y =  df_MechData['lambda_y'].to_numpy()
    sigma_yy =  df_MechData['sigma_yy [MPa]'].to_numpy()
    
    break_list = [72, 73+75, 73+76+80, 73+76+81+100, 73+76+81+101+71]
    
    # c_lis = ['b','g','r','k','m']
    
    st=73+76+81+ 101
    end=73+76+81 +101+71
    
    
    
    st=0
    all_lam_x = []
    all_lam_y = []
    all_Sigma_xx = []
    all_Sigma_yy = []
    for i, end in enumerate(break_list):
        all_lam_x.append(lambda_x[st:end])
        all_lam_y.append(lambda_y[st:end])
        all_Sigma_xx.append(sigma_xx[st:end])
        all_Sigma_yy.append(sigma_yy[st:end])
        st = end + 1
    
    
    fig = plt.figure(figsize=(800/72,600/72))
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0:2,0:2])
    ax2 = fig.add_subplot(spec[2,0:2])
    ax3 = fig.add_subplot(spec[2,2])
           
    
    
    
    
    #%% Init
    modelFit_mode_all = ['all', '1', '2', '3', "4", "5"]
    
    # Training settings
    epochs = 8000
    batch_size = 32 
    Alpha_init = np.pi/4
    
    # Other settings
    model_type = 'Model_p'
    weight_flag = True
    weight_plot_Map = True
    PlotALL_Solo = False
    PlotALL_combine = False
    train = False
    setAlpha = True
    
    # set result paths
    path2saveResults_0 = 'Results/'+filename+'/'+model_type
    makeDIR(path2saveResults_0)
    Model_summary = path2saveResults_0 + '/Model_summary.txt'
    path2saveResults = path2saveResults_0
    path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
    makeDIR(path2saveResults)
    makeDIR(path2saveResults_check)
    
    
    #%% Training
    
    R2_all = np.zeros([10, len(modelFit_mode_all)])
    R2_pick = np.zeros([2, len(modelFit_mode_all)])
    
    count = 1
    for id1, modelFit_mode in enumerate(modelFit_mode_all):
        
        print(40*'=')
        # print('Comp '+str(count)+'/'+str())
        print("Comp {:d} / {:d}".format(count, len(modelFit_mode_all)))
        print(40*'=')
        print("Fitting Mode: ", modelFit_mode)
        print(40*'=')
        count += 1
            
        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
    
        #%% PI-CANN, define energy and constitutive model
        
       
        Psi_model, terms = StrainEnergy_i5()
        model_BT = modelArchitecture_I5(Psi_model, setAlpha, Alpha_init)
    
        
        with open(Model_summary,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
            
        
        #%% preprocess training and validation data
        if modelFit_mode == 'all':
            input_train = [lambda_x, lambda_y]
            output_train = [sigma_xx, sigma_yy]
            input_val = [np.nan, np.nan]
            output_val = [np.nan, np.nan]
        else:
    
            input_train = [all_lam_x[id1-1], all_lam_y[id1-1]]
            output_train = [all_Sigma_xx[id1-1], all_Sigma_yy[id1-1]]
            input_val = [np.nan, np.nan]
            output_val = [np.nan, np.nan]
        
        
        
        #%%  Model training 
        model_given = model_BT
        
        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        Stored_weights = path2saveResults_0 + '/weights'
        path_checkpoint = path2saveResults_check + '/best_weights' 
        
        if train:
    
            if modelFit_mode =='all':
                pass
            else:
                model_given.load_weights(os.path.join(path2saveResults_0, 'all') + '/weights', by_name=False, skip_mismatch=False)
                
            
            model_given, history = Compile_and_fit(modelFit_mode, model_given, input_train, output_train,
                                                   input_val, output_val,
                                                   epochs, path_checkpoint)
            
            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
            tf.keras.models.save_model(model_given, Save_path, overwrite=True)
            model_given.save_weights(Save_weights, overwrite=True)
    
            loss_history = history.history['loss']
            fig, axe = plt.subplots(figsize=[6, 5])  # inches
            
            # Plot loss function
            plotLoss(axe, loss_history)
    
            plt.savefig(path2saveResults+'/Plot_loss_.pdf')
            plt.close()
            
        else:
            model_given.load_weights(Save_weights, by_name=False, skip_mismatch=False)
        
        
        # PI-CANN  get model response
        Stress_predicted = []
        for j in range(len(all_lam_x)):
            Stress_pre = model_BT.predict([all_lam_x[j], all_lam_y[j]])
            Stress_predicted.append(Stress_pre)
        
        
        
        #%% Plotting
        
        fig2 = plt.figure(figsize=(1200/72,800/72))
        spec = gridspec.GridSpec(ncols=3, nrows=4, figure=fig)
        ax1 = fig2.add_subplot(spec[0:2,0:2])
        ax2 = fig2.add_subplot(spec[2:,0:2])
        ax3 = fig2.add_subplot(spec[0,2])
        
        R2x_all, R2y_all = PlotCycles(id1, ax1, ax2, ax3, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted)
        
        
        fig2.tight_layout()
        plt.savefig(path2saveResults+'/Plot_PI-CANN_skin.pdf')
        plt.close()
    
        
    
        if modelFit_mode != 'all':
            fig2 = plt.figure(figsize=(1200/72,400/72))
            spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
            ax1 = fig2.add_subplot(spec[0:2,0:2])
            ax3 = fig2.add_subplot(spec[0,2])
            R2x_fit, R2y_fit = PlotSolo_FungBiax(id1-1, ax1, 0, ax3, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy, Stress_predicted)
            
            fig2.tight_layout()
            plt.savefig(path2saveResults+'/Plot_PI-CANN_skin_solo.pdf')
            
            
    
    
        print('='*30)
        if setAlpha:
            alpha = model_BT.get_weights()[0]
        else:
            alpha = Alpha_init
        print('alpha [deg]:', np.rad2deg(alpha))
        print('='*30)
        Alpha_init = alpha
        setAlpha = False
        
        #%% Show weights
        weight_matrix = np.empty([int(terms), 2])
        weight_matrix[:] = np.nan
        second_layer = Psi_model.get_weights()[terms].flatten()
        j=0
        if weight_flag:
            for i in range(int(terms)):
                try:
                    value = Psi_model.get_weights()[i][0][0]
                    weight_matrix[i,1] =  second_layer[j]
                    j+=1
                except ValueError:
                    value = np.nan
                
                weight_matrix[i,0] = value
            
        print("weight_matrix")  
        print(weight_matrix)
        
        
        #%% plotting_ color map
        model_weights_0 = Psi_model.get_weights()
        
    
        if weight_plot_Map:
            
            for kk in range(len(all_lam_x)):
                fig2 = plt.figure(figsize=(600/72,600/72))
                spec2 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig2)
                fig_ax1 = fig2.add_subplot(spec2[0,0])
                ax2 = fig2.add_subplot(spec2[1,0])
                
                
                plotMapTenCom16(fig2, fig_ax1, Psi_model, model_weights_0, model_BT, terms,
                                all_lam_x[kk], all_lam_y[kk], all_lam_x[kk], all_Sigma_xx[kk], Stress_predicted[kk][0], 'x',kk+1)
                plotMapTenCom16(fig2, ax2, Psi_model, model_weights_0, model_BT,
                                terms, all_lam_x[kk], all_lam_y[kk], all_lam_y[kk], all_Sigma_yy[kk], Stress_predicted[kk][1], 'y',kk+1)
                
                fig2.tight_layout()
                plt.savefig(path2saveResults+'/Plot_PI-CANN_MAP_'+modelFit_mode+'_cy_'+str(kk+1)+'.pdf')
                plt.close()
                
    
        #%% Storing data
        Config = {modelFit_mode:modelFit_mode, "R2_x1":R2x_all[0], "R2_x2": R2x_all[1], "R2_x3": R2x_all[2], "R2_x4": R2x_all[3], "R2_x5": R2x_all[4],
                  "R2_y1":R2y_all[0], "R2_y2": R2y_all[1], "R2_y3": R2y_all[2], "R2_y4": R2y_all[3], "R2_y5": R2y_all[4], "alpha_ang": (alpha).item(),
                  "weigths": weight_matrix.tolist()}
        json.dump(Config, open(path2saveResults+"/Config_file.txt",'w'))
    
        R2_all[:,id1] = np.array(R2x_all + R2y_all)
        
        if modelFit_mode=='all':
            R2x_p = statistics.mean(R2x_all)
            R2y_p = statistics.mean(R2y_all)
        else:
            R2x_p = R2x_all[id1-1]
            R2y_p = R2y_all[id1-1]
        
        R2_pick[0,id1] = R2x_p
        R2_pick[1,id1] = R2y_p
    
        print(40*'=')
        print("Rx: ", R2x_p, "|  Ry:", R2y_p)
        print(40*'=')
    
    
    
    
    #%% Plotting summmaries
    
    if PlotALL_Solo:
        fig2 = plt.figure(figsize=(2400/72,1200/72))
        spec = gridspec.GridSpec(ncols=5, nrows=3, figure=fig)
        
        AllPlots_solo(fig2, path2saveResults_0, spec, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy)
        
        fig2.tight_layout()
        plt.savefig(path2saveResults_0+'/All_plots_solo.pdf')
        plt.close()
    
    
    if PlotALL_combine:
        fig2 = plt.figure(figsize=(2400/72,1200/72))
        spec = gridspec.GridSpec(ncols=5, nrows=3, figure=fig)
        
        AllPlots_combine(fig2, path2saveResults_0, spec, all_lam_x, all_lam_y, all_Sigma_xx, all_Sigma_yy)
        
        plt.savefig(path2saveResults_0+'/All_plots_combined.pdf')
        plt.close()
        
        
    
    #%% plotting bars
    
    f, axes = plt.subplots(figsize=(800/72,800/(3*72)))
    barPlot(path2saveResults_0, model_type, R2_all, R2_pick, f, axes )
    f.savefig(path2saveResults_0+'/R2_results.pdf')
    
    
    
