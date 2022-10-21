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
from sklearn.metrics import r2_score

from plottingRubber import*
from Models import*
from ContinuumMechHelper import*
#%% Uts
filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()

def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def flatten(l):
    return [item for sublist in l for item in sublist]
    
def r2_score_own(Truth, Prediction):
    try:
        R2 = r2_score(Truth,Prediction)
        R2 = max(R2,0.0)
    except ValueError:
        R2 = np.nan
    return R2


#%% Functions
   
def traindata(modelFit_mode):
    
    if modelFit_mode == 'UT':
        model_given = model_UT
        input_train = lam_UT
        output_train = P_UT
        sample_weights = np.array([1.0]*input_train.shape[0])
        
    elif modelFit_mode == "PS":
        model_given = model_PS
        input_train = lam_PS
        output_train = P_PS
        sample_weights = np.array([1.0]*input_train.shape[0])
        
    elif modelFit_mode == "ET":
        model_given = model_ET
        input_train = lam_ET
        output_train = P_ET
        sample_weights = np.array([1.0]*input_train.shape[0])
        
    elif modelFit_mode == "All":
        model_given = model
        
        
        lam_inter_PS = np.linspace(1.0,np.amax(lam_PS), lam_UT.shape[0])
        P_inter_PS = np.interp(lam_inter_PS, lam_PS, P_PS)
        
        lam_inter_ET = np.linspace(1.0,np.amax(lam_ET), lam_UT.shape[0])
        P_inter_ET = np.interp(lam_inter_ET, lam_ET, P_ET)
        
        input_train = [lam_UT, lam_inter_PS, lam_inter_ET]
        output_train = [P_UT, P_inter_PS, P_inter_ET]

        sample_weights_UT = np.array([1.0]*lam_UT.shape[0])
        sample_weights_PS = np.array([1.0]*lam_inter_PS.shape[0])
        sample_weights_ET = np.array([1.0]*lam_inter_ET.shape[0])
        sample_weights = [sample_weights_UT, sample_weights_PS, sample_weights_ET]
    
    return model_given, input_train, output_train, sample_weights



def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights):
    
    mse_loss = keras.losses.MeanSquaredError()
    metrics  =[keras.metrics.MeanSquaredError()]
    opti1    = tf.optimizers.Adam(learning_rate=0.001)
    
    model_given.compile(loss=mse_loss,
                  optimizer=opti1,
                  metrics=metrics)
    
    
    # es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=3)
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=1000, restore_best_weights=True)
    term = keras.callbacks.TerminateOnNaN()
    # csvLogger = tf.keras.callbacks.CSVLogger(Model_train_logger, separator=',', append=True)


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
                        callbacks=[es_callback, modelckpt_callback, term],
                        shuffle = True,
                        verbose = 1, 
                        # sample_weight = sample_weights,
                        )
    
    return history, model_given



if __name__ == "__main__": 
    
    #%% Data
    with open('input/Treloar.pkl', 'rb') as f:
        F, P11, extra, _, _, _, _, _, _ = pickle.load(f)
    
    lam_UT =  F[:15,0,0]
    lam_ET =  F[15:30,0,0]
    lam_PS =  F[30:,0,0]
    
    P_UT = P11[:15]*10
    P_ET = P11[15:30]*10
    P_PS = P11[30:]*10
    
    
    #%% Init
    
    train = False
    epochs = 12000 
    batch_size = 8 
    model_type = 'all'
    
    
    weight_flag = True
    weight_plot_Map = True
    
    
    path2saveResults_0 = 'Results/'+filename+'/'+model_type
    makeDIR(path2saveResults_0)
    Model_summary = path2saveResults_0 + '/Model_summary.txt'
    
            
    modelFit_mode_all = ['UT', 'PS', 'ET', 'All']
    # modelFit_mode_all = ['ET']
    
    R2_all = np.zeros([len(modelFit_mode_all) + 2])
    R2_all_fit = []
    
    #%% Model Training and evaluation
    
    count = 1
    for id2, modelFit_mode in enumerate(modelFit_mode_all):
        
        print(40*'=')
        print("Comp {:d} / {:d}".format(count, len(modelFit_mode_all)))
        print(40*'=')
        print("Fitting Mode: ", modelFit_mode)
        print(40*'=')
        count += 1
        
        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
    
    
        
        #%% CANN
        # Get strain energy model
        Psi_model, terms = StrainEnergyCANN()
        # Get Stress models
        model_UT, model_PS, model_ET, model = ContinuumMechanicsFramework(Psi_model)
        

        #%%  Model training
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode)
    
        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        path_checkpoint = path2saveResults_check + '/best_weights' 
        if train:
            prediction = np.nan
            counter_it = 1
            while np.isnan(prediction):
                print("Fitting Interation: ", counter_it)
                
                history, model_given = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights)
                prediction = model_UT.predict([1.0])[0][0]
                counter_it +=1
                if counter_it ==5:
                    break
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
            Psi_model.save_weights(Save_weights, overwrite=True)
            
            # Plot loss function
            loss_history = history.history['loss']
            fig, axe = plt.subplots(figsize=[6, 5])  # inches
            plotLoss(axe, loss_history)
            plt.savefig(path2saveResults+'/Plot_loss_'+modelFit_mode+'.pdf')
            plt.close()
            
        else:
            # Load save weights
            Psi_model.load_weights(Save_weights, by_name=False, skip_mismatch=False)
        
        
        # Save model architecture
        with open(Model_summary,'w') as fh:
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
        
        # PI-CANN  get model response
        Stress_predict_UT = model_UT.predict(lam_UT)
        Stress_predict_PS = model_PS.predict(lam_PS)
        Stress_predict_ET = model_ET.predict(lam_ET)
        R2_UT = r2_score_own(P_UT, Stress_predict_UT)
        R2_PS = r2_score_own(P_PS, Stress_predict_PS)
        R2_ET = r2_score_own(P_ET, Stress_predict_ET)
        
              
        #%% Display weights
        weight_matrix = np.empty([int(terms), 2])
        weight_matrix[:] = np.nan
        second_layer = Psi_model.get_layer('wx2').get_weights()[0].flatten()
        j=0
        if weight_flag:
            for i in range(1,int(terms +1)):
                try:
                    value = Psi_model.get_layer('w'+str(i)+'1').get_weights()[0][0][0]
                    weight_matrix[i-1,1] =  second_layer[j]
                    j+=1
                except ValueError:
                    value = np.nan
                
                weight_matrix[i-1,0] = value
                
        model_weights_0 = Psi_model.get_weights()
        print("weight_matrix")  
        print(weight_matrix)
    
    
        #%% Plotting
        
        # Plot stress-strain
        fig, ax2 = plt.subplots(figsize=(600/72,400/72))
        PlotTre(ax2, lam_UT, lam_ET, lam_PS, P_UT, P_PS, P_ET,
                model_UT, model_PS, model_ET, modelFit_mode)
    
        fig.tight_layout()        
        plt.savefig(path2saveResults+'/Plot_PI-CANN_'+modelFit_mode+'.pdf')
        plt.close()
       
        # Plot color maps
        if weight_plot_Map:
            fig2 = plt.figure(figsize=(600/72,800/72))
            spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig2)
            ax1 = fig2.add_subplot(spec2[0,0])
            ax2 = fig2.add_subplot(spec2[1,0])
            ax3 = fig2.add_subplot(spec2[2,0])
            
            plotMap(ax1, lam_UT, P_UT, model_UT, terms, Psi_model, model_weights_0, 'UT')
            plotMap(ax2, lam_PS, P_PS, model_PS, terms, Psi_model, model_weights_0, 'PS')
            plotMap(ax3, lam_ET, P_ET, model_ET, terms, Psi_model, model_weights_0, 'ET')
            
            fig2.tight_layout()
            plt.savefig(path2saveResults+'/Plot_PI-CANN_MAP_'+modelFit_mode+'.pdf')
            plt.close()
        
    
        #%% Savin meta results
        Config = {modelFit_mode:modelFit_mode, "R2_UT":R2_UT, "R2_PS": R2_PS, "R2_ET": R2_ET, "weigths": weight_matrix.tolist()}
        json.dump(Config, open(path2saveResults+"/Config_file.txt",'w'))
        
