#!/usr/bin/env python
# coding: utf-8

# In[20]:


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
from plotting_brain import*

from sklearn.metrics import r2_score



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


def getStressStrain(Region):

    if Region =='leap':
        P_ut = dfs.iloc[3:,1].astype(np.float64)
        lam_ut = dfs.iloc[3:,0].astype(np.float64)
        
        P_ss = dfs.iloc[3:,3].astype(np.float64).values
        gamma_ss = dfs.iloc[3:,2].astype(np.float64).values
    elif Region =='turbo':
        P_ut = dfs.iloc[3:,5].astype(np.float64)
        lam_ut = dfs.iloc[3:,4].astype(np.float64)

        P_ss = dfs.iloc[3:,7].astype(np.float64).values
        gamma_ss = dfs.iloc[3:,6].astype(np.float64).values


    return P_ut, lam_ut, P_ss, gamma_ss

   
def traindata(modelFit_mode):
    model_given = model
    if modelFit_mode == 'T':
        stretch_in = lam_ut[16:]
        shear_in = np.array([0.0] * stretch_in.shape[0])
        input_train = [stretch_in, shear_in]
        stress_axial = P_ut[16:]
        stress_trans = np.array([0.0] * stress_axial.shape[0])
        stress_shear = np.array([0.0] * stress_axial.shape[0])
        output_train = [stress_axial, stress_trans, stress_shear]
        sample_weights = [np.array([1.0]*shear_in.shape[0]), np.array([1.0]*shear_in.shape[0]), np.array([0.0]*shear_in.shape[0])]
        
    elif modelFit_mode == "C":
        stretch_in = lam_ut[:17]
        shear_in = np.array([0.0] * stretch_in.shape[0])
        input_train = [stretch_in, shear_in]
        stress_axial = P_ut[:17]
        stress_trans = np.array([0.0] * stress_axial.shape[0])
        stress_shear = np.array([0.0] * stress_axial.shape[0])
        output_train = [stress_axial, stress_trans, stress_shear]
        sample_weights = [np.array([1.0] * shear_in.shape[0]), np.array([1.0]*shear_in.shape[0]), np.array([0.0] * shear_in.shape[0])]
        
    elif modelFit_mode == "TC":
        stretch_in = lam_ut
        shear_in = np.array([0.0] * stretch_in.shape[0])
        input_train = [stretch_in, shear_in]
        stress_axial = P_ut
        stress_trans = np.array([0.0] * stress_axial.shape[0])
        stress_shear = np.array([0.0] * stress_axial.shape[0])
        output_train = [stress_axial, stress_trans, stress_shear]
        sample_weights = [np.array([1.0] * shear_in.shape[0]), np.array([1.0]*shear_in.shape[0]), np.array([0.0] * shear_in.shape[0])]
        
    elif modelFit_mode == "SS":
        shear_in = gamma_ss
        stretch_in = np.array([0.8] * shear_in.shape[0])
        input_train = [stretch_in, shear_in]
        stress_shear = P_ss
        stress_axial = np.array([0.0] * stress_shear.shape[0])
        stress_trans = np.array([0.0] * stress_axial.shape[0])
        output_train = [stress_axial, stress_trans, stress_shear]
        sample_weights = [np.array([0.0] * shear_in.shape[0]), np.array([0.0] * shear_in.shape[0]), np.array([1.0] * shear_in.shape[0])]
        
    elif modelFit_mode == "TC_and_SS":
        _, input_train_1, output_train_1, sample_weights_1 = traindata("TC")
        _, input_train_2, output_train_2, sample_weights_2 = traindata("SS")
        input_train = [np.concatenate([x, y], axis=0) for (x, y) in zip(input_train_1, input_train_2)]
        output_train = [np.concatenate([x, y], axis=0) for (x, y) in zip(output_train_1, output_train_2)]
        sample_weights = [np.concatenate([x, y], axis=0) for (x, y) in zip(sample_weights_1, sample_weights_2)]

    
    return model_given, input_train, output_train, sample_weights

        

def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights):
    
    mse_loss = keras.losses.MeanSquaredError()
    metrics  =[keras.metrics.MeanSquaredError()] * 2
    opti1    = tf.optimizers.Adam(learning_rate=0.001)
    
    model_given.compile(loss=mse_loss,
                  optimizer=opti1,
                  metrics=metrics)
    
    
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=3000, restore_best_weights=True)

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
                        sample_weight = sample_weights)
    
    return model_given, history


# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

def Stress_calc_axial(inputs):
    (dWdI1, dWdI2, dWdJ, Stretch, Gamma) = inputs
    return 2 * Stretch * dWdI1 + 4 * Stretch * dWdI2 + dWdJ

def Stress_calc_shear(inputs):
    (dWdI1, dWdI2, dWdJ, Stretch, Gamma) = inputs
    return 2 * Gamma * dWdI1 + 2 * Gamma * dWdI2

def Stress_calc_trans(inputs):
    (dWdI1, dWdI2, dWdJ, Stretch, Gamma) = inputs
    return 2 * dWdI1 + 2 * (1 + Stretch ** 2) * dWdI2 + Stretch * dWdJ


# Complte model architecture definition
def modelArchitecture(Psi_model): # TODO: redo so we are considering stretch, 0 and 0.8, gamma
    # Stretch and Gamma as input
    Stretch = keras.layers.Input(shape = (1,),
                                  name = 'Stretch')
    Gamma = keras.layers.Input(shape = (1,),
                                  name = 'gamma')

    # specific Invariants UT
    I1 = keras.layers.Lambda(lambda x: 2 + x[0]**2 + x[1] ** 2  )([Stretch, Gamma])
    I2 = keras.layers.Lambda(lambda x: 1 + 2 * x[0] ** 2 + x[1] ** 2)([Stretch, Gamma])
    J = keras.layers.Lambda(lambda x: x[0])([Stretch, Gamma])
    
    #% load specific models
    Psi = Psi_model([I1, I2, J])

    # derivative UT
    dWdI1  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I1])
    dWdI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I2])
    dWdJ  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, J ])

    # Stress UT
    Stress_axial = keras.layers.Lambda(function = Stress_calc_axial,
                                name = 'Stress_axial')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])
    # Stress UT
    Stress_trans = keras.layers.Lambda(function=Stress_calc_trans,
                                       name='Stress_trans')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])
    # Stress SS
    Stress_shear = keras.layers.Lambda(function = Stress_calc_shear,
                                name = 'Stress_shear')([dWdI1, dWdI2, dWdJ, Stretch, Gamma])

    # Define model for different load case
    model = keras.models.Model(inputs=[Stretch, Gamma], outputs= [Stress_axial, Stress_trans, Stress_shear])
    
    return Psi_model, model






#%% Init
train = True
epochs = 8000
batch_size = 64
model_type = 'CANN_test'
weight_flag = True



path2saveResults_0 = 'Results/'+filename+'/'+model_type
makeDIR(path2saveResults_0)
Model_summary = path2saveResults_0 + '/Model_summary.txt'


# Select loading modes and brain regions of interest        
modelFit_mode_all = ['SS', 'C', 'T', 'TC', "TC_and_SS"]
Region_all = ['leap', 'turbo']

# prepare R2 array
R2_all = np.zeros([len(Region_all),len(modelFit_mode_all)+2])

#Import excel file
file_name = 'input/FoamData.xlsx'
dfs = pd.read_excel(file_name, sheet_name='Sheet1')

#%%  Training and validation loop 
count = 1
for id1, Region in enumerate(Region_all):
    
    R2_all_Regions = []
    for id2, modelFit_mode in enumerate(modelFit_mode_all):
        
        print(40*'=')
        print("Comp {:d} / {:d}".format(count, len(Region_all)*len(modelFit_mode_all)))
        print(40*'=')
        print("Region: ", Region ,"| Fitting Mode: ", modelFit_mode)
        print(40*'=')
        count += 1
        
        path2saveResults = os.path.join(path2saveResults_0,Region, modelFit_mode)
        path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
        makeDIR(path2saveResults)
        makeDIR(path2saveResults_check)
        
        P_ut, lam_ut, P_ss, gamma_ss = getStressStrain(Region)
        
        #%% PI-CANN
        # Model selection
        Psi_model, terms = StrainEnergyCANN()
        Psi_model, model = modelArchitecture(Psi_model)

        
        with open(Model_summary,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
            
        #%%  Model training


        
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode)

            
        Save_path = path2saveResults + '/model.keras'
        Save_weights = path2saveResults + '/weights.weights.h5'
        path_checkpoint = path2saveResults_check + '/best_weights.weights.h5'
        if train:
            model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights)
            
            model_given.load_weights(path_checkpoint,  skip_mismatch=False)
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
            Psi_model.save_weights(Save_weights, overwrite=True)
            
            # Plot loss function
            loss_history = history.history['loss']
            fig, axe = plt.subplots(figsize=[6, 5])  # inches
            plotLoss(axe, loss_history)
            plt.savefig(path2saveResults+'/Plot_loss_'+Region+'_'+modelFit_mode+'.pdf')
            plt.close()
            
        else:
            # Psi_model = tf.keras.models.load_model(Save_path)
            Psi_model.load_weights(Save_weights,  skip_mismatch=False)
        
        
        # PI-CANN  get model response
        lam_ut_model = np.linspace(np.amin(lam_ut),np.amax(lam_ut),50)
        gamma_model = np.linspace(np.amin(gamma_ss),np.amax(gamma_ss),50)
        Stress_predict_UT = model.predict(lam_ut)
        Stress_predict_SS = model.predict(gamma_ss)
        print(Stress_predict_UT)
        

        
        #%% Plotting
        
        
        fig = plt.figure(figsize=(600/72,600/72))
        spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
        fig_ax1 = fig.add_subplot(spec[0,0])
        ax2 = fig.add_subplot(spec[1,0])
        
        R2, R2_c, R2_t = plotTenCom(fig_ax1, lam_ut, P_ut, Stress_predict_UT, Region)
        R2ss = plotShear(ax2, gamma_ss, P_ss, Stress_predict_SS, Region)
        fig.tight_layout()        

        plt.savefig(path2saveResults+'/Plot_PI-CANN_'+Region+'_'+modelFit_mode+'.pdf')
        plt.close()





        # terms =4*7
        #%% Show weights
        weight_matrix = np.empty([int(terms), 2])
        weight_matrix[:] = np.nan        
        Num_add_var = 0
        
        if weight_flag:

            second_layer = Psi_model.get_weights()[terms+Num_add_var].flatten()
            j=0
            if weight_flag:
                for i in range(int(terms)):
                    try:
                        value = Psi_model.get_weights()[i + Num_add_var][0][0]
                        weight_matrix[i,1] =  second_layer[j]
                        j+=1
                    except ValueError:
                        value = np.nan
                    
                    weight_matrix[i,0] = value
                
            
            
            print("weight_matrix")  
            print(weight_matrix)
    
    
        
        #%% Plotting
        
        Config = {Region:Region, modelFit_mode:modelFit_mode, "R2_c":R2_c, "R2_t": R2_t, "R2_TC": R2, "R2_ss": R2ss, "weigths": weight_matrix.tolist()}
        json.dump(Config, open(path2saveResults+"/Config_file.txt",'w'))
        
        if modelFit_mode == 'T':
            R2_cur = [R2_t]
        elif modelFit_mode == "C":
            R2_cur = [R2_c]
        elif modelFit_mode == "TC":
            R2_cur = [R2] 
        elif modelFit_mode == "SS":
            R2_cur = [R2ss]    
        elif modelFit_mode == "TC_and_SS":
            R2_cur = [R2ss, R2_c, R2_t]
         
        R2_all_Regions.append(R2_cur)
        print("R2: ", R2_cur) 
            

    R2_all[id1,:] = np.array(flatten(R2_all_Regions))
    
    


#%% Summarizing results    
modelFit_mode_all_table = ['SS', 'C', 'T', 'TC', "SS-sim", "C-sim", "T-sim"]
R2_mean = np.expand_dims(np.mean(R2_all,axis=0), axis=0)
R2_sd = np.expand_dims(np.std(R2_all,axis=0), axis=0)
R2_all_mean = np.concatenate((R2_all,R2_mean,R2_sd), axis=0)
R2_df = pd.DataFrame(R2_all_mean, index=Region_all + ['mean', 'SD'], columns=modelFit_mode_all_table)
R2_df.to_latex(path2saveResults_0+'/R2_table.tex',index=True)
R2_df.to_csv(path2saveResults_0+'/R2_table.csv',index=True)

