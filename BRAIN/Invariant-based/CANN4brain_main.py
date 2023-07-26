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
    #FOR FITTING compression
    
    if Region =='CX':
        P_ut = dfs.iloc[3:,1].astype(np.float64)
        lam_ut = dfs.iloc[3:,0].astype(np.float64)
        
        P_ss = dfs.iloc[3:,3].astype(np.float64).values
        gamma_ss = dfs.iloc[3:,2].astype(np.float64).values
    elif Region =='CR':
        P_ut = dfs.iloc[3:,6].astype(np.float64)
        lam_ut = dfs.iloc[3:,5].astype(np.float64)
        
        P_ss = dfs.iloc[3:,8].astype(np.float64).values
        gamma_ss = dfs.iloc[3:,7].astype(np.float64).values
    elif Region =='BG':
        P_ut = dfs.iloc[3:,11].astype(np.float64)
        lam_ut = dfs.iloc[3:,10].astype(np.float64)
        
        P_ss = dfs.iloc[3:,13].astype(np.float64).values
        gamma_ss = dfs.iloc[3:,12].astype(np.float64).values
    elif Region =='CC':
        P_ut = dfs.iloc[3:,16].astype(np.float64)
        lam_ut = dfs.iloc[3:,15].astype(np.float64)
        
        P_ss = dfs.iloc[3:,18].astype(np.float64).values
        gamma_ss = dfs.iloc[3:,17].astype(np.float64).values
        
    
    P_ut_all =P_ut
    lam_ut_all =lam_ut

    return P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss

   
def traindata(modelFit_mode):
    weighing_TC = np.array([0.5]*lam_ut[:16].shape[0] + [1.5]*lam_ut[16:].shape[0])
    
    if modelFit_mode == 'T':
        model_given = model_UT
        input_train = lam_ut[16:]
        output_train = P_ut[16:]
        sample_weights = np.array([1.0]*input_train.shape[0])
        
    elif modelFit_mode == "C":
        model_given = model_UT
        input_train = lam_ut[:17]
        output_train = P_ut[:17]
        sample_weights = np.array([1.0]*input_train.shape[0])
        
    elif modelFit_mode == "TC":
        model_given = model_UT
        input_train = lam_ut
        output_train = P_ut
        sample_weights = weighing_TC
        
    elif modelFit_mode == "SS":
        model_given = model_SS
        input_train = gamma_ss
        output_train = P_ss
        sample_weights = np.array([1.0]*input_train.shape[0])
        
    elif modelFit_mode == "TC_and_SS":
        model_given = model
        input_train = [lam_ut, gamma_ss]
        output_train = [P_ut, P_ss]
        sample_weights_tc = weighing_TC
        sample_weights_ss = np.array([1.0]*gamma_ss.shape[0])
        sample_weights = [sample_weights_tc, sample_weights_ss]
    
    return model_given, input_train, output_train, sample_weights

        

def Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights):
    
    mse_loss = keras.losses.MeanSquaredError()
    metrics  =[keras.metrics.MeanSquaredError()]
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




#%% modeling


# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

def Stress_calc_TC(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

#   calculate cauchy stress sigma
    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    minus  = two * ( dPsidI1 *             1/ K.square(Stretch)  + dPsidI2 *      1/K.pow(Stretch,3)   ) 
    stress = two * ( dPsidI1 *  Stretch                          + dPsidI2 *  one                      ) - minus

    return stress

# Simple stress P12
def Stress_cal_SS(inputs):

    (dPsidI1, dPsidI2, gamma) = inputs

    two = tf.constant(2.0,dtype='float32')     

    # Shear stress 
    stress = two * gamma * ( dPsidI1  + dPsidI2 )

    return stress



# Complte model architecture definition
def modelArchitecture(Psi_model):
    # Stretch and Gamma as input
    Stretch = keras.layers.Input(shape = (1,),
                                  name = 'Stretch')
    Gamma = keras.layers.Input(shape = (1,),
                                  name = 'gamma')

    # specific Invariants UT
    I1_UT = keras.layers.Lambda(lambda x: x**2   + 2.0/x  )(Stretch)
    I2_UT = keras.layers.Lambda(lambda x: 2.0*x  + 1/x**2 )(Stretch)
    # specific Invariants SS
    I1_SS = keras.layers.Lambda(lambda x: x**2 + 3.0 )(Gamma)
    I2_SS = keras.layers.Lambda(lambda x: x**2 + 3.0 )(Gamma)
    
    #% load specific models
    Psi_UT = Psi_model([I1_UT, I2_UT])
    Psi_SS = Psi_model([I1_SS, I2_SS])
    
    # derivative UT
    dWI1_UT  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I1_UT])
    dWdI2_UT = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_UT, I2_UT])
    # derivative SS
    dWI1_SS  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS, I1_SS])
    dWdI2_SS = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi_SS, I2_SS])
    
    # Stress UT
    Stress_UT = keras.layers.Lambda(function = Stress_calc_TC,
                                name = 'Stress_UT')([dWI1_UT,dWdI2_UT,Stretch])
    # Stress SS
    Stress_SS = keras.layers.Lambda(function = Stress_cal_SS,
                                name = 'Stress_SS')([dWI1_SS,dWdI2_SS,Gamma])
    
    
    # Define model for different load case
    model_UT = keras.models.Model(inputs=Stretch, outputs= Stress_UT)
    model_SS = keras.models.Model(inputs=Gamma, outputs= Stress_SS)
    # Combined model
    model = keras.models.Model(inputs=[model_UT.inputs, model_SS.inputs], outputs=[model_UT.outputs, model_SS.outputs])
    
    return model_UT, model_SS, Psi_model, model






#%% Init
train = True
epochs = 8000
batch_size = 8 
model_type = 'CANN_test'
weight_flag = True



path2saveResults_0 = 'Results/'+filename+'/'+model_type
makeDIR(path2saveResults_0)
Model_summary = path2saveResults_0 + '/Model_summary.txt'


# Select loading modes and brain regions of interest        
modelFit_mode_all = ['SS', 'C', 'T', 'TC', "TC_and_SS"]
Region_all = ['CC', 'BG', 'CX', 'CR']

# prepare R2 array
R2_all = np.zeros([len(Region_all),len(modelFit_mode_all)+2])

#Import excel file
file_name = 'input/CANNsBRAINdata.xlsx'
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
        
        P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss = getStressStrain(Region)
        
        #%% PI-CANN
        # Model selection
        Psi_model, terms = StrainEnergyCANN()
        model_UT, model_SS, Psi_model, model = modelArchitecture(Psi_model)

        
        with open(Model_summary,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            Psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
            
        #%%  Model training


        
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode)

            
        Save_path = path2saveResults + '/model.h5'
        Save_weights = path2saveResults + '/weights'
        path_checkpoint = path2saveResults_check + '/best_weights' 
        if train:
            model_given, history = Compile_and_fit(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights)
            
            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
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
            Psi_model.load_weights(Save_weights, by_name=False, skip_mismatch=False)
        
        
        # PI-CANN  get model response
        lam_ut_model = np.linspace(np.amin(lam_ut),np.amax(lam_ut),50)
        gamma_model = np.linspace(np.amin(gamma_ss),np.amax(gamma_ss),50)
        Stress_predict_UT = model_UT.predict(lam_ut_all)
        Stress_predict_SS = model_SS.predict(gamma_ss)
        
        

        
        #%% Plotting
        
        
        fig = plt.figure(figsize=(600/72,600/72))
        spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
        fig_ax1 = fig.add_subplot(spec[0,0])
        ax2 = fig.add_subplot(spec[1,0])
        
        R2, R2_c, R2_t = plotTenCom(fig_ax1, lam_ut_all, P_ut_all, Stress_predict_UT, Region)
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

