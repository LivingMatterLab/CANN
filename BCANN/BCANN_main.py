#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:30:27 2021

@author: kvn
"""
import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow.keras as keras

import tensorflow_probability as tfp
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
        

filename = os.path.basename(__file__)[:-3]
cwd = os.getcwd()
path2saveResults_0 = 'Results/'+filename



def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
makeDIR(path2saveResults_0)

    
tfd = tfp.distributions
tfb = tfp.bijectors



#%% Helper functions
def myGradient(a, b):
	der = tf.gradients(a, b, unconnected_gradients='zero')
	return der[0]

def NLL(y, distr): 
  return -distr.log_prob(y) 


Softplus_off_np = np.log(np.exp(0) + 1)
Softplus_off = tf.constant(Softplus_off_np,dtype='float32')    



#%% Create data
## use custom function to simulate some data 
def Ogden(stretch,mu,alpha):
 return 2* mu/alpha *(stretch**(alpha-1) - stretch**(-alpha/2 -1))

def MR_UT(lam_UT, w1, w5):
 return 2.0*(lam_UT - 1.0/lam_UT**2)*(w1 + w5/lam_UT)

def MR_ET(lam_ET, w1, w5):
 return 2.0*(lam_ET - 1.0/lam_ET**5)*(w1 + w5*lam_ET**2)

def MR_PS(lam_PS, w1, w5):
 return 2.0*(w1 + w5)*(lam_PS - 1.0/lam_PS**3)


SD = 0.1
def create_Noise2data(y1, n = 2048, SD=0.15):
    np.random.seed(32)
    y1 = y1+np.random.normal(0,SD*np.abs(y1),n)
    return y1

Data_points = 512
mu = 1
alpha = -3
LamStart = 0.8
LamEnd = 1.3

lam_UT = np.linspace(LamStart, LamEnd, Data_points)
lam_ET = np.linspace(1.0, LamEnd, Data_points)
lam_PS = np.linspace(1.0, LamEnd, Data_points)
y1_UT = MR_UT(lam_UT,1.0,1.0)
y1_ET = MR_ET(lam_ET,1.0,1.0)
y1_PS = MR_PS(lam_PS,1.0,1.0)

stress_ut = create_Noise2data(y1_UT, Data_points, SD) 
stress_et = create_Noise2data(y1_ET, Data_points, SD) 
stress_ps = create_Noise2data(y1_PS, Data_points, SD) 




#%% Model 
from tensorflow.keras.optimizers import Adam


acti1='tanh'
acti2 = 'sigmoid'
acti3 = 'softplus'
acti4 = 'linear'
def flatten(l):
    return [item for sublist in l for item in sublist]

# Self defined activation functions for exp 
def activation_Exp(x):
    return 1.0*(tf.math.exp(x) -1.0) 

def activation_ln(x):
    return -1.0*tf.math.log(1.0 - (x))

# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

# Continuum mechanics stress definition for uniaxial tension only! #LN is this cauchy = 1/J * P * F^t, i.e., P_LN * Stretch (?)
def Stress_calc_TC(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

#   calculate cauchy stress sigma
    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    minus  = two * ( dPsidI1 *             1/ K.square(Stretch)  + dPsidI2 *      1/K.pow(Stretch,3)   ) 
    stress = two * ( dPsidI1 *  Stretch                          + dPsidI2 *  one                      ) - minus

    return stress

# Continuum mechanics stress definition for biaxial tension only
def Stress_calc_ET(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    # minus  = two * ( dPsidI1 *            one/K.pow(Stretch,5)  + dPsidI2 *                   one/K.pow(Stretch,3)   ) 
    # stress = two * ( dPsidI1 *  Stretch                       + dPsidI2 *  K.pow(Stretch,3)                      ) - minus

    stress = two*(dPsidI1 + K.pow(Stretch,2)* dPsidI2)*(Stretch - one/K.pow(Stretch,5))
    
    return stress

# Continuum mechanics stress definition for pure shear only
def Stress_calc_PS(inputs):

    (dPsidI1, dPsidI2, Stretch) = inputs

    one = tf.constant(1.0,dtype='float32')    
    two = tf.constant(2.0,dtype='float32')     

    minus  = two * ( dPsidI1 *             one/K.pow(Stretch,3)  + dPsidI2 *          one/K.pow(Stretch,3)   ) 
    stress = two * ( dPsidI1 *  Stretch                        + dPsidI2 *  Stretch                      ) - minus

    return stress


init_all = tf.keras.initializers.RandomUniform(minval=0.2, maxval=1.0)
initializer_exp = tf.keras.initializers.RandomUniform(minval=0.2, maxval=1.0) # worked off and on, starts with huge residual

def identity(x):
    return x

def make_nonneg(x):
    return tf.clip_by_value(x,0.0,np.inf)

# scalar network layer
class NormalizedDense(keras.layers.Layer):
    def __init__(self, kernel_initializer, activation, weight_name, should_normalize, p, alpha, **kwargs):
        super().__init__(**kwargs)
        # Save inputs as class variables
        self.kernel_initializer=kernel_initializer
        self.activation=identity if activation is None else activation
        self.weight_name=weight_name
        self.p = p
        self.alpha = alpha
        self.should_normalize = should_normalize


    # Required function for reading / writing layer to file
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "activation": self.activation,
            "weight_name": self.weight_name,
            "p": self.p,
            "alpha": self.alpha,
            "should_normalize": self.should_normalize
        })
        return config

    # Create relevant weights
    def build(self, input_shape):

        self.w1 = self.add_weight(
            shape=(1, ),
            initializer=tf.keras.initializers.RandomUniform(minval=1.0, maxval=2.5),
            constraint=keras.constraints.NonNeg(),
            trainable=True,
            name=self.weight_name + '1'
        )
        self.w2 = self.add_weight(
            shape=(1,),
            initializer = tf.keras.initializers.Constant(value=1.0),
            constraint=keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(self.alpha),
            trainable=True,
            name=self.weight_name + '2'
        )

    # Compute and return output of layer given input
    def call(self, inputs):

        return self.w2  * self.activation(inputs * self.w1)

        
        
def StrainEnergyPICANN_BCANN(x):
    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I_in = tf.keras.layers.concatenate([I1_in, I2_in], axis=1)

    # Invariants reference confi  
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)


    L2  = 0.000
    fac  = 0.000

    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (lam_UT.shape[0] * 1.0)
    bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (lam_UT.shape[0] * 1.0)

    Input_list = [I1_ref, I1_ref, tf.math.square(I1_ref), tf.math.square(I1_ref), I2_ref, I2_ref, tf.math.square(I2_ref), tf.math.square(I2_ref)]
    activation_list = [None, activation_Exp, None, activation_Exp, None, activation_Exp, None, activation_Exp] 
    init_list = [init_all, initializer_exp, init_all, initializer_exp, init_all, initializer_exp, init_all, initializer_exp]

    
    terms = len(Input_list)
    W_para =list()
    for ii in range(terms):
        # Here is the scalar layer of the network
        I_1_w11 = NormalizedDense(kernel_initializer=init_list[ii], activation=activation_list[ii], weight_name='w'+str(1+ii),
                                      should_normalize=False, p=1.0, alpha=0.001)(Input_list[ii])

        # Uncertainty layer
        W_para_I1 = tfp.layers.DenseFlipout(2,bias_posterior_fn=None,
                               bias_prior_fn=None,
                               kernel_divergence_fn=kernel_divergence_fn,
                               activation=None)(I_1_w11)
        
        W_para.append(W_para_I1)
    
    Psi_model = keras.models.Model(inputs=[I1_in, I2_in], outputs=W_para, name='Psi')
    
    
    return Psi_model, terms




Para_SD = 0.05
def normal_sp(params): 
  return tfd.Normal(loc=params[:,0:1], scale=1e-3 +  tf.nn.elu(Para_SD *  params[:,1:2]) )# both parameters are learnable


# Here we compute the stress from the strain energy slitted for the different model terms
def Calc_Stress_contributions(terms, Psi, Stress_calc_func, I1, I2, Stretch, label):
    
    Stress_UT_all_list =list()
    for i in range(terms):
        all_dWI1_UT_cur = (make_nonneg(keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi[i][:,0], I1])))
        all_dWI2_UT_cur = (make_nonneg(keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi[i][:,0], I2])))
        all_dWI1_UT_sd_cur = (make_nonneg(keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi[i][:,1], I1])))
        all_dWI2_UT_sd_cur = (make_nonneg(keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi[i][:,1], I2])))
        
        
        Stress_UT = keras.layers.Lambda(function = Stress_calc_func,
                                    name = 'Stress_'+label+'_'+str(i))([(all_dWI1_UT_cur),(all_dWI2_UT_cur),Stretch])

        Stress_UT_sd = keras.layers.Lambda(function = Stress_calc_func,
                                    name = 'Stress_'+label+'_sd'+str(i))([(all_dWI1_UT_sd_cur),(all_dWI2_UT_sd_cur),Stretch])
        
        To_append_UT  = tf.expand_dims(tf.keras.layers.concatenate([Stress_UT, tf.math.abs(Stress_UT_sd)],axis=1), axis=2)
        Stress_UT_all_list.append(To_append_UT)
            

    Stress_UT_all_list_con = tf.keras.layers.concatenate(Stress_UT_all_list, axis=2)
     

    Stress_UT_dist_all =list()
    for i in range(terms):
        Stress_UT_dist_all.append(tfp.layers.DistributionLambda(normal_sp)(Stress_UT_all_list_con[:,:,i]))
        
    Stress_UT_dist_end = tf.reduce_sum(Stress_UT_all_list_con, axis=2)
 
    Stress_UT_dist = tfp.layers.DistributionLambda(normal_sp)(Stress_UT_dist_end)

    return Stress_UT_dist_all, Stress_UT_dist
    
    
    



def modelArchitecture(Psi_model,terms):

    ##########################################
    ##########################################
    ## UT
    ##########################################
    ##########################################    
    Stretch_UT = keras.layers.Input(shape = (1,),
                                  name = 'Stretch_UT')
    # specific Invariants UT
    I1_UT = keras.layers.Lambda(lambda x: x**2   + 2.0/x  )(Stretch_UT)
    I2_UT = keras.layers.Lambda(lambda x: 2.0*x  + 1/x**2 )(Stretch_UT)
    #% load specific models
    Psi_UT = Psi_model([I1_UT, I2_UT])


    Stress_UT_dist_all, Stress_UT_dist = Calc_Stress_contributions(terms, Psi_UT, Stress_calc_TC, I1_UT, I2_UT, Stretch_UT, 'UT')
    
    
    ##########################################
    ##########################################
    ## ET
    ##########################################
    ##########################################
    Stretch_ET = keras.layers.Input(shape = (1,), name = 'Stretch_ET')
    # specific Invariants ET
    I1_ET = keras.layers.Lambda(lambda x: 2.0*x**2 + 1.0/x**4 )(Stretch_ET)
    I2_ET = keras.layers.Lambda(lambda x:     x**4 + 2.0/x**2 )(Stretch_ET)
    #Psi ET
    Psi_ET = Psi_model([I1_ET, I2_ET])

    Stress_ET_dist_all, Stress_ET_dist = Calc_Stress_contributions(terms, Psi_ET, Stress_calc_ET, I1_ET, I2_ET, Stretch_ET, 'ET')

    ##########################################
    ##########################################
    ## PS
    ##########################################
    ##########################################
    Stretch_PS = keras.layers.Input(shape = (1,), name = 'Stretch_PS')
    # specific Invariants PS
    I1_PS = keras.layers.Lambda(lambda x: x**2 + 1.0 + 1.0/x**2 )(Stretch_PS)
    I2_PS = keras.layers.Lambda(lambda x: x**2 + 1.0 + 1.0/x**2 )(Stretch_PS)
    # Psi Energy
    Psi_PS = Psi_model([I1_PS, I2_PS])

    # derivative PS
    Stress_PS_dist_all, Stress_PS_dist = Calc_Stress_contributions(terms, Psi_PS, Stress_calc_PS, I1_PS, I2_PS, Stretch_PS, 'PS')
    
    ##########################################
    ##########################################
    ## ALL Models 
    ##########################################
    ##########################################
    # Define model for different load case
    model_UT = keras.models.Model(inputs=Stretch_UT, outputs= Stress_UT_dist)
    model_UT_single = keras.models.Model(inputs=Stretch_UT, outputs= Stress_UT_dist_all)
    
    model_ET = keras.models.Model(inputs=Stretch_ET, outputs= Stress_ET_dist)
    model_ET_single = keras.models.Model(inputs=Stretch_ET, outputs= Stress_ET_dist_all)
    
    model_PS = keras.models.Model(inputs=Stretch_PS, outputs= Stress_PS_dist)
    model_PS_single = keras.models.Model(inputs=Stretch_PS, outputs= Stress_PS_dist_all)
    
    model = keras.models.Model(inputs=[Stretch_UT, Stretch_ET, Stretch_PS],
                            outputs=[Stress_UT_dist, Stress_ET_dist, Stress_PS_dist])
    
    model_bi = keras.models.Model(inputs=[Stretch_UT, Stretch_ET],
                            outputs=[Stress_UT_dist, Stress_ET_dist])    
    
    return model_UT, model_UT_single, model_ET, model_ET_single, model_PS, model_PS_single, model_bi, model




#%% Training
train=True
Model_version = 'test_v1'

Psi_model, terms = StrainEnergyPICANN_BCANN(lam_UT)
model_UT, model_UT_single, model_ET, model_ET_single, model_PS, model_PS_single, model_bi, model= modelArchitecture(Psi_model,terms)


input_train = [lam_UT, lam_ET]
output_train = [stress_ut, stress_et]
model_given = model_bi

model_given.compile(Adam(learning_rate=0.0002),loss=NLL)



#%% paths
path2saveResults = os.path.join(path2saveResults_0, Model_version)
makeDIR(path2saveResults)
Save_weights = path2saveResults + '/weights'

path2saveResults_check = os.path.join(path2saveResults,'Checkpoints')
makeDIR(path2saveResults_check)
#%% Model training

modelckpt_callback = keras.callbacks.ModelCheckpoint(
monitor="loss",
filepath=path2saveResults_check + '/best_weights',
verbose=0,
save_weights_only=True,
save_best_only=True,
)


if train:
    print('Start fitting')
    history = model_given.fit(input_train, output_train, epochs=2500, callbacks=[modelckpt_callback], verbose=2,batch_size=32)
    Psi_model.save_weights(Save_weights, overwrite=True)
else:
    print('loading weights fitting')
    Psi_model.load_weights(Save_weights, by_name=False, skip_mismatch=False)
    

# Loss plot
if train:
    LossTrainL = history.history['loss'][-1]
    plt.figure(figsize=[6, 5])  # inches

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.ylim([-1.0, 5.0])
    plt.show()


#%% Post=processing
runs = 500
x_pred_ut_all = np.linspace(np.amin(lam_UT),np.amax(lam_UT),50) 
x_pred_uc = np.linspace(np.amin(lam_UT),1.0,50)
x_pred_ut = np.linspace(1.0,np.amax(lam_UT),50) 
x_pred_et = np.linspace(np.amin(lam_ET),np.amax(lam_ET),50) 
x_pred_ps = np.linspace(np.amin(lam_PS),np.amax(lam_PS),50) 

preds_UT_all =np.zeros((runs,len(x_pred_ut_all)))
preds_UC =np.zeros((runs,len(x_pred_uc)))
preds_UT =np.zeros((runs,len(x_pred_ut)))
preds_ET =np.zeros((runs,len(x_pred_et)))
preds_PS =np.zeros((runs,len(x_pred_ps)))


preds_UT_single_all =np.zeros((runs,len(x_pred_ut_all),terms))
preds_UC_single =np.zeros((runs,len(x_pred_uc),terms))
preds_UT_single =np.zeros((runs,len(x_pred_ut),terms))
preds_ET_single =np.zeros((runs,len(x_pred_ut),terms))
preds_PS_single =np.zeros((runs,len(x_pred_ut),terms))
from tqdm import tqdm_notebook as tqdm
for i in tqdm(range(0,runs)):
    preds_UT_all[i,:]=np.reshape(model_UT.predict(x_pred_ut_all),len(x_pred_ut_all))
    preds_UC[i,:]=np.reshape(model_UT.predict(x_pred_uc),len(x_pred_uc))
    preds_UT[i,:]=np.reshape(model_UT.predict(x_pred_ut),len(x_pred_ut))
    preds_ET[i,:]=np.reshape(model_ET.predict(x_pred_et),len(x_pred_et))
    preds_PS[i,:]=np.reshape(model_PS.predict(x_pred_ps),len(x_pred_ps))
    for jj in range(terms):
        preds_UT_single_all[i,:,jj] = model_UT_single.predict(x_pred_ut_all)[jj].flatten()
        preds_UC_single[i,:,jj] = model_UT_single.predict(x_pred_uc)[jj].flatten()
        preds_UT_single[i,:,jj] = model_UT_single.predict(x_pred_ut)[jj].flatten()
        preds_ET_single[i,:,jj] = model_ET_single.predict(x_pred_et)[jj].flatten()
        preds_PS_single[i,:,jj] = model_PS_single.predict(x_pred_ps)[jj].flatten()




#%% Plotting

# plt.show()
def MapPlot(fig, axe, x_pred_ut, lam_UT, stress_ut, preds_UT, preds_UT_single, xlim, ylim, yticks, xticks, label, LabelFlag, TickFlag):
    cmap = plt.cm.get_cmap('turbo',8)   # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap_r = list(reversed(cmaplist))
    
    if LabelFlag:
        axe.axvline(x=1, lw=0.5,alpha=0.5,color='k',zorder=1)
        axe.axhline(y=0, lw=0.5,alpha=0.5,color='k',zorder=1)
    All_pre = np.zeros((len(x_pred_ut),terms))
    for jj in range(terms):
        
        cur = np.mean(preds_UT_single[:,:,jj],axis=0).flatten()
        
        lower = np.sum(All_pre,axis=1)
        upper = lower + cur
        All_pre[:,jj] = cur
        
        axe.fill_between(x_pred_ut, lower.flatten(), upper.flatten(), lw=0,zorder=1, alpha=1.0, color=cmap_r[jj])
        axe.plot(x_pred_ut, upper, lw=0.4, zorder=100, color='k')
        
        axe.tick_params(axis='both', labelsize='xx-large')

    axe.plot(x_pred_ut,np.mean(preds_UT,axis=0),color="k",linewidth=2.0,zorder=100)
    axe.scatter(lam_UT, stress_ut, s=2 ,color="k", alpha=1.0, zorder=150, marker='.',label='data '+label) #observerd    

    
    if not TickFlag:
        axe.set_xticklabels([])
        axe.set_yticklabels([])
    else:
        axe.set_xlabel(r'stretch $\lambda$ [-]',fontsize='xx-large')

    
    if LabelFlag:
        axe.set_ylabel(r'stress $P$ [kPa]',fontsize='xx-large')    
        
    
    axe.set_xlim(xlim)
    axe.set_ylim(ylim)
    axe.set_yticks(yticks)
    axe.set_xticks(xticks)
    


#%% Plotting

colR = (0./256,21./256,187./256,1.0)   #0015bb

def make_plot_runs_avg_full(ax,lam, x_pred, stress, Stress_mean, preds, SetLabel, YTicks, Ylim, XTicks, Xlim, LabelFlag, YTIC):

    ax.plot(x_pred,np.mean(preds,axis=0),color=colR,linewidth=2.0,zorder=10)
    ax.scatter(lam, stress,color="k",s=2, alpha=1.0,zorder=5, marker='.',label=SetLabel) #observerd  
    ax.fill_between(x_pred,np.quantile(preds, 0.025, axis=0), np.quantile(preds, 0.975, axis=0), lw=0,zorder=1, alpha=0.25, color=colR)
    ax.scatter(lam[::40], Stress_mean[::40],s=50, zorder=9, marker='o', facecolor='w', lw=0.5, color='k');
    
    if YTIC:
        ax.set_yticks(YTicks)
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])

    if LabelFlag:
        ax.set_ylabel(r'stress $P$ [kPa]',fontsize='xx-large')    
                
    ax.set_xticks(XTicks)
    ax.set_ylim(Ylim)
    ax.set_xlim(Xlim)
    # fig_ax0.legend(loc='upper left',ncol=1, fancybox=True, framealpha=0.,fontsize='xx-large')
    # fig_ax0.set_xlabel(r'stretch $\lambda$ [-]',fontsize='xx-large')
    ax.tick_params(axis='both', labelsize='xx-large')
    ax.set_xticklabels([])

    
 
lam_UT_p = np.linspace(1.0, LamEnd, Data_points)
y1_UT_p = MR_UT(lam_UT_p,1.0,1.0)
stress_ut_p = create_Noise2data(y1_UT_p, Data_points, SD) 

lam_UC = np.linspace(LamStart, 1.0, Data_points)
y1_UC = MR_UT(lam_UC,1.0,1.0)
stress_uc = create_Noise2data(y1_UC, Data_points, SD) 
compression_filter = np.where(lam_UT <= 1.0)
tension_filter = np.where(lam_UT <= 1.0)
tension_filter = np.where(lam_UT >= 1.0)


lines = 5
xLim = 1.3

YTIC =True

#%% Plotting

px = 1/plt.rcParams['figure.dpi']  # pixel in inches

fig =  plt.figure(figsize=(600*px, 260*px))


width_r = [3/5]*2

spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig, wspace=0.2, hspace=0.1, height_ratios=width_r)

fig_ax0 = fig.add_subplot(spec[0,0])
fig_ax1 = fig.add_subplot(spec[0,1])
ax2 = fig.add_subplot(spec[0,2])
ax3 = fig.add_subplot(spec[0,3])



#########################
##### Compression
#########################
# ax,lam, x_pred, stress, preds, SetLabel, YTicks, Ylim, XTicks, Xlim, LabelFlag, YTIC
# YTicks, Ylim, XTicks, Xlim, YTIC
make_plot_runs_avg_full(fig_ax0, lam_UC, x_pred_uc, stress_uc, y1_UC, preds_UC, 'data UC', [0, -4.0], [0, -4.0], [1.0, 0.8], [1.0, 0.8],False, YTIC)


#########################
##### Tension
#########################
make_plot_runs_avg_full(fig_ax1,lam_UT_p, x_pred_ut, stress_ut_p, y1_UT_p, preds_UT, 'data UT', [0, 3.00], [0.0, 3.0], [1.0, 1.3], [1.0, 1.3],False,YTIC)



#########################
##### Biaxial  tensipn
#########################
make_plot_runs_avg_full(ax2,lam_ET, x_pred_et, stress_et, y1_ET, preds_ET, 'data ET', [0, 7.00], [0.0, 7.0], [1.0, 1.3], [1.0, 1.3],False,YTIC)



#########################
##### Pure Shear
#########################
make_plot_runs_avg_full(ax3,lam_PS, x_pred_ps, stress_ps, y1_PS, preds_PS, 'data PS', [0, 4.00], [0.0, 4.0], [1.0, 1.3], [1.0, 1.3],False,YTIC)

#%%

ax0 = fig.add_subplot(spec[1,0])
ax1 = fig.add_subplot(spec[1,1])
ax2 = fig.add_subplot(spec[1,2])
ax3 = fig.add_subplot(spec[1,3])


MapPlot(fig, ax0, x_pred_uc, lam_UC, stress_uc, preds_UC, preds_UC_single, [1.0, 0.8], [0, -4.0], [0, -4.00], [1.0, 0.80], "UC",False,YTIC)
MapPlot(fig, ax1, x_pred_ut, lam_UT, stress_ut, preds_UT, preds_UT_single, [1.0, 1.3], [0, 3.0], [0, 3.00], [1.0, 1.30], "UT",False,YTIC)
MapPlot(fig, ax2, x_pred_et, lam_ET, stress_et, preds_ET, preds_ET_single, [1.0, 1.3], [0.0, 7.0], [0, 7.00], [1.0, 1.30], "ET", False,YTIC)
MapPlot(fig, ax3, x_pred_ps, lam_PS, stress_ps, preds_PS, preds_PS_single, [1.0, 1.3], [0.0, 4.0], [0, 4.00], [1.0, 1.30], "PS", False,YTIC)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(path2saveResults+'/results_map3_600_160_2.pdf')  





