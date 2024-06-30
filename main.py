# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:30:26 2020

@author: Recklexx
"""
# import keras
# import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from curvefitdeep import train_model, loss_model_sqexp, loss_model_ncexp
from data import read_dicom
from data import create_data_PixelModelTrain, load_data_PixelModelTrain
from data import create_data_ResPredCheck, load_data_ResPredCheck
from fit import fit_pixel_offset, fit_pixel_trunc, fit_pixel_ncexp, fit_pixel_sqexp, fit_pixel_net
from fit import fit_image_offset, fit_image_net, fit_image_trunc, fit_image_sqexp, fit_image_ncexp
from visualization import vis_mean_std_simu_pixel, vis_vivo_pixel_data, vis_map_r2s
# from preprocessing import denoise_nlm_multi
from postprocessing import res_stack
import config

if __name__ =='__main__':
    
# =============================================================================
#     config environment
# =============================================================================
    config.config_gpu(2)
    
# =============================================================================
#     Training and fitting processing parameters
# =============================================================================
    # whether to recreate the data for training
    flag_recreate_train_data = False
    # flag_recreate_train_data = True
    
    # whether to retrain the deep learning model 
    # flag_retrain = False
    flag_retrain = True
    
    # whether to recreate the data for testing
    flag_recreate_test_data = False
    # flag_recreate_test_data = True
    
    # whether to refit the result using conventional model
    flag_refit_conventional = False
    # flag_refit_conventional = True
    
    # whether to refit the result using deep learning model
    # flag_refit_net = False
    flag_refit_net = True
    
    flag_recalcT2map_conventional = False
    # flag_recalcT2map_conventional = True
    
    # choose the depp lraning version
    version = 6
    
# =============================================================================
# Create data for deep learning model training by simulation.
# The noise simulated is noncentral chi square distribution noise.
# =============================================================================
    # recreate train data
    if flag_recreate_train_data:
        create_data_PixelModelTrain()
    
    # load saved simulated train data
    data_train_x,data_train_y,data_train_pars = load_data_PixelModelTrain()
    data_train_x = data_train_x.astype('float32')
    data_train_y = data_train_y.astype('float32')
    TEs_train = data_train_pars['TEs']
        
# =============================================================================
# Train model
# =============================================================================
    # start training or reconstruction
    if flag_retrain:
        print('Retrain the deep learning model ...')
        # start training
        model = train_model(train_x=data_train_x,train_y=data_train_y,TEs_train = TEs_train,version=version)
        
        # save model into folder
        model.save(os.path.join('model','%s%d%s'%('DeepT2modelv',version,'.h5')),save_format='h5')

    else:
        print('Reconstruct the saved model ...')
        custom_object = {'loss_model_sqexp':loss_model_sqexp,
                         'loss_model_ncexp':loss_model_ncexp}
        model = keras.models.load_model(os.path.join('model','%s%d%s'%('DeepT2modelv',version,'.h5')),custom_objects=custom_object)
        # model = keras.models.load_model('%s%d'%('DeepT2modelv',version),custom_objects=custom_object)
    
# =============================================================================
# Create data for testing model by simulation
# =============================================================================
    # start simulation
    if flag_recreate_test_data:
        create_data_ResPredCheck()
        
    # load saved simulated test data
    data_test_x,_,data_test_pars = load_data_ResPredCheck()
    data_test_x = data_test_x.astype('float32')
    
# =============================================================================
# Use traditional model to predict test data's result
# =============================================================================  
    # calculate the means of the signal of pixels over ROI
    data_test_x = np.mean(data_test_x,axis=-2)
    TEs_test = data_test_pars['TEs']
    
    if flag_refit_conventional:
        fit_pixel_offset(data_test_x,data_test_pars)
        fit_pixel_trunc(data_test_x,data_test_pars)
        fit_pixel_ncexp(data_test_x,data_test_pars)
        fit_pixel_sqexp(data_test_x,data_test_pars)
    
# =============================================================================
# Use the deep learning model to predict test data's result    
# =============================================================================
    if flag_refit_net:
        fit_pixel_net(model,version,data_test_x,data_test_pars)
    
# =============================================================================
# Plot accuracy and percision of the result
# ============================================================================= 
    index_noise_level = 0
    index_s0 = 0
    # visulization of the mean and std on test data
    vis_mean_std_simu_pixel(index_noise_level,index_s0,version)
    
# =============================================================================
# Test on clinial data 
# Based on num_point mean points.
# =============================================================================
    # vis_vivo_pixel_data(model,version)
    
# =============================================================================
# Apply to real liver dicom images
# =============================================================================
    print('Load the dicom images...')
    filepath_dicom = os.path.join('datadcm','study_severe','*.dcm')
    data_dicom, TEs_dicom = read_dicom(filepath=filepath_dicom)
    data_dicom = np.array(data_dicom)
    data_dicom = data_dicom.astype('float32')
    print('Load done.')
    
    # data_dicom = denoise_nlm_multi(data_dicom)
    
    # calculate map using conventional model
    if flag_recalcT2map_conventional:   
        fit_image_offset(data_dicom,TEs_dicom)
        fit_image_trunc(data_dicom,TEs_dicom)
        fit_image_sqexp(data_dicom,TEs_dicom)
        fit_image_ncexp(data_dicom,TEs_dicom)
    
    # calculate map using deep learning model
    fit_image_net(data_dicom,TEs_dicom,model,version)
    
    # visualization of fitted map
    vis_map_r2s()
    
# =============================================================================
#     
# =============================================================================

    from fit import load_res_fit_pixel_net
    
    res_s0,res_r2,res_sigma = load_res_fit_pixel_net(version,data_test_pars)

    from fit import load_res_fit_image_net
    map_t2_net, map_s0_net, map_sigma_net = load_res_fit_image_net()

# =============================================================================
# 
# =============================================================================


    import data
    import fit
    # data.create_data_NoisePredCheck()
    data_noise,data_noise_pars=data.load_data_NoisePredCheck()
    data_noise = np.mean(data_noise,axis=-2)
    
    fit_pixel_net(model,version,data_noise,data_noise_pars)
    res_s0,res_r2,res_sigma = fit.load_res_fit_pixel_net(version,data_noise_pars)
    
    
    # fit_pixel_sqexp(data_noise,data_noise_pars)
    # res_s0,res_r2,res_sigma = fit.load_res_fit_pixel_sqexp(data_noise_pars)
    
    # fit_pixel_ncexp(data_noise,data_noise_pars)
    # res_s0,res_r2,res_sigma = fit.load_res_fit_pixel_ncexp(data_noise_pars)
    
    res_sigma_mean = np.mean(res_sigma,axis=-1)
    
    id_s0 = 1
    
    import matplotlib.pyplot as plt

    plt.figure()
    noise_level = data_noise_pars['noise_level']
    noise_real = np.zeros([res_sigma.shape[0],res_sigma.shape[2]])
    for i in range(noise_real.shape[0]):
        for j in range(noise_real.shape[1]):
            noise_real[i,j] = noise_level[i]
    plt.imshow(noise_real,cmap='jet',interpolation='none',vmin=0,vmax=70)
    plt.title('Identity noise SD')
    
    plt.figure()
    res_stacks = res_stack(res_sigma,id_s0)
    plt.imshow(res_stacks, cmap='jet',interpolation='none',vmin=0,vmax=70)
    plt.title('Fitted noise SD')
    
    plt.figure()
    plt.imshow(res_sigma_mean[:,id_s0,:],cmap='jet',interpolation='none',vmin=0,vmax=70)
    plt.title('Fitted noise SD-mean')
    


