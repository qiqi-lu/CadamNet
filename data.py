#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:58:32 2020

@author: QiqiLu
"""
from tqdm import tqdm
from curvefitsimulation import fun_mono_exp
# from curvefitdeep import read_dicom
import numpy as np
import os
import SimpleITK as sitk
import sys
import time
import matplotlib.pyplot as plt
import glob
import pydicom

def create_refference_map():
    num_row = 64
    num_col = 18
    T2s = np.array([1000./35., 1000./200.,1000./300.,1000./400.,1000./500.,1000./600.,
           1000./800.,1000./1000.])
    num_region = T2s.shape[0]
    
    T2s_ref_map = np.ones([num_row, num_col])*T2s[0]
    
    for index_region in range(1,num_region):
        T2s_ref_map = np.concatenate((T2s_ref_map,np.ones([num_row, num_col]).dot(T2s[index_region])),axis = 1)
    plt.figure()
    plt.imshow(T2s_ref_map)
    return T2s_ref_map

def create_data_NoisePredCheck():
    pars = {'noise_level' : np.linspace(5,60,12),
                  'S0' : np.array([0.0,100.0,200.0]),
                  'R2' : np.array([35.0,50,100,200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0]),
                  'num_repeat': 100,
                  'num_pixel':1,
                  'TEs' : np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0]),
                  'num_channel':1,
                  'name_data': 'NoisePredCheck'}
    
    noise_level = pars['noise_level']
    S0 = pars['S0']  
    R2 = pars['R2']
    T2 = 1000.0/R2
    num_repeat  = pars['num_repeat']
    num_pixel   = pars['num_pixel']
    TEs = pars['TEs']
    # num_channel = pars['num_channel']
    
    print('Create ideal data and simulated noise data...')
    # create ideal data
    num_te = TEs.shape[0]
    num_T2 = T2.shape[0]
    num_noise_level = noise_level.shape[0]
    num_S0 = S0.shape[0]
    data_ideal = np.zeros([num_noise_level,num_S0,num_T2,num_repeat,num_pixel,num_te])
    data_noise = np.zeros([num_noise_level,num_S0,num_T2,num_repeat,num_pixel,num_te])
    data_t2 = np.ones([num_noise_level,num_S0,num_T2,num_repeat,num_pixel,1])
    
    time.sleep(1)
    pbar = tqdm(total=num_noise_level*num_S0*num_T2,desc='Process:')
    
    for index_noise_level in range(0,num_noise_level):
        sigma_g = noise_level[index_noise_level]
        for index_S0 in range(0,num_S0):            
            for index_T2 in range(0,num_T2):
                pbar.update(1)
                data_t2[index_noise_level,index_S0,index_T2,:,:,:] *= T2[index_T2]
                for index_pixel in range(0,num_pixel):
                    for index_repeat in range(0,num_repeat):
                        data_ideal[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,:]=fun_mono_exp(S0=S0[index_S0],T2=T2[index_T2],TEs=TEs)
                        # @YanqiuFeng, only 1 channel
                        x = sigma_g*np.random.standard_normal(num_te) + data_ideal[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,:]
                        y = sigma_g*np.random.standard_normal(num_te)
                        r = np.sqrt(x**2+y**2)
                        data_noise[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,:] = r
                        # @QiqiLu
                        # for index_TE in range(0,num_te):
                        #     s = data_ideal[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,index_TE]
                        #     n = np.random.noncentral_chisquare(num_channel*2,s**2/sigma_g**2,1)
                        #     n = np.sqrt(n)*sigma_g
                        #     data_noise[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,index_TE]=n
    pbar.close()
    print('')
    
    # save the simulated train data
    np.save(os.path.join('dataCheck','simulDataNoisePredCheck'),data_noise)
    np.save(os.path.join('dataCheck','simulDataNoisePredCheckPars'),pars)

def load_data_NoisePredCheck():
    print('Load the saved simulated noise data...')
    data_noise = np.load(os.path.join('dataCheck','simulDataNoisePredCheck.npy'))
    data_noise_pars = np.load(os.path.join('dataCheck','simulDataNoisePredCheckPars.npy'),allow_pickle=True).item()  
    return data_noise, data_noise_pars

def create_data_PixelModelTrain():
    # train data parameters
    # _,TEs = read_dicom(os.path.join('datadcm','study_moderate','*.dcm'))
    TEs = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    pars_train = {'noise_level' : np.linspace(15,60,10),
                  # 'noise_level' : np.linspace(1,100,10),
                  # 'noise_level' : np.array([15,30,60]),
                  # 'noise_level' : np.linspace(15,60,10),
                  
                  # 'S0': np.linspace(5,400,80), # option
                  # 'S0' : np.linspace(0,500,250),
                   'S0' : np.linspace(0.0,500.0,100),
                  # 'S0_train' : np.random.choice(np.linspace(5,400,800),80),
                  
                  # 'R2' : np.linspace(1,2500,500),
                  # 'R2' : np.linspace(5,1000,600), # option
                   'R2' : np.linspace(1,2000,1000),
                  # 'R2' : np.linspace(1,2000,200),
                  # 'R2' : np.random.choice(np.linspace(30,1000,2000),200),
                  
                  'num_repeat': 1,
                  'num_pixel':1,
                  'TEs' : TEs,
                  'num_channel':1,
                  'name_data': 'PixelModelTrain'}

    noise_level = pars_train['noise_level']
    S0 = pars_train['S0']  
    R2 = pars_train['R2']
    T2 = 1000.0/R2
    num_repeat  = pars_train['num_repeat']
    num_pixel   = pars_train['num_pixel']
    num_channel = pars_train['num_channel']

    print('create train data ...')
    # create simulated data with noise and t2 value for each data
    _, data_train_x, data_train_y = create_ideal_and_simu_data(noise_level,S0,T2,num_repeat,num_pixel,TEs,num_channel)
    
    # reshape the simulated train data into data shape fit for model training
    data_train_x = np.reshape(data_train_x,(-1,data_train_x.shape[-1],1))
    data_train_y = np.reshape(data_train_y,(-1,3))
    
    # save the simulated train data
    np.save(os.path.join('dataTrain','simulDataPixelModelTrainX'),data_train_x)
    np.save(os.path.join('dataTrain','simulDataPixelModelTrainY'),data_train_y)
    np.save(os.path.join('dataTrain','simulDataPixelModelTrainPars'),pars_train)
    
def load_data_PixelModelTrain():
    print('Load the saved train data...')
    data_train_x = np.load(os.path.join('dataTrain','simulDataPixelModelTrainX.npy'))
    data_train_y = np.load(os.path.join('dataTrain','simulDataPixelModelTrainY.npy'))
    data_train_pars = np.load(os.path.join('dataTrain', 'simulDataPixelModelTrainPars.npy'),allow_pickle=True).item()   
    return data_train_x, data_train_y, data_train_pars

def create_data_ResPredCheck():
    print('create the test simulation data ...')
    # _,TEs = read_dicom(os.path.join('datadcm','study_moderate','*.dcm'))
    TEs = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    pars_test = {'noise_level' : np.array([15,30,60]),
                  'S0': np.array([200]),
                  'R2' : np.linspace(30.0,1000.0,20),
                  'num_repeat': 100,
                  'num_pixel':100,
                  'TEs' : TEs,
                  'num_channel':1,
                  'name_data': 'ResPredCheck'}
    
    noise_level = pars_test['noise_level']
    S0 = pars_test['S0']  
    R2 = pars_test['R2']
    T2 = 1000.0/R2
    num_repeat  = pars_test['num_repeat']
    num_pixel   = pars_test['num_pixel']
    num_channel = pars_test['num_channel']
    
    # create ideal and simulated noise data
    _, data_test_x, data_test_y = create_ideal_and_simu_data(noise_level,S0,T2,num_repeat,num_pixel,TEs,num_channel)
    # save the simulated test data
    np.save(os.path.join('dataCheck','simulDataResPredCheckX'),data_test_x)
    np.save(os.path.join('dataCheck','simulDataResPredCheckY'),data_test_y)
    np.save(os.path.join('dataCheck','simulDataResPredCheckPars'),pars_test)
    
def load_data_ResPredCheck():
    print('Load the saved train data...')
    data_test_x = np.load(os.path.join('dataCheck','simulDataResPredCheckX.npy'))
    data_test_y = np.load(os.path.join('dataCheck','simulDataResPredCheckY.npy'))
    data_test_pars = np.load(os.path.join('dataCheck','simulDataResPredCheckPars.npy'),allow_pickle=True).item()
    return data_test_x,data_test_y, data_test_pars

def create_data_ImageModelTrain():
    import glob
    filepath = os.path.join('data_liver_same_te', 'Study*')

    name_folders = sorted(glob.glob(filepath),key=os.path.getmtime,reverse=True)
    num_study = np.array(name_folders).shape[0]
    
    data_study = np.zeros([num_study,64,128,12])
    
    # check te
    for id_study in range(num_study):
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(name_folders[id_study])
         
        if not series_ids:
            print("ERROR: given directory dose not a DICOM series.")
            sys.exit(1)
         
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(name_folders[id_study],series_ids[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        images = series_reader.Execute()
        
        images_resample = resample_sitk_image_series([data_study.shape[-2],data_study.shape[-3]], images)
        images_array = sitk.GetArrayFromImage(images_resample)
        for id_image in range(data_study.shape[-1]):
                data_study[id_study,:,:,id_image] = images_array[id_image,:,:]   
    
    np.save(os.path.join('dataTrain', 'vivoDataImageModelTrain'),data_study)
    return data_study

def load_data_ImageModelTrain():
    print('Load the saved vivo train data...')
    data_train = np.load(os.path.join('dataTrain','vivoDataImageModelTrain.npy'))
    return data_train
    
def resample_sitk_image_series(out_size,images_sitk):
    import numpy as np
    input_size = images_sitk.GetSize()
    input_spacing = images_sitk.GetSpacing()
    
    output_size = (out_size[0],out_size[1],input_size[2])
    
    output_spacing = np.array([0.,0.,0.]).astype('float64')
    output_spacing[0] = input_size[0]*input_spacing[0]/output_size[0]
    output_spacing[1] = input_size[1]*input_spacing[1]/output_size[1]
    output_spacing[2] = input_size[2]*input_spacing[2]/output_size[2]
    
    transform = sitk.Transform()
    transform.SetIdentity()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(images_sitk.GetOrigin())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputDirection(images_sitk.GetDirection())
    resampler.SetSize(output_size)
    images_sitk_resample = resampler.Execute(images_sitk)

    return images_sitk_resample

def create_ideal_and_simu_data(noise_level=np.array([15.0,30.0,60.0]),
                               S0=np.array([200.0]),
                               T2=1000.0/np.linspace(30.0,1000.0,20),
                               num_repeat=10,num_pixel=100,
                               TEs=np.array([0.93,2.27,3.61,4.95,6.29,7.63,8.97,
                                             10.4,11.8,13.2,14.6,16.0]),num_channel=4):
    """
    create ideal data and simulated data with noise, and the real te of each 
    signal

    Parameters
    ----------
    noise_level : TYPE, optional
        DESCRIPTION. The default is np.array([15.0,30.0,60.0]).
    S0 : TYPE, optional
        DESCRIPTION. The default is np.array([200.0]).
    T2 : TYPE, optional
        DESCRIPTION. The default is 1000.0/np.linspace(30.0,1000.0,20).
    num_repeat : TYPE, optional
        DESCRIPTION. The default is 10.
    num_pixel : TYPE, optional
        DESCRIPTION. The default is 100.
    TEs : TYPE, optional
        DESCRIPTION. The default is np.array([0.93,2.27,3.61,4.95,6.29,7.63,8.97,                                             10.4,11.8,13.2,14.6,16.0]).
    num_channel : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    data_ideal : tensor
        simulated data without noise.
    data_noise : tensor
        simulated data with noise.
    data_t2 : tensor
        real t2 corresponded to each simulated pixel signal.

    """
    print('Create ideal data and simulated noise data...')
    
    # create ideal data
    num_te = TEs.shape[0]
    num_T2 = T2.shape[0]
    num_noise_level = noise_level.shape[0]
    num_S0 = S0.shape[0]
    data_ideal = np.zeros([num_noise_level,num_S0,num_T2,num_repeat,num_pixel,num_te])
    data_noise = np.zeros([num_noise_level,num_S0,num_T2,num_repeat,num_pixel,num_te])
    data_pars = np.ones([num_noise_level,num_S0,num_T2,num_repeat,num_pixel,3])
    
    time.sleep(1)
    pbar = tqdm(total=num_noise_level*num_S0*num_T2,desc='Process:')
    
    for index_noise_level in range(0,num_noise_level):        
        for index_S0 in range(0,num_S0):            
            sigma_g = S0[index_S0]/noise_level[index_noise_level]
            # sigma_g = 200.0/noise_level[index_noise_level]
            for index_T2 in range(0,num_T2):
                pbar.update(1)
                data_pars[index_noise_level,index_S0,index_T2,:,:,0] *= S0[index_S0]
                data_pars[index_noise_level,index_S0,index_T2,:,:,1] *= 1000.0/T2[index_T2]
                data_pars[index_noise_level,index_S0,index_T2,:,:,2] *= sigma_g
                for index_pixel in range(0,num_pixel):
                    for index_repeat in range(0,num_repeat):
                        data_ideal[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,:]=fun_mono_exp(S0=S0[index_S0],T2=T2[index_T2],TEs=TEs)
                        # @YanqiuFeng, only 1 channel
                        x = sigma_g*np.random.standard_normal(num_te) + data_ideal[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,:]
                        y = sigma_g*np.random.standard_normal(num_te)
                        r = np.sqrt(x**2+y**2)
                        data_noise[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,:] = r
                        # @QiqiLu
                        # for index_TE in range(0,num_te):
                        #     s = data_ideal[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,index_TE]
                        #     n = np.random.noncentral_chisquare(num_channel*2,s**2/sigma_g**2,1)
                        #     n = np.sqrt(n)*sigma_g
                        #     data_noise[index_noise_level,index_S0,index_T2,index_repeat,index_pixel,index_TE]=n
    pbar.close()
    return data_ideal, data_noise, data_pars

def read_dicom(file_path):

    file_names = glob.glob(file_path)
    num_files = np.array(file_names).shape[0]
    image_multite = []
    tes = []
    
    for file,i in zip(file_names, range(0,num_files)):
        image_multite.append(pydicom.dcmread(file).pixel_array)
        tes.append(pydicom.dcmread(file).EchoTime)
        
    image_multite = np.array(image_multite).transpose((1,2,0))

    return image_multite, np.array(tes)

def load_txt_data(split=0.8):
    """read processed data in the txt from matlab
    # Returns
        train_x (array): shape = (num_pixel,num_image)
        train_y (array): shape = (num_pixel)
        test_x (array): shape = (num_pixel,num_image)
        test_y (array): shape = (num_pixel)
        TEs (array): shape = (num_image), the TE value of the images
    """
    # read txt data into array
    data = np.loadtxt(os.path.join('datatxt', 'roiMultiEchoData.txt'))
    TEs  = np.loadtxt(os.path.join('datatxt', 'roiMultiEchoTE.txt'))
    y_true = np.loadtxt("datatxt/roiT2AutoTrunc.txt")
    
    # split data into train data and test data
    split_boundary = int(data.shape[0] * split)  
    train_x = data[:split_boundary,:]
    test_x = data[split_boundary:,:]
    train_y = y_true[:split_boundary]
    test_y = y_true[split_boundary:]
    
    return train_x,train_y,test_x,test_y,TEs