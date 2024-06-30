#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:02:24 2020
Test denoising model.
@author: luqiqi
"""
model_sigma      = 13
test_noise_level = 13
model_epoch      = 300
# model_name,model_type = 'SeparableCNN',1
model_name,model_type = 'DeepT2s',2
# model_name,model_type = 'DeepT2s(2)',2

print(':: Calculate the PSNR and SSIM of noise images and denoised images...')
# set gpu config
import skimage.metrics
import config
config.config_gpu(0)

# load model
import tensorflow as tf
import os
model_dir = os.path.join('model',model_name+'_sigma'+str(model_sigma))
model = tf.keras.models.load_model(os.path.join(model_dir,'model_'+str(model_epoch)+'.h5'),compile=False)

# # show model training results
# from vis import show_loss
# show_loss(model_name,model_sigma,model_type=model_type)

# test model denoising performance
from metrics import OtoO_denoise
train_noise_level= model_sigma
result = OtoO_denoise(model,model_type=model_type,model_name=model_name,train_noise_level=train_noise_level,test_noise_level=test_noise_level)

# # show test denoising result
# from vis import show_test_denoising_result
# show_test_denoising_result(result,test_noise_level,study_id=19)

# # show vivo data test results
# from vis import show_vivo_denoising_result
# show_vivo_denoising_result(model,model_type=model_type)
import numpy as np
data_test_name = 'simulated_data_test_'+str(test_noise_level)+'.npy'
data_test = np.load(os.path.join('data_test_simulated',data_test_name),allow_pickle=True).item()
data_noise_test = data_test['noise data']
data_noise_free_test = data_test['noise free data']
num_study_test  = data_noise_test.shape[0]
xp  = result['xp']

from data_generator import get_mask_test,get_mask_liver

mask_liver_whole_test,mask_liver_parenchyma_test = get_mask_liver()
masks = mask_liver_whole_test[-21:]
# masks = mask_liver_parenchyma_test[-21:]
# masks = get_mask_test() # body mask

from metrics import SSIM_multi,PSNR_multi
te_id=11
ssim_prop = SSIM_multi(data_noise_free_test[:,:,:,te_id],xp[:,:,:,te_id],data_mask=masks)
ssim_noise = SSIM_multi(data_noise_free_test[:,:,:,te_id],data_noise_test[:,:,:,te_id],data_mask=masks)

print(':: SSIM (xp vs true): '+str(ssim_prop[-1])+'('+str(np.std(ssim_prop[0:-2]))+')')
print(':: SSIM (noise vs true): '+str(ssim_noise[-1])+'('+str(np.std(ssim_noise[0:-2]))+')')

psnr_prop = PSNR_multi(data_noise_free_test[:,:,:,te_id],xp[:,:,:,te_id])
psnr_noise = PSNR_multi(data_noise_free_test[:,:,:,te_id],data_noise_test[:,:,:,te_id])

print(':: PSNR (xp vs true): '+str(psnr_prop[-1])+'('+str(np.std(psnr_prop[0:-2]))+')')
print(':: PSNR (noise vs true): '+str(psnr_noise[-1])+'('+str(np.std(psnr_noise[0:-2]))+')')

ssims_prop = []
psnrs_prop = []
ssims_noise = []
psnrs_noise = []

ssims_prop_err = []
psnrs_prop_err = []
ssims_noise_err = []
psnrs_noise_err = []

for te_id in range(12):
    ssim_prop = SSIM_multi(data_noise_free_test[:,:,:,te_id],xp[:,:,:,te_id],data_mask=masks)
    ssim_noise = SSIM_multi(data_noise_free_test[:,:,:,te_id],data_noise_test[:,:,:,te_id],data_mask=masks)
    psnr_prop = PSNR_multi(data_noise_free_test[:,:,:,te_id],xp[:,:,:,te_id])
    psnr_noise = PSNR_multi(data_noise_free_test[:,:,:,te_id],data_noise_test[:,:,:,te_id])

    ssims_prop.append(ssim_prop[-1])
    psnrs_prop.append(psnr_prop[-1])
    ssims_noise.append(ssim_noise[-1])
    psnrs_noise.append(psnr_noise[-1])

    ssims_prop_err.append(np.std(ssim_prop[0:-2]))
    psnrs_prop_err.append(np.std(psnr_prop[0:-2]))
    ssims_noise_err.append(np.std(ssim_noise[0:-2]))
    psnrs_noise_err.append(np.std(psnr_noise[0:-2]))

TEs=[0.93,2.27,3.61,4.95,6.29,7.63,8.97,10.4,11.8,13.2,14.6,16.0]

import matplotlib.pyplot as plt
plt.figure(figsize=(15,7.5))
plt.subplot(121)
plt.plot(TEs,ssims_prop,'o-r',label='Denoising-Net')
plt.errorbar(TEs,ssims_prop,fmt='',color='k',yerr=ssims_prop_err)
plt.plot(TEs,ssims_noise,'o-k',label='Noise')
plt.errorbar(TEs,ssims_noise,fmt='',color='k',yerr=ssims_noise_err)

plt.legend()
plt.subplot(122)
plt.plot(TEs,psnrs_prop,'o-r',label='Denoising-Net')
plt.errorbar(TEs,psnrs_prop,fmt='',color='k',yerr=psnrs_prop_err)
plt.plot(TEs,psnrs_noise,'o-k',label='Noise')
plt.errorbar(TEs,psnrs_noise,fmt='',color='k',yerr=psnrs_noise_err)

plt.legend()
plt.savefig(os.path.join('figure','tevsdenoise.png'))


