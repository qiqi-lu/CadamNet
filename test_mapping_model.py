#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:02:24 2020
Test all traditional mehtod and depp learning based model.
@author: luqiqi
"""
test_noise_level = 9
index_gpu        = 4

# proposed model infomation
model_sigma_prop      = 111
model_epoch_prop      = '200'
# model_name_prop, model_type_prop = 'DeepT2s',2 # the number of losses, 2 for denoising loss and mapping loss, 1 for one of them
model_name_prop, model_type_prop = 'DeepT2s(2)',2
# model_name_prop, model_type_prop = 'UNet',1
# model_name_prop, model_type_prop = 'UNet_full',1
# model_name_prop, model_type_prop = 'DeepT2s(1)',2

# compared model infomation
model_sigma_comp      = 111
model_epoch_comp      = 200
# model_name_comp, model_type_comp = 'DeepT2s',2
# model_name_comp, model_type_comp = 'DeepT2s(2)',2
model_name_comp, model_type_comp = 'UNet',1
# model_name_comp, model_type_comp = 'UNet_full',1
# model_name_comp, model_type_comp = 'DeepT2s(1)',2

# load test data
import os
import numpy as np
data_test_name = 'simulated_data_test_'+str(test_noise_level)+'.npy'
print('>> Load '+ data_test_name)
data_test = np.load(os.path.join('data_test_simulated',data_test_name),allow_pickle=True).item()
data_noise_free_test = data_test['noise free data']
data_noise_test = data_test['noise data']
map_r2s_test = data_test['r2s map'] # R2* ground truth
map_s0_test = data_test['s0 map']
num_study_test = data_noise_free_test.shape[0]

# set gpu configs
import skimage.metrics
import config
print(':: set gpu '+str(index_gpu))
config.config_gpu(index_gpu)

# load model
import tensorflow as tf
model_dir_prop = os.path.join('model',model_name_prop+'_sigma'+str(model_sigma_prop))
print('>> Load '+ model_dir_prop)
model_prop = tf.keras.models.load_model(os.path.join(model_dir_prop,'model_'+str(model_epoch_prop)+'.h5'),compile=False) # model proposed

model_dir_comp = os.path.join('model',model_name_comp+'_sigma'+str(model_sigma_comp))
print('>> Load '+ model_dir_comp)
model_comp = tf.keras.models.load_model(os.path.join(model_dir_comp,'model_'+str(model_epoch_comp)+'.h5'),compile=False) # model compared

# show model training results
from vis import show_loss
show_loss(model_name=model_name_prop,model_sigma=model_sigma_prop,model_type=model_type_prop)
show_loss(model_name=model_name_comp,model_sigma=model_sigma_comp,model_type=model_type_comp)

# test model mapping performance
from metrics import OtoO_mapping
result_mapping_prop  = OtoO_mapping(model_prop,model_type=model_type_prop,train_noise_level=model_sigma_prop,test_noise_level=test_noise_level)
result_mapping_comp  = OtoO_mapping(model_comp,model_type=model_type_comp,train_noise_level=model_sigma_comp,test_noise_level=test_noise_level)

# show mapping results on test data
from vis import show_test_mapping_results
# show_test_mapping_results(result_mapping_prop,test_noise_level)
show_test_mapping_results(result_mapping_prop,test_noise_level,study_id=19)


# i=1
# import matplotlib.pyplot as plt
# maps_pred_prop  = result_mapping_prop['mapp'][:,:,:,1]
# maps_pred_comp  = result_mapping_comp['mapp'][:,:,:,1]
# from data_generator import get_mask_test, get_mask_liver
# masks_body_test = get_mask_test() # body mask
# maps_pred_prop=maps_pred_prop*masks_body_test
# maps_pred_comp=maps_pred_comp*masks_body_test

# plt.figure(figsize=(20,10))
# plt.subplot(231),plt.axis('off')
# plt.imshow(maps_pred_prop[i],interpolation='none',cmap='jet',vmin=0,vmax=250),plt.title('prop'),plt.colorbar(fraction=0.022)
# plt.subplot(232),plt.axis('off')
# plt.imshow(maps_pred_comp[i],cmap='jet',interpolation='none',vmin=0,vmax=250),plt.title('comp'),plt.colorbar(fraction=0.022)
# plt.subplot(233),plt.axis('off')
# plt.imshow(map_r2s_test[i],cmap='jet',interpolation='none',vmin=0,vmax=250),plt.title('Ground Truth R2*'),plt.colorbar(fraction=0.022)
# up=75
# plt.subplot(234),plt.axis('off')
# plt.imshow(np.abs(map_r2s_test[i]-maps_pred_prop[i]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title('prop Difference'),plt.colorbar(fraction=0.022)
# plt.subplot(235),plt.axis('off')
# plt.imshow(np.abs(map_r2s_test[i]-maps_pred_comp[i]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title('comp Difference'),plt.colorbar(fraction=0.022)
# plt.subplot(236),plt.axis('off')
# plt.imshow(np.abs(maps_pred_prop[i]-maps_pred_comp[i]), cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title(' Difference'),plt.colorbar(fraction=0.022)
# # plt.tight_layout()
# plt.savefig(os.path.join('figure','test_map_'+str(i)))


from comparison import regression_plot
regression_plot(result_mapping_comp,result_mapping_prop,test_noise_level)
# from comparison import R_square_map
# tes = [0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0]
# maps = result_mapping_prop['mapp']
# # maps = result_mapping_comp['mapp']
# maps_v = np.reshape(maps,[-1,maps.shape[-1]])
# im_pred_v = np.reshape(maps_v[:,0],[-1,1])*np.exp(-1.0*np.reshape(maps_v[:,1],[-1,1])/1000.0*tes)
# im_pred = np.reshape(im_pred_v,[maps.shape[0],maps.shape[1],maps.shape[2],12])
# im_true = data_noise_free_test
# i=19
# rs_map = R_square_map(im_pred[i],im_true[i])


# check vivo data
from vis import show_vivo_mapping_result
show_vivo_mapping_result(model_prop,model_type=model_type_prop)

# calculate p value
from metrics import get_p_value

re_r2s_prop = result_mapping_prop['re_r2s']
re_r2s_comp = result_mapping_comp['re_r2s']
p_re = get_p_value(re_r2s_prop[0:-1],re_r2s_comp[0:-1],alt='greater')
# p_re = get_p_value(re_r2s_prop[0:-1],re_r2s_comp[0:-1],alt='less')

# print('>> RE (proposed): '+str(re_r2s_prop[19]))
# print('>> RE (compared): '+str(re_r2s_comp[19]))
print('>> p (RE) : '+str(p_re))

ssim_r2s_prop = result_mapping_prop['ssim_r2s']
ssim_r2s_comp = result_mapping_comp['ssim_r2s']
# p_ssim = get_p_value(ssim_r2s_prop[0:-2]*100,ssim_r2s_comp[0:-2]*100,alt='greater')
# p_ssim = get_p_value(ssim_r2s_prop[0:-2]*100,ssim_r2s_comp[0:-2]*100,alt='less')
p_ssim = get_p_value(ssim_r2s_prop[0:-1],ssim_r2s_comp[0:-1],alt='less')
# p_ssim = get_p_value(ssim_r2s_prop[0:-1],ssim_r2s_comp[0:-1],alt='greater')
# p_ssim = get_p_value(ssim_r2s_prop,ssim_r2s_comp,alt='greater')

# print('>> SSIM (proposed): '+str(ssim_r2s_prop[19]))
# print('>> SSIM (compared): '+str(ssim_r2s_comp[19]))
print('>> p (ssim) : '+str(p_ssim))
# print(ssim_r2s_prop[0:-2])
# print(ssim_r2s_comp)


# rmse_r2s_prop = result_mapping_prop['rmse_r2s']
# rmse_r2s_comp = result_mapping_comp['rmse_r2s']
# p_rmse = get_p_value(rmse_r2s_prop[0:-2],rmse_r2s_comp[0:-2])
# print('>> p(rmse) : '+str(p_rmse))




# # other methods
# from metrics import SSIM_multi,RMSE_multi,RE_multi,NRMSE_multi
# # load test data masks
# from data_generator import get_mask_test,get_mask_liver

# mask_liver_whole_test,mask_liver_parenchyma_test = get_mask_liver()
# masks = mask_liver_whole_test[-21:]
# masks = mask_liver_parenchyma_test[-21:]
# masks = get_mask_test() # body mask

# map_r2s_pcanr = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'_map_r2s_pcanr.npy')) 
# ssim_pcanr = SSIM_multi(map_r2s_pcanr,map_r2s_test,data_mask=masks)
# rmse_pcanr = RMSE_multi(map_r2s_pcanr,map_r2s_test,data_mask=masks)
# re_pcanr   = RE_multi(map_r2s_pcanr,map_r2s_test,data_mask=masks)
# print(':: SSIM (PCANR vs true): '+str(ssim_pcanr[-1])+'('+str(np.std(ssim_pcanr[0:-1]))+')')
# print(':: RMSE (PCANR vs true): '+str(rmse_pcanr[-1])+'('+str(np.std(rmse_pcanr[0:-1]))+')')
# print(':: RE   (PCANR vs true): '+str(re_pcanr[-1])+'('+str(np.std(re_pcanr[0:-1]))+')')
# print('>> SSIM (PCANR): '+str(ssim_pcanr[19]))
# print('>> RE (PCANR): '+str(re_pcanr[19]))

# # np.save(os.path.join('data_test_simulated','map_r2s_pcanr'),map_r2s_pcanr[19])

# map_r2s_m1ncm = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'_map_r2s_m1ncm.npy')) 
# ssim_m1ncm = SSIM_multi(map_r2s_m1ncm,map_r2s_test,data_mask=masks)
# rmse_m1ncm = RMSE_multi(map_r2s_m1ncm,map_r2s_test,data_mask=masks)
# re_m1ncm   = RE_multi  (map_r2s_m1ncm,map_r2s_test,data_mask=masks)
# nrmse_m1ncm   = NRMSE_multi  (map_r2s_m1ncm,map_r2s_test,data_mask=masks)
# print(':: SSIM (M1NCM vs true): '+str(ssim_m1ncm[-1])+'('+str(np.std(ssim_m1ncm[0:-1]))+')')
# print(':: RMSE (M1NCM vs true): '+str(rmse_m1ncm[-1])+'('+str(np.std(rmse_m1ncm[0:-1]))+')')
# print(':: RE   (M1NCM vs true): '+str(re_m1ncm[-1])+'('+str(np.std(re_m1ncm[0:-1]))+')')
# print(':: NRMSE(M1NCM vs true): '+str(nrmse_m1ncm[-1])+'('+str(np.std(nrmse_m1ncm[0:-1]))+')')
# print('>> SSIM (M1NCM): '+str(ssim_m1ncm[19]))
# print('>> RE (M1NCM): '+str(re_m1ncm[19]))

from comparison import vivo_test
vivo_test(model_prop,model_comp,model_type_comp=model_type_comp,model_type_prop=model_type_prop,filepath='data_test_clinical')

# data = np.load(os.path.join('data_clinical','clinical_data.npy'))
# if model_type_prop==2:
#     _,map_prop = model_prop.predict(data)
# else:
#     map_prop = model_prop.predict(data)

# if model_type_comp==2:
#     _,map_comp = model_comp.predict(data)
# else:
#     map_comp = model_comp.predict(data)

# import matplotlib.pyplot as plt
# i=18
# plt.figure(figsize=(30,10))
# plt.subplot(2,4,1),plt.axis('off'),plt.title('compared',loc='left')
# plt.imshow(map_comp[i+0,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,2),plt.axis('off')
# plt.imshow(map_comp[i+1,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,3),plt.axis('off')
# plt.imshow(map_comp[i+2,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,4),plt.axis('off')
# plt.imshow(map_comp[i+3,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,5),plt.axis('off'),plt.title('proposed',loc='left')
# plt.imshow(map_prop[i+0,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,6),plt.axis('off')
# plt.imshow(map_prop[i+1,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,7),plt.axis('off')
# plt.imshow(map_prop[i+2,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.subplot(2,4,8),plt.axis('off')
# plt.imshow(map_prop[i+3,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)
# plt.savefig(os.path.join('figure','maps_clinical'))
