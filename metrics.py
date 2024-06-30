#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:08:42 2020
Functions for model testing.
@author: luqiqi
"""
def save_result(result,path):
    import os
    import numpy as np
    import skimage
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        skimage.io.imsave(path,np.clip(result,0,1))
        
def OtoO_denoise(model,model_type=2,model_name='DeepT2s',train_noise_level=5,test_noise_level=5):
    """Test model using simulated data with different noise level.
    Test the power of the model for denoising.
    """
    import os
    import numpy as np
    import time
    import skimage
    
    result_dir = 'results_denoise'

    # create results save folder
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_path = os.path.join(result_dir,model_name+str(train_noise_level))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    # load test data
    data_test_name = 'simulated_data_test_'+str(test_noise_level)+'.npy'
    data_test = np.load(os.path.join('data_test_simulated',data_test_name),allow_pickle=True).item()
    
    data_test_noise_free = data_test['noise free data']
    data_test_noise = data_test['noise data']
    num_study_test = data_test_noise_free.shape[0]
    
    # noise data
    y = data_test_noise.astype(np.float32)
    # noise free data
    x = data_test_noise_free.astype(np.float32)
    
    # predict
    start_time = time.time()
    if model_type==2:
        xp,mapp = model.predict(y) # inference, xp predicted denoised images, mapp predicted maps
    elif model_type==1:
        # inference, xp predicted denoised images, mapp predicted maps
        xp = model.predict(y)
        mapp = 0
    elapsed_time = time.time() - start_time
    print('%10s: %2.4f second'%(data_test_name,elapsed_time))
    
    psnrs_ori = PSNR_multi(x,y)
    psnrs     = PSNR_multi(x,xp)
    ssims_ori = SSIM_multi(x,y)
    ssims     = SSIM_multi(x,xp)
    
    # save results
    save_result(np.hstack((psnrs_ori,psnrs,ssims_ori,ssims)),path=os.path.join(result_path,'resultsOn'+str(test_noise_level)+'.txt'))
        
    print(':: Dataset: {0:10s} PSNR = {1:2.4f}dB, SSIM = {2:1.4f} (original)'.format(data_test_name, psnrs_ori[-1], ssims_ori[-1]))
    print(':: Dataset: {0:10s} PSNR = {1:2.4f}dB, SSIM = {2:1.4f}'.format(data_test_name, psnrs[-1], ssims[-1]))
    
    result = {'xp':xp,
              'mapp':mapp,
              'psnr':psnrs,
              'ssim':ssims,
              'psnr_ori':psnrs_ori,
              'ssim_ori':ssims_ori,
              }
    
    return result

def OtoO_mapping(model,model_type=2,model_name='DeepT2s',train_noise_level=5,test_noise_level=5):
    import os
    import numpy as np
    import time
    import skimage
    
    result_dir = 'results_mapping'
    
    # create results save folder
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_path = os.path.join(result_dir,model_name+str(train_noise_level))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    # load test data
    data_test_name = 'simulated_data_test_'+str(test_noise_level)+'.npy'
    data_test = np.load(os.path.join('data_test_simulated',data_test_name),allow_pickle=True).item()
    data_test_noise = data_test['noise data']
    map_r2s_test = data_test['r2s map']
    map_s0_test = data_test['s0 map']
    
    # noise data
    y = data_test_noise.astype(np.float32)
    
    # predict
    start_time = time.time()
    if model_type==2:
        xp,mapp = model.predict(y) # inference, xp predicted denoised images, mapp predicted maps
    elif model_type==1:
        mapp = model.predict(y)
        xp = 0
    elapsed_time = time.time() - start_time
    print('>> %10s: %2.4f second'%(data_test_name,elapsed_time))
    
    # load test data masks
    # from data_generator import get_mask_test
    # masks_test = get_mask_test()

    from data_generator import get_mask_liver
    mask_liver_whole_test,mask_liver_parenchyma_test = get_mask_liver()
    masks_test = mask_liver_whole_test[-21:]

    map_r2s_true = map_r2s_test
    map_s0_true = map_s0_test
    map_r2s_pred = mapp[:,:,:,1]
    map_s0_pred = mapp[:,:,:,0]
    
    # calculate metrics
    print(':: Calculate the RMSE, RE and SSIM of noisy and predicted images in the body region ...')

    ssims_s0  = SSIM_multi(map_s0_true,map_s0_pred,data_mask=masks_test)
    # print('>> SSIM(R2s)')
    ssims_r2s = SSIM_multi(map_r2s_true,map_r2s_pred,data_mask=masks_test)

    rmses_s0  = RMSE_multi(map_s0_true,map_s0_pred,data_mask=masks_test)
    rmses_r2s = RMSE_multi(map_r2s_true,map_r2s_pred,data_mask=masks_test)

    re_s0     = RE_multi(map_s0_true,map_s0_pred,data_mask=masks_test)
    # print('>> RE(R2s)')
    re_r2s    = RE_multi(map_r2s_true,map_r2s_pred,data_mask=masks_test)

    # save results
    save_result(np.hstack((ssims_s0,ssims_r2s,rmses_s0,rmses_r2s)),path=os.path.join(result_path,'resultsOn'+str(test_noise_level)+'.txt'))
    print('>> Dataset: {0:3s} SSIM(R2s map) = {1:2.4f}, SSIM(S0 map) = {2:1.4f}'.format(str(test_noise_level), ssims_r2s[-1], ssims_s0[-1])+' (SD(R2*) '+str(np.std(ssims_r2s[0:-1]))+')')
    print('>> Dataset: {0:3s} RMSE(R2s map) = {1:2.4f}, RMSE(S0 map) = {2:1.4f}'.format(str(test_noise_level), rmses_r2s[-1], rmses_s0[-1])+' (SD(R2*) '+str(np.std(rmses_r2s[0:-1]))+')')
    print('>> Dataset: {0:3s} RE  (R2s map) = {1:2.4f}, RE  (S0 map) = {2:1.4f}'.format(str(test_noise_level), re_r2s[-1], re_s0[-1])+' (SD(R2*) '+str(np.std(re_r2s[0:-1]))+')')

        
    result = {'xp':xp,
              'mapp':mapp,
              'ssim_s0':ssims_s0,
              'ssim_r2s':ssims_r2s,
              'rmse_s0':rmses_s0,
              'rmse_r2s':rmses_r2s,
              're_s0':re_s0,
              're_r2s':re_r2s
              }

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,10))
    # plt.subplot(131),plt.axis('off')
    # plt.imshow(map_r2s_true[19],cmap='jet',interpolation='none',vmin=0,vmax=1000)
    # plt.subplot(132),plt.axis('off')
    # plt.imshow(map_r2s_pred[19],cmap='jet',interpolation='none',vmin=0,vmax=1000)
    # plt.subplot(133),plt.axis('off')
    # plt.imshow(np.abs(map_r2s_true[19]-map_r2s_pred[19]),cmap='jet',interpolation='none',vmin=0,vmax=250)
    # plt.savefig(os.path.join('figure','mapping_study19'))

    # mean_parenchyma = np.sqrt(np.sum((((map_r2s_true[19]-map_r2s_pred[19]))**2)*masks_test[19])/np.sum(masks_test[19]))
    # print('>> '+str(mean_parenchyma))

    return result

def PSNR_multi(data_true,data_pred):
    import skimage
    import numpy as np
    # calculate PSNR of each study
    psnrs = []
    num_study = data_true.shape[0]
    
    for i in range(num_study):
        psnr = skimage.metrics.peak_signal_noise_ratio(data_true[i], data_pred[i],data_range=2**10)
        psnrs.append(psnr)

    # mean metrics of all study
    psnr_avg = np.mean(psnrs)

    # print(psnrs[19])

    # the last value is the mean of all study in the test data
    psnrs.append(psnr_avg)

    return psnrs

def SSIM_multi(data_true,data_pred,data_mask=None):
    import skimage
    import numpy as np
    # calculate SSIM of each study
    ssims = []
    num_study = data_true.shape[0]
    
    for i in range(num_study):
        ssim,ssim_map = skimage.metrics.structural_similarity(data_true[i], data_pred[i],data_range=1024,full=True)
        if data_mask is not None:
            ssim = np.sum(ssim_map*data_mask[i])/np.sum(data_mask[i])
        ssims.append(ssim)
    
    # print(ssims[0])
    print(ssims)

    # mean metrics of all study
    ssim_avg = np.mean(ssims)

    # the last value is the mean of all study in the test data
    ssims.append(ssim_avg)

    return ssims

def RMSE_multi(data_true,data_pred,data_mask=None):
    import skimage
    import numpy as np
    # calculate SSIM of each study
    rmses = []
    num_study = data_true.shape[0]

    if data_mask is not None:
        for i in range(num_study):
            rmse = np.sqrt(np.sum(((data_true[i]-data_pred[i])**2)*data_mask[i])/np.sum(data_mask[i]))
            rmses.append(rmse)
    else:
        for i in range(num_study):
            # rmse = np.sqrt(np.sum((data_true[i]-data_pred[i])**2)/(data_pred[i].shape[0]*data_pred[i].shape[1]))
            rmse = np.sqrt(skimage.metrics.mean_squared_error(data_true[i], data_pred[i]))
            rmses.append(rmse)
    # print(rmses[19])
    # mean metrics of all study
    rmse_avg = np.mean(rmses)

    # the last value is the mean of all study in the test data
    rmses.append(rmse_avg)

    return rmses

def RE_multi(data_true,data_pred,data_mask=None):
    '''Relative Error (RE) metric 
    A means to quantitatively compare generated R2* maps with the ground truth.
    '''
    import numpy as np
    res = []
    num_study = data_true.shape[0]

    if data_mask is not None:
        data_true=data_true*data_mask
        data_pred=data_pred*data_mask

    for i in range(num_study):
        re = np.linalg.norm(data_true[i]-data_pred[i])/np.linalg.norm(data_true[i])
        # re_d = np.sum(abs(data_pred[i]-data_true[i]))/np.sum(data_mask[i])
        # re_t = np.sum(abs(data_true[i]))/np.sum(data_mask[i])
        # re   = re_d/re_t
        res.append(re)

    print('>> REs: '+str(res))
    
    # mean re of all study
    re_avg = np.mean(res)
    res.append(re_avg)
    return res

def NRMSE_multi(data_true,data_pred,data_mask=None):
    '''Normalized relative mean square error metric 
    A means to quantitatively compare generated R2* maps with the ground truth.
    '''
    import skimage
    import numpy as np
    nrmses = []
    num_study = data_true.shape[0]

    if data_mask is not None:
        data_true=data_true*data_mask
        data_pred=data_pred*data_mask

    for i in range(num_study):
        nrmse = skimage.metrics.normalized_root_mse(data_true[i],data_pred[i],normalization='euclidean')
        nrmses.append(nrmse)
    # mean re of all study
    nrmse_avg = np.mean(nrmses)
    nrmses.append(nrmse_avg)
    return nrmses

def get_p_value(A,B,alt='less'):
    from scipy import stats
    import numpy as np
    a = np.array(A)
    b = np.array(B)
    # t,p = stats.ttest_ind(a,b)
    # w,p = stats.wilcoxon(a,b)
    # w,p = stats.wilcoxon(a,b,alternative='greater')
    # w,p = stats.wilcoxon(a,b,correction=True, alternative='greater')
    w,p = stats.wilcoxon(a,b,correction=True, alternative=alt)
    return p

def get_sigma_g(data_study,num_coil=1):
    # calculate the mean of the bkg in [each study, each TE]
    import numpy as np
    h=5 # roi size
    region_bkg_mean1=np.zeros((data_study.shape[0],data_study.shape[-1]))
    region_bkg_mean2=np.zeros((data_study.shape[0],data_study.shape[-1]))
    for i in range(0,data_study.shape[0]): # each study
        for j in range(0,data_study.shape[-1]): # each TE time
            region_bkg_mean1[i,j] = np.mean(data_study[i,1:1+h,1:1+h,j]**2)
            region_bkg_mean2[i,j] = np.mean(data_study[i,1:1+h,122:122+h,j]**2)
    

    mean_bkg1 = np.mean(region_bkg_mean1,axis=-1)
    mean_bkg2 = np.mean(region_bkg_mean2,axis=-1)

    # calculate mean bkg signal in each study
    mean_bkg  = np.mean([mean_bkg1,mean_bkg2],axis=0)
    sigma_g   = np.sqrt(mean_bkg/(2*num_coil))
    return sigma_g