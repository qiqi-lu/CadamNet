# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:13:45 2020

@author: Loo
"""

import argparse
import os, time, datetime
#import PIL.Image as Image
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='dataTest', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['simulated_data_test_5'], type=list, help='name of test dataset')
    parser.add_argument('--sigma_g', default=5, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('model','SeparableCNN_sigma5'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_150.h5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()
    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        imsave(path,np.clip(result,0,1))


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':
    import skimage
    import config
    
    config.config_gpu(0)
    
    args = parse_args()
    model = load_model(os.path.join(args.model_dir, args.model_name),compile=False)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
        
    for set_cur in os.listdir(args.set_dir):
        set_name, end = os.path.splitext(set_cur)
        if not os.path.exists(os.path.join(args.result_dir,set_name)):
            os.mkdir(os.path.join(args.result_dir,set_name))
            
        psnrs_ori = []
        ssims_ori = []
        
        psnrs = []
        ssims = []
        
        data_test = np.load(os.path.join('dataTest',set_cur),allow_pickle=True).item() 
        data_noise_free_test = data_test['noise free data']
        data_noise_test = data_test['noise data']
        map_r2s_test = data_test['r2s map']
        map_s0_test = data_test['s0 map']
        num_study_test = data_noise_free_test.shape[0]
        
        y = data_noise_test.astype(np.float32)
        
        start_time = time.time()
        xp = model.predict(y) # inference
        elapsed_time = time.time() - start_time
        print('%10s : %10s : %2.4f second'%(set_cur,1,elapsed_time))
        
        x = data_noise_free_test
        
        
        for i in range(0,num_study_test):
            psnr_x_ = skimage.metrics.peak_signal_noise_ratio(x[i], xp[i],data_range=1024)
            ssim_x_ = skimage.metrics.structural_similarity(x[i], xp[i],data_range=1024) 
            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)
            
            psnr_x_ori = skimage.metrics.peak_signal_noise_ratio(x[i], y[i],data_range=1024)
            ssim_x_ori = skimage.metrics.structural_similarity(x[i], y[i],data_range=1024) 
            psnrs_ori.append(psnr_x_ori)
            ssims_ori.append(ssim_x_ori)
    
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        
        psnr_avg_ori = np.mean(psnrs_ori)
        ssim_avg_ori = np.mean(ssims_ori)
        psnrs_ori.append(psnr_avg_ori)
        ssims_ori.append(ssim_avg_ori)
        
        # if args.save_result:
        #     save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,set_cur,'results.txt'))
            
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
        log('Datset: {0:10s} \n  PSNR_ori = {1:2.2f}dB, SSIM_ori = {2:1.4f}'.format(set_cur, psnr_avg_ori, ssim_avg_ori))

    import matplotlib.pyplot as plt
    from preprocessing import denoise_nlm_multi
    
    id_study = 13
    id_te = 11
    x_nlm = denoise_nlm_multi(y[id_study])
    
    plt.figure()
    vmax=100
    plt.subplot(221)
    plt.imshow(x[id_study,:,:,id_te],cmap='jet',vmin = 0,vmax=vmax,interpolation='none')
    plt.title('x')
    plt.subplot(222)
    plt.imshow(y[id_study,:,:,id_te],cmap='jet',vmin = 0,vmax=vmax,interpolation='none')
    plt.title('y')
    plt.subplot(223)
    plt.imshow(xp[id_study,:,:,id_te],cmap='jet',vmin = 0,vmax=vmax,interpolation='none')
    plt.title('x_pred')
    plt.subplot(224)
    plt.imshow(x_nlm[:,:,id_te],cmap='jet',vmin = 0,vmax=vmax,interpolation='none')
    plt.title('x_nlm')
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(x[id_study,:,:,id_te],cmap='jet',vmin = 0,vmax=100)
    plt.title('x')
    plt.subplot(222)
    plt.imshow(y[id_study,:,:,id_te]-x[id_study,:,:,id_te],cmap='jet',vmin = 0,vmax=100)
    plt.title('y')
    plt.subplot(223)
    plt.imshow(y[id_study,:,:,id_te]-xp[id_study,:,:,id_te],cmap='jet',vmin = 0,vmax=100)
    plt.title('x_pred')
    plt.subplot(224)
    plt.imshow(y[id_study,:,:,id_te]-x_nlm[:,:,id_te],cmap='jet',vmin = 0,vmax=100)
    plt.title('x_nlm')
    
    for i in range(0,12):
        plt.figure()
        plt.subplot(121),plt.imshow(y[id_study,:,:,i]-x[id_study,:,:,i],cmap='jet',vmax=100,vmin=0,interpolation='none'),plt.title(i)
        plt.subplot(122),plt.imshow(y[id_study,:,:,i]-xp[id_study,:,:,i],cmap='jet',vmax=100,vmin=0,interpolation='none'),plt.title(i)
    
# =============================================================================
#     test model on vivo data
# =============================================================================    
    from data_generator import read_dicom
    
    data_vivo,data_vivo_te = read_dicom(os.path.join('datadcm','study_test','*.dcm'))
    data_vivo = data_vivo[np.newaxis,...].astype('float32')
    # denoise vivo data using deep learning model
    data_vivo_p = model.predict(data_vivo)
    # denoise vivo data using nonlocal mean
    data_vivo_nlm = denoise_nlm_multi(data_vivo[0])
    
    id_te = 0
    
    plt.figure()
    plt.subplot(221)                  
    plt.imshow(data_vivo[0,:,:,id_te],cmap='jet',vmax=150,vmin=0,interpolation='none'),plt.title('vivo image')
    plt.subplot(222)                  
    plt.imshow(data_vivo_p[0,:,:,id_te],cmap='jet',vmax=150,vmin=0,interpolation='none'),plt.title('deep learning')
    plt.subplot(223)  
    plt.imshow(data_vivo_nlm[:,:,id_te],cmap='jet',vmax=150,vmin=0,interpolation='none'),plt.title('nlm') 
    
    plt.figure()
    plt.subplot(221)                  
    plt.imshow(data_vivo_p[0,:,:,id_te],cmap='jet',vmax=100,vmin=0,interpolation='none'),plt.title('deep learning')
    plt.subplot(222)                  
    plt.imshow(data_vivo_nlm[:,:,id_te],cmap='jet',vmax=100,vmin=0,interpolation='none'),plt.title('nlm') 
    plt.subplot(223)                  
    plt.imshow(data_vivo[0,:,:,id_te]-data_vivo_p[0,:,:,id_te],cmap='jet',vmax=100,vmin=0,interpolation='none'),plt.title('deep learning-noise')
    plt.subplot(224)                  
    plt.imshow(data_vivo[0,:,:,id_te]-data_vivo_nlm[:,:,id_te],cmap='jet',vmax=100,vmin=0,interpolation='none'),plt.title('nlm-noise') 
    
    
    for i in range(0,12):
        plt.figure()
        i=11
        plt.subplot(131)
        plt.imshow(data_vivo[0,:,:,i],cmap='jet',vmax=100,vmin=0,interpolation='none')
        plt.title(i)
        plt.subplot(133)
        plt.imshow(data_vivo[0,:,:,i]-data_vivo_p[0,:,:,i],cmap='jet',vmax=100,vmin=0,interpolation='none')
        plt.title(i)
        plt.subplot(132)
        plt.imshow(data_vivo_p[0,:,:,i],cmap='jet',vmax=100,vmin=0,interpolation='none')
        plt.title(i)
        
    
    data_dicom = data_vivo[0]
    data_dicom_p = data_vivo_p[0]
    # TEs_dicom = data_vivo_te
    
    pos_x = [115,18,35,82,96]
    pos_y = [3,5,12,44,56]
    size = [10,7,10,7,5]
    
    img = plt.figure()
    ax_img = img.add_subplot(1,1,1)
    i=0
    rect_bkg = plt.Rectangle((pos_x[i],pos_y[i]), size[i], size[i], fill=False,edgecolor='red',linewidth=1)
    mean_bkg = np.mean(np.mean(data_dicom[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    mean_bkg_p = np.mean(np.mean(data_dicom_p[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    i=1
    rect_lu  = plt.Rectangle((pos_x[i],pos_y[i]), size[i], size[i], fill=False,edgecolor='green',linewidth=1)
    mean_lu = np.mean(np.mean(data_dicom[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    mean_lu_p = np.mean(np.mean(data_dicom_p[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    i=2
    rect_liver = plt.Rectangle((pos_x[i],pos_y[i]), size[i], size[i], fill=False,edgecolor='yellow',linewidth=1)
    mean_liver = np.mean(np.mean(data_dicom[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    mean_liver_p = np.mean(np.mean(data_dicom_p[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    i=3
    rect_ms  = plt.Rectangle((pos_x[i],pos_y[i]), size[i], size[i], fill=False,edgecolor='cyan',linewidth=1)
    mean_ms = np.mean(np.mean(data_dicom[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    mean_ms_p = np.mean(np.mean(data_dicom_p[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    i=4
    rect_rd  = plt.Rectangle((pos_x[i],pos_y[i]), size[i], size[i], fill=False,edgecolor='blue',linewidth=1)
    mean_rd = np.mean(np.mean(data_dicom[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    mean_rd_p = np.mean(np.mean(data_dicom_p[pos_y[i]:pos_y[i]+size[i],pos_x[i]:pos_x[i]+size[i],:],axis=0),axis=0)
    
    
    ax_img.add_patch(rect_bkg)
    ax_img.add_patch(rect_lu)
    ax_img.add_patch(rect_liver)
    ax_img.add_patch(rect_ms)
    ax_img.add_patch(rect_rd)
    
    plt.imshow(data_dicom[:,:,0],cmap='gray',interpolation='none')
    plt.title('T2*w image TE[0]')
    plt.show()
    
    plt.figure()
    plt.plot(data_vivo_te,mean_bkg,'*--r',label='background')
    plt.plot(data_vivo_te,mean_lu,'*--g',label='fat')
    plt.plot(data_vivo_te,mean_liver,'*--y',label='liver')
    plt.plot(data_vivo_te,mean_ms,'*--c',label='spleen')
    plt.plot(data_vivo_te,mean_rd,'*--b',label='fat')
    plt.legend()
    
    plt.figure()
    plt.plot(data_vivo_te,mean_bkg_p,'*-r',label='background')
    plt.plot(data_vivo_te,mean_lu_p,'*-g',label='fat')
    plt.plot(data_vivo_te,mean_liver_p,'*-y',label='liver')
    plt.plot(data_vivo_te,mean_ms_p,'*-c',label='spleen')
    plt.plot(data_vivo_te,mean_rd_p,'*-b',label='fat')
    plt.legend()
    
    plt.figure()
    plt.plot(data_vivo_te,mean_bkg,'*--r',label='background')
    plt.plot(data_vivo_te,mean_liver,'*--y',label='liver')
    plt.plot(data_vivo_te,mean_ms,'*--c',label='spleen')
    
    plt.plot(data_vivo_te,mean_bkg_p,'*-r',label='background')
    plt.plot(data_vivo_te,mean_liver_p,'*-y',label='liver')
    plt.plot(data_vivo_te,mean_ms_p,'*-c',label='spleen')
    plt.legend()