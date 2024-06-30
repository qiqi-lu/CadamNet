#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:56:20 2020

@author: QiqiLu
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

from fit import load_res_fit_pixel_offset, load_res_fit_pixel_net
from fit import load_res_fit_pixel_trunc, load_res_fit_pixel_sqexp, load_res_fit_pixel_ncexp
from fit import load_res_fit_image_net
from fit import load_res_fit_image_offset, load_res_fit_image_trunc, load_res_fit_image_ncexp, load_res_fit_image_sqexp
from data import load_data_ResPredCheck, read_dicom
from curvefitsimulation import calc_T2_NCEXP, calc_T2_offset, calc_T2_trunc, calc_T2_SQEXP

def vis_mean_std_simu_pixel(index_noise_level,index_s0,version):
    
    # load ResPredCheck data parameters
    _,_,data_test_pars = load_data_ResPredCheck()
    # load the saved results
    _,res_r2_offset,_ = load_res_fit_pixel_offset(data_test_pars)
    _,res_r2_trunc,_ = load_res_fit_pixel_trunc(data_test_pars)
    _,res_r2_ncexp,_ = load_res_fit_pixel_ncexp(data_test_pars)
    _,res_r2_sqexp,_ = load_res_fit_pixel_sqexp(data_test_pars)
    _,res_r2_net,_=load_res_fit_pixel_net(version,data_test_pars)
    

    
    R2_test = data_test_pars['R2']
    noise_level_test = data_test_pars['noise_level']
    
    # Calculate the mean (accuracy) of R2* quantification from different model    
    mean_r2_offset = np.mean(res_r2_offset[index_noise_level,index_s0,:,:],axis=1)
    mean_r2_trunc  = np.mean(res_r2_trunc[index_noise_level,index_s0,:,:],axis=1)
    mean_r2_ncexp  = np.mean(res_r2_ncexp[index_noise_level,index_s0,:,:],axis=1)
    mean_r2_sqexp  = np.mean(res_r2_sqexp[index_noise_level,index_s0,:,:],axis=1)
    mean_r2_net    = np.mean(res_r2_net[index_noise_level,index_s0,:,:],axis=1)
    
    # Calculate the std (precision) of R2* quantification using different model    
    std_r2_offset = np.std(res_r2_offset[index_noise_level,index_s0,:,:],axis=1)
    std_r2_trunc  = np.std(res_r2_trunc[index_noise_level,index_s0,:,:], axis=1)
    std_r2_ncexp  = np.std(res_r2_ncexp[index_noise_level,index_s0,:,:], axis=1)
    std_r2_sqexp  = np.std(res_r2_sqexp[index_noise_level,index_s0,:,:], axis=1)
    std_r2_net    = np.std(res_r2_net[index_noise_level,index_s0,:,:],   axis=1)
    
    # plot the mean of the repeat result
    plt.figure()
    plt.plot(R2_test,R2_test,'-r',marker='o',label='Identity')
    plt.plot(R2_test,mean_r2_offset,'-b',marker='o',label='Offset')
    plt.plot(R2_test,mean_r2_trunc,'-g',marker='o',label='Truncation')
    plt.plot(R2_test,mean_r2_ncexp,'-k',marker='o',label='NCEXP')
    plt.plot(R2_test,mean_r2_sqexp,'-c',marker='o',label='SQEXP')
    plt.plot(R2_test,mean_r2_net,'-m',marker='o',label='DeepT2')
    plt.plot([0,1000],[1000,1000],'--k')
    plt.legend()
    plt.title("%s%d"%('SNR=',noise_level_test[index_noise_level]))
    plt.xlabel('True R2* (s-1)')
    plt.ylabel('Mean of Estimated R2* (s-1)')
    
    # plot the std of the repeat result 
    plt.figure()
    plt.plot(R2_test,std_r2_offset,'-b',marker='o',label='Offset')
    plt.plot(R2_test,std_r2_trunc,'-g',marker='o',label='Truncation')
    plt.plot(R2_test,std_r2_ncexp,'-k',marker='o',label='NCEXP')
    plt.plot(R2_test,std_r2_sqexp,'-c',marker='o',label='SQEXP')
    plt.plot(R2_test,std_r2_net,'-m',marker='o',label='DeepT2')
    plt.plot([0,1000],[100,100],'--k')
    plt.legend()
    plt.title("%s%d"%('SNR=',noise_level_test[index_noise_level]))
    plt.xlabel('True R2* (s-1)')
    plt.ylabel('SD of Estimated R2* (s-1)')
    
def vis_vivo_pixel_data(model,version):
    # =============================================================================
    # Test on clinial data 
    # Based on num_point mean points.
    # =============================================================================
    # load clinical data
    data_test_x_vivo = np.loadtxt(os.path.join('datatxt','roiMultiEchoData.txt'))
    TEs_test_vivo  = np.loadtxt(os.path.join('datatxt','roiMultiEchoTE.txt'))      
    
    # mean to 100 points
    # trunc test data into n*100
    num_point = 100
    num_each_point = int(data_test_x_vivo.shape[0]/num_point)
    data_test_x_vivo_trunc = data_test_x_vivo[:num_point*num_each_point,:]
    # mean
    data_test_x_vivo = np.zeros([num_point,data_test_x_vivo_trunc.shape[1]])
    for i in range(0,num_point):
        data_test_x_vivo[i,:] = np.mean(data_test_x_vivo_trunc[num_each_point*i:num_each_point*(i+1),:],axis=0)
        
    # predict result using troditional model
    res_t2_trunc_vivo  = np.zeros([data_test_x_vivo.shape[0]])
    res_t2_offset_vivo = np.zeros([data_test_x_vivo.shape[0]])
    res_t2_ncexp_vivo  = np.zeros([data_test_x_vivo.shape[0]])
    res_t2_sqexp_vivo  = np.zeros([data_test_x_vivo.shape[0]])
    
    for index_point in range(0,data_test_x_vivo.shape[0]):
        _, res_t2_trunc_vivo[index_point], _= calc_T2_trunc(TEs=TEs_test_vivo,signal_measured=data_test_x_vivo[index_point,:])
    
    for index_point in range(0,data_test_x_vivo.shape[0]):
        _, res_t2_offset_vivo[index_point], _= calc_T2_offset(TEs=TEs_test_vivo,signal_measured=data_test_x_vivo[index_point,:])
        
    for index_point in range(0,data_test_x_vivo.shape[0]):
        _, res_t2_ncexp_vivo[index_point], _= calc_T2_NCEXP(TEs=TEs_test_vivo,signal_measured=data_test_x_vivo[index_point,:])
        
    for index_point in range(0,data_test_x_vivo.shape[0]):
        _, res_t2_sqexp_vivo[index_point], _= calc_T2_SQEXP(TEs=TEs_test_vivo,signal_measured_square=data_test_x_vivo[index_point,:]**2)
    
    if version==6:
        data_test_x_vivo = data_test_x_vivo
        res_t2_net_vivo = model.predict(data_test_x_vivo)
        res_t2_net_vivo = 1000.0/res_t2_net_vivo[:,1]
    else:
        # predict result using deep learning model
        data_test_x_vivo = np.reshape(data_test_x_vivo,[data_test_x_vivo.shape[0],
                                                        data_test_x_vivo.shape[1],1])
        # add TE infromation to the test data
        data_test_x_vivo = np.insert(data_test_x_vivo,1,TEs_test_vivo,axis=2)
        
        # the version 4 deep learning model need a 1D input, reshape the input data shape
        if version == 4:
            data_test_x_vivo = np.reshape(data_test_x_vivo,[data_test_x_vivo.shape[0],-1])
    
        res_t2_net_vivo = model.predict(data_test_x_vivo)
    
    # =============================================================================
    # Plot the comparation bettween traditional model and deep learning model result
    # =============================================================================
    # load the saved traditional model's results
    #res_t2_trunc_vivo  = np.loadtxt('datatxt\\roiT2AutoTrunc.txt')
    #res_t2_offset_vivo = np.loadtxt('datatxt\\roiT2Offset.txt')
    #res_t2_ncexp_vivo  = np.loadtxt('datatxt\\roiNCEXPRes.txt')
    
    # convert the t2 results into r2 results
    res_r2_trunc_vivo  = 1000.0/res_t2_trunc_vivo
    res_r2_offset_vivo = 1000.0/res_t2_offset_vivo
    res_r2_ncexp_vivo = 1000.0/res_t2_ncexp_vivo
    res_r2_sqexp_vivo = 1000.0/res_t2_sqexp_vivo
    res_r2_net_vivo   = 1000.0/res_t2_net_vivo
    
    # x axis range
    x_range = [0,1000]
    
    # Trunction model vs deep learning model
    plt.figure()
    # linear regression
    z_trunc_net = np.polyfit(res_r2_trunc_vivo,res_r2_net_vivo,1)
    p_trunc_net = np.poly1d(np.squeeze(z_trunc_net))
    y = p_trunc_net(x_range)
    
    plt.plot(x_range,y,'--k',label='$y=%fx%f$'%(z_trunc_net[0],z_trunc_net[1]))
    plt.plot(res_r2_trunc_vivo,res_r2_net_vivo,'*')
    plt.plot(x_range,x_range,'--r',label='Identity')
    plt.xlabel('Auto Truncation R2* (s-1)')
    plt.ylabel('DeepT2 R2* (s-1)')
    plt.legend()
    
    # Offset model vs deep learning model
    plt.figure()
    # linear function fitting
    z_offset_net = np.polyfit(res_r2_offset_vivo,res_r2_net_vivo,1)
    p_offset_net = np.poly1d(np.squeeze(z_offset_net))
    y = p_offset_net(x_range)
    
    plt.plot(x_range,y,'--k',label='$y=%fx%f$'%(z_offset_net[0],z_offset_net[1]))
    plt.plot(res_r2_offset_vivo,res_r2_net_vivo,'*')
    plt.plot(x_range,x_range,'--r',label='Identity')
    plt.xlabel('Offset R2* (s-1)')
    plt.ylabel('DeepT2 R2* (s-1)')
    plt.legend()
    
    # NCEXP model vs deep learning model
    plt.figure()
    # linear function fitting
    z_ncexp_net = np.polyfit(res_r2_ncexp_vivo,res_r2_net_vivo,1)
    p_ncexp_net = np.poly1d(np.squeeze(z_ncexp_net))
    y = p_ncexp_net(x_range)
    
    plt.plot(x_range,y,'--k',label='$y=%fx%f$'%(z_ncexp_net[0],z_ncexp_net[1]))
    plt.plot(res_r2_ncexp_vivo,res_r2_net_vivo,'*')
    plt.plot(x_range,x_range,'--r',label='Identity')
    plt.xlabel('NCEXP R2* (s-1)')
    plt.ylabel('DeepT2 R2* (s-1)')
    plt.legend()
    
    # SQEXP model vs deep learning model
    plt.figure()
    # linear function fitting
    z_sqexp_net = np.polyfit(res_r2_sqexp_vivo,res_r2_net_vivo,1)
    p_sqexp_net = np.poly1d(np.squeeze(z_sqexp_net))
    y = p_sqexp_net(x_range)
    
    plt.plot(x_range,y,'--k',label='$y=%fx%f$'%(z_sqexp_net[0],z_sqexp_net[1]))
    plt.plot(res_r2_sqexp_vivo,res_r2_net_vivo,'*')
    plt.plot(x_range,x_range,'--r',label='Identity')
    plt.xlabel('SQEXP R2* (s-1)')
    plt.ylabel('DeepT2 R2* (s-1)')
    plt.legend()
    
    # Construct a Tukey/Bland-Altman Mean Difference Plot
    plt.figure()
    sm.graphics.mean_diff_plot(res_r2_offset_vivo,np.squeeze(res_r2_net_vivo))
    plt.ylim([-150,150])
    plt.xlim([200,600])
    plt.xlabel('Mean R2* Fitted by Offset & DeepT2 (s-1)')
    plt.ylabel('Difference: Offset-DeepT2')
    
    plt.figure()
    sm.graphics.mean_diff_plot(res_r2_trunc_vivo,np.squeeze(res_r2_net_vivo))
    plt.ylim([-150,150])
    plt.xlim([200,600])
    plt.xlabel('Mean R2* Fitted by Truncation & DeepT2 (s-1)')
    plt.ylabel('Difference: Truncation-DeepT2')
    
    plt.figure()
    sm.graphics.mean_diff_plot(res_r2_ncexp_vivo,np.squeeze(res_r2_net_vivo))
    plt.ylim([-150,150])
    plt.xlim([200,600])
    plt.xlabel('Mean R2* Fitted by NCEXP & DeepT2 (s-1)')
    plt.ylabel('Difference: NCEXP-DeepT2')
    
    plt.figure()
    sm.graphics.mean_diff_plot(res_r2_sqexp_vivo,np.squeeze(res_r2_net_vivo))
    plt.ylim([-150,150])
    plt.xlim([200,600])
    plt.xlabel('Mean R2* Fitted by SQEXP & DeepT2 (s-1)')
    plt.ylabel('Difference: SQEXP-DeepT2')
    
    # Estimated R2* for num_point points using traditional model and deep learning
    # model plotted in ascending order of R2* fitted by the NCEXP model.
    plt.figure()
    # index = np.argsort(res_r2_ncexp_vivo)
    index = np.argsort(res_r2_sqexp_vivo)
    #index = np.flipud(index)
    
    res_r2_vivo_trunc_sort = res_r2_trunc_vivo[index]
    res_r2_vivo_offset_sort = res_r2_offset_vivo[index]
    res_r2_vivo_ncexp_sort = res_r2_ncexp_vivo[index]
    res_r2_vivo_sqexp_sort = res_r2_sqexp_vivo[index]
    res_r2_vivo_net_sort = res_r2_net_vivo[index]
    
    plt.plot(res_r2_vivo_trunc_sort,'*g',label='Truncation')
    plt.plot(res_r2_vivo_offset_sort,'ob',label='Offset')
    plt.plot(res_r2_vivo_ncexp_sort,'ok',label='NCEXP')
    plt.plot(res_r2_vivo_sqexp_sort,'oc',label='SQEXP')
    plt.plot(res_r2_vivo_net_sort,'om',label='DeepT2')
    
    plt.xlabel('Point')
    plt.ylabel('Estimated R2* (s-1)')
    plt.legend()
    
    # Estimated R2* for num_point points using traditional model and deep learning
    # model plotted in random order
    plt.figure()
    plt.plot(res_r2_trunc_vivo,'-g',label='Truncation')
    plt.plot(res_r2_offset_vivo,'-b',label='Offset')
    plt.plot(res_r2_ncexp_vivo,'-k',label='NCEXP')
    plt.plot(res_r2_sqexp_vivo,'-c',label='SQEXP')
    plt.plot(res_r2_net_vivo,'-m',label='DeepT2')
    plt.xlabel('Pixel')
    plt.ylabel('Estimated R2* (s-1)')
    plt.legend()

def vis_map_r2s():
    _,map_t2_offset,_ = load_res_fit_image_offset()
    _,map_t2_trunc,_ = load_res_fit_image_trunc()
    _,map_t2_ncexp,_ = load_res_fit_image_ncexp()
    _,map_t2_sqexp,_ = load_res_fit_image_sqexp()
    map_s0_net, map_t2_net, map_sigma_net = load_res_fit_image_net()
    
    map_r2_offset = 1000.0/map_t2_offset
    map_r2_trunc  = 1000.0/map_t2_trunc
    map_r2_ncexp  = 1000.0/map_t2_ncexp
    map_r2_sqexp  = 1000.0/map_t2_sqexp
    map_r2_net = 1000.0/map_t2_net
    map_r2_net[np.isinf(map_r2_net)]=0
    map_r2_net[np.isnan(map_r2_net)]=0
    
    # show T2 map
    plt.figure()
    plt.imshow(map_r2_offset,vmin=0,vmax=2500.0, cmap = 'jet',interpolation='none')
    plt.title('Offset model')
    plt.colorbar(fraction=0.022)
    
    plt.figure()
    plt.imshow(map_r2_trunc,vmin=0,vmax=2500.0,cmap='jet',interpolation='none')
    plt.colorbar(fraction=0.022)
    plt.title('Truncation model')
    
    plt.figure()
    plt.imshow(map_r2_ncexp,vmin=0,vmax=2500.0,cmap='jet',interpolation='none')
    plt.title('NCEXP model')
    plt.colorbar(fraction=0.022)
    
    plt.figure()
    plt.imshow(map_r2_sqexp,vmin=0,vmax=2500.0,cmap='jet',interpolation='none')
    plt.title('SQEXP model')
    plt.colorbar(fraction=0.022)
    
    plt.figure()
    plt.imshow(map_r2_net,  vmin=0,vmax=2500.0, cmap='jet',interpolation='none')
    plt.title('Deep Learning model')
    plt.colorbar(fraction=0.022)
    
    plt.figure()
    plt.imshow(abs(map_r2_net-map_r2_sqexp),vmin=0.0,vmax=100.0,cmap='jet')
    plt.title('Deep Learning model - NCEXP model')
    plt.colorbar(fraction=0.022)
    
    plt.figure()
    plt.subplot(221)
    plt.title('S0 map')
    plt.imshow(map_s0_net,cmap='jet',interpolation='none',vmin=0.0,vmax=1000.0)
    plt.colorbar(fraction=0.022)
    plt.subplot(222)
    plt.title('R2* map')
    plt.imshow(map_r2_net,cmap='jet',interpolation='none',vmin=0.0,vmax=2500.0)
    plt.colorbar(fraction=0.022)
    plt.subplot(223)
    plt.title('Sigma map')
    plt.imshow(map_sigma_net,cmap='jet',interpolation='none',vmin=0.0,vmax=25.0)
    plt.colorbar(fraction=0.022)
    
    filepath_dicom = os.path.join('datadcm','study_severe','*.dcm')
    data_dicom, TEs_dicom = read_dicom(file_path=filepath_dicom)
    data_dicom = np.array(data_dicom)
    data_dicom = data_dicom.astype('float32')
    
    plt.subplot(224)
    plt.title('T2*w image(TE0)')
    plt.imshow(data_dicom[:,:,0],interpolation='none',cmap='gray')
    plt.colorbar(fraction=0.022)

def vis_map_sigma():
    import fit
    _,_,map_sg_ncexp = fit.load_res_fit_image_ncexp()
    _,_,map_sg_sqexp = fit.load_res_fit_image_sqexp()
    _,_,map_sg_net   = fit.load_res_fit_image_net()
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(map_sg_ncexp,cmap='jet',interpolation='none',vmin=0,vmax=25)
    plt.title('NCEXP Sigma map')
    plt.subplot(222)
    plt.imshow(map_sg_sqexp,cmap='jet',interpolation='none',vmin=0,vmax=25)
    plt.title('SQEXP Sigma map')
    plt.subplot(223)
    plt.imshow(map_sg_net,cmap='jet',interpolation='none',vmin=0,vmax=25)
    plt.title('Deep learning Sigma map')
    