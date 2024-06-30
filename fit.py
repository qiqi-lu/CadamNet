#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:34:29 2020

@author: QiqiLu
"""
import numpy as np
import time
import tqdm
import os
from curvefitsimulation import calc_T2_NCEXP, calc_T2_offset, calc_T2_trunc, calc_T2_SQEXP, calc_T2_trunc_2sigma

def fit_pixel_trunc(data_test_x,data_test_pars):
    TEs_test = data_test_pars['TEs']
    S0 = data_test_pars['S0']
    name_data = data_test_pars['name_data']
    noise_level = data_test_pars['noise_level']
    
    res_s0_trunc  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    res_r2_trunc  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    res_tr_trunc  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    # total number of repeat
    num_total = data_test_x.shape[0]*data_test_x.shape[1]*data_test_x.shape[2]*data_test_x.shape[3]
    
    # Fitting every repeat
    print('Fitting pixels using truncation model ...')
    
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total,desc='Trunc')
    
    for index_noise_level in range(0,data_test_x.shape[0]):
        for index_s0 in range(0,data_test_x.shape[1]):
            sigma = S0[index_s0]/noise_level[index_noise_level]
            for index_t2 in range(0,data_test_x.shape[2]):
                for index_repeat in range(0,data_test_x.shape[3]):
                    pbar.update(1)
                    s0_trunc,T2_trunc,num_trunc = calc_T2_trunc_2sigma(TEs=TEs_test,signal_measured=data_test_x[index_noise_level,index_s0,index_t2,index_repeat,:],sigma=sigma)
                    res_s0_trunc[index_noise_level,index_s0,index_t2,index_repeat]  = s0_trunc
                    res_r2_trunc[index_noise_level,index_s0,index_t2,index_repeat]  = 1000.0/T2_trunc
                    res_tr_trunc[index_noise_level,index_s0,index_t2,index_repeat]  = num_trunc
                    
    pbar.close()
    
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resS0Truncation')),res_s0_trunc)
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resR2Truncation')),res_r2_trunc)
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resTrTruncation')),res_tr_trunc)
    
def fit_pixel_ncexp(data_test_x,data_test_pars):
    name_data = data_test_pars['name_data']
    TEs_test = data_test_pars['TEs']
    
    res_s0_ncexp  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    res_r2_ncexp  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    res_sg_ncexp  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    num_total = data_test_x.shape[0]*data_test_x.shape[1]*data_test_x.shape[2]*data_test_x.shape[3]
    
    print('Fitting pixels using NCEXP model ...')
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total,desc='NCEXP')
    for index_noise_level in range(0,data_test_x.shape[0]):
        for index_s0 in range(0,data_test_x.shape[1]):
            for index_t2 in range(0,data_test_x.shape[2]):
                for index_repeat in range(0,data_test_x.shape[3]):
                    pbar.update(1)
                    s0_ncexp,T2_ncexp,sg_ncexp = calc_T2_NCEXP(TEs=TEs_test,signal_measured=data_test_x[index_noise_level,index_s0,index_t2,index_repeat,:])
                    res_s0_ncexp[index_noise_level,index_s0,index_t2,index_repeat]  = s0_ncexp
                    res_r2_ncexp[index_noise_level,index_s0,index_t2,index_repeat]  = 1000.0/T2_ncexp
                    res_sg_ncexp[index_noise_level,index_s0,index_t2,index_repeat]  = sg_ncexp
    pbar.close()
    
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resS0NCEXP')),res_s0_ncexp)
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resR2NCEXP')),res_r2_ncexp)
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resSgNCEXP')),res_sg_ncexp)
    
def fit_pixel_sqexp(data_test_x,data_test_pars):
    name_data = data_test_pars['name_data']
    TEs_test = data_test_pars['TEs']
    res_s0_sqexp  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    res_r2_sqexp  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    res_sg_sqexp  = np.zeros([data_test_x.shape[0],data_test_x.shape[1],
                              data_test_x.shape[2],data_test_x.shape[3]])
    num_total = data_test_x.shape[0]*data_test_x.shape[1]*data_test_x.shape[2]*data_test_x.shape[3]
    print('Fitting pixels using SQEXP model ...')
    
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total,desc='SQEXP')
    for index_noise_level in range(0,data_test_x.shape[0]):
        for index_s0 in range(0,data_test_x.shape[1]):
            for index_t2 in range(0,data_test_x.shape[2]):
                for index_repeat in range(0,data_test_x.shape[3]):
                    pbar.update(1)
                    s0_sqexp,T2_sqexp,sg_sqexp = calc_T2_SQEXP(TEs=TEs_test,signal_measured_square =data_test_x[index_noise_level,index_s0,index_t2,index_repeat,:]**2)
                    res_s0_sqexp[index_noise_level,index_s0,index_t2,index_repeat]  = s0_sqexp
                    res_r2_sqexp[index_noise_level,index_s0,index_t2,index_repeat]  = 1000.0/T2_sqexp
                    res_sg_sqexp[index_noise_level,index_s0,index_t2,index_repeat]  = sg_sqexp
    pbar.close()
    
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resS0SQEXP')),res_s0_sqexp)
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resR2SQEXP')),res_r2_sqexp)
    np.save(os.path.join('resCheck','%s%s'%(name_data,'resSgSQEXP')),res_sg_sqexp)

def load_res_fit_pixel_sqexp(data_pars):
    name_data = data_pars['name_data']
    print('Load the saved results calculated by SQEXP model ...')
    res_s0_sqexp  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resS0SQEXP.npy')))
    res_r2_sqexp  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resR2SQEXP.npy')))
    res_sg_sqexp  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resSgSQEXP.npy')))
    return res_s0_sqexp,res_r2_sqexp,res_sg_sqexp

def load_res_fit_pixel_ncexp(data_pars):
    name_data = data_pars['name_data']
    print('Load the saved results calculated by NCEXP model ...')
    res_s0_ncexp  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resS0NCEXP.npy')))
    res_r2_ncexp  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resR2NCEXP.npy')))
    res_sg_ncexp  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resSgNCEXP.npy')))
    return res_s0_ncexp,res_r2_ncexp,res_sg_ncexp

def load_res_fit_pixel_trunc(data_pars):
    name_data = data_pars['name_data']
    print('Load the saved results calculated by Truncation model ...')
    res_s0_trunc  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resS0Truncation.npy')))
    res_r2_trunc  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resR2Truncation.npy')))
    res_tr_trunc  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resTrTruncation.npy')))
    return res_s0_trunc,res_r2_trunc,res_tr_trunc

def load_res_fit_pixel_offset(data_pars):
    name_data = data_pars['name_data']
    print('Load the saved results calculated by Offset model ...')
    res_s0_offset  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resS0Offset.npy')))
    res_r2_offset  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resR2Offset.npy')))
    res_of_offset  = np.load(os.path.join('resCheck','%s%s'%(name_data,'resOfOffset.npy')))
    return res_s0_offset,res_r2_offset,res_of_offset

def fit_pixel_net(model,version,data_test_x,data_test_pars):
    name_data = data_test_pars['name_data']
    TEs = data_test_pars['TEs']
    print('Fitting pixels using deep learning model ...')
    
    # total number of repeat
    num_total = data_test_x.shape[0]*data_test_x.shape[1]*data_test_x.shape[2]*data_test_x.shape[3]
    

    
    if version == 6:
        # reshape to predict shape
        data_test_x_mean_reshape = np.reshape(data_test_x,[num_total,-1])
        res = model.predict(data_test_x_mean_reshape)  
        res_r2_net = np.reshape(res[:,1],[data_test_x.shape[0],data_test_x.shape[1],
                                          data_test_x.shape[2],data_test_x.shape[3]])
        res_s0_net = np.reshape(res[:,0],[data_test_x.shape[0],data_test_x.shape[1],
                                          data_test_x.shape[2],data_test_x.shape[3]])
        res_sigma_net = np.reshape(res[:,2],[data_test_x.shape[0],data_test_x.shape[1],
                                             data_test_x.shape[2],data_test_x.shape[3]])
        
        # save the results
        np.save(os.path.join('resCheck','%s%s%d'%(name_data,'resS0Net',version)),res_s0_net)
        np.save(os.path.join('resCheck','%s%s%d'%(name_data,'resR2Net',version)),res_r2_net)
        np.save(os.path.join('resCheck','%s%s%d'%(name_data,'resSgNet',version)),res_sigma_net)
    else:
        # reshape to predict shape
        data_test_x_mean_reshape = np.reshape(data_test_x,[num_total,-1,1])
        
        # add TEs informatio to the train data
        data_test_x_mean_reshape = np.insert(data_test_x_mean_reshape,1,TEs,axis=2)
        if version == 4:
            data_test_x_mean_reshape = np.reshape(data_test_x_mean_reshape,[num_total,-1])

        res_t2_net = model.predict(data_test_x_mean_reshape)  
        res_r2_net = np.reshape(1000.0/res_t2_net,[data_test_x.shape[0],data_test_x.shape[1],
                                                   data_test_x.shape[2],data_test_x.shape[3]])
        # save the results
        np.save(os.path.join('resCheck','%s%s%d'%(name_data,'resR2Net',version)),res_r2_net)


    
def load_res_fit_pixel_net(version,data_pars):
    name_data = data_pars['name_data']
    print('Load the saved results calculated by last trained depp learning model ...')
    if version==6:
        res_s0_net = np.load(os.path.join('resCheck','%s%s%d%s'%(name_data,'resS0Net',version,'.npy')))
        res_r2_net = np.load(os.path.join('resCheck','%s%s%d%s'%(name_data,'resR2Net',version,'.npy')))
        res_sigma_net = np.load(os.path.join('resCheck','%s%s%d%s'%(name_data,'resSgNet',version,'.npy')))
    else:
        res_r2_net = np.load(os.path.join('resCheck','%s%s%d%s'%(name_data,'resR2Net',version,'.npy')))
        res_s0_net = 0
        res_sigma_net = 0
    return res_s0_net, res_r2_net, res_sigma_net

def fit_image_offset(data_dicom,TEs_dicom):
    num_row = data_dicom.shape[0]
    num_col =data_dicom.shape[1]
    num_total_dicom = num_row*num_col
    
    map_s0_offset = np.zeros([num_row,num_col])
    map_t2_offset = np.zeros([num_row,num_col])
    map_of_offset = np.zeros([num_row,num_col])
    
    print('Calculate T2 map using offset model...')
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total_dicom,desc='Offse')
    for index_row in range(0,num_row):
        for index_col in range(0,num_col):
            pbar.update(1)
            map_s0_offset[index_row,index_col],map_t2_offset[index_row,index_col],map_of_offset[index_row,index_col]=calc_T2_offset(TEs_dicom,data_dicom[index_row,index_col,:])
    pbar.close()
    np.save(os.path.join('resImage','mapS0Offset'),map_s0_offset) 
    np.save(os.path.join('resImage','mapT2Offset'),map_t2_offset) 
    np.save(os.path.join('resImage','mapOfOffset'),map_of_offset) 
    
def fit_image_trunc(data_dicom,TEs_dicom):
    num_row = data_dicom.shape[0]
    num_col =data_dicom.shape[1]
    num_total_dicom = num_row*num_col
    
    map_s0_trunc = np.zeros([num_row,num_col])
    map_t2_trunc = np.zeros([num_row,num_col])
    map_tr_trunc = np.zeros([num_row,num_col])
    
    print('Calculate T2 map using truncation model...')
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total_dicom,desc='Trunc')
    for index_row in range(0,num_row):
        for index_col in range(0,num_col):
            pbar.update(1)
            map_s0_trunc[index_row,index_col],map_t2_trunc[index_row,index_col],map_tr_trunc[index_row,index_col]=calc_T2_trunc(TEs_dicom,data_dicom[index_row,index_col,:])
    pbar.close()

    np.save(os.path.join('resImage','mapS0Truncation'),map_s0_trunc) 
    np.save(os.path.join('resImage','mapT2Truncation'),map_t2_trunc) 
    np.save(os.path.join('resImage','mapTrTruncation'),map_tr_trunc) 
    
def fit_image_ncexp(data_dicom,TEs_dicom):
    num_row = data_dicom.shape[0]
    num_col =data_dicom.shape[1]
    num_total_dicom = num_row*num_col
    
    map_s0_ncexp = np.zeros([num_row,num_col])
    map_t2_ncexp = np.zeros([num_row,num_col])
    map_sg_ncexp = np.zeros([num_row,num_col])
    
    print('Calculate T2 map using NCEXP model...')
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total_dicom,desc='NCEXP')
    for index_row in range(0,num_row):
        for index_col in range(0,num_col):
            pbar.update(1)
            map_s0_ncexp[index_row,index_col],map_t2_ncexp[index_row,index_col],map_sg_ncexp[index_row,index_col]=calc_T2_NCEXP(TEs_dicom,data_dicom[index_row,index_col,:])
    pbar.close()

    np.save(os.path.join('resImage','mapS0NCEXP'),map_s0_ncexp) 
    np.save(os.path.join('resImage','mapT2NCEXP'),map_t2_ncexp) 
    np.save(os.path.join('resImage','mapSgNCEXP'),map_sg_ncexp) 
    
def fit_image_sqexp(data_dicom,TEs_dicom):
    num_row = data_dicom.shape[0]
    num_col =data_dicom.shape[1]
    num_total_dicom = num_row*num_col
    
    map_s0_sqexp = np.zeros([num_row,num_col])
    map_t2_sqexp = np.zeros([num_row,num_col])
    map_sg_sqexp = np.zeros([num_row,num_col])
    
    print('Calculate T2 map using SQEXP model...')
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_total_dicom,desc='SQEXP')
    for index_row in range(0,num_row):
        for index_col in range(0,num_col):
            pbar.update(1)
            map_s0_sqexp[index_row,index_col],map_t2_sqexp[index_row,index_col],map_sg_sqexp[index_row,index_col]=calc_T2_SQEXP(TEs_dicom,data_dicom[index_row,index_col,:]**2)
    pbar.close()

    np.save(os.path.join('resImage','mapS0SQEXP'),map_s0_sqexp) 
    np.save(os.path.join('resImage','mapT2SQEXP'),map_t2_sqexp) 
    np.save(os.path.join('resImage','mapSgSQEXP'),map_sg_sqexp) 
    
def load_res_fit_image_offset():
    print('Load the saved map results calculated by offset model ...')
    map_t2 = np.load(os.path.join('resImage','mapT2Offset.npy'))
    map_s0 = np.load(os.path.join('resImage','mapS0Offset.npy'))
    map_of = np.load(os.path.join('resImage','mapOfOffset.npy'))

    return map_s0, map_t2, map_of

def load_res_fit_image_trunc():
    print('Load the saved map results calculated by truncation model ...')
    map_t2 = np.load(os.path.join('resImage','mapT2Truncation.npy'))
    map_s0 = np.load(os.path.join('resImage','mapS0Truncation.npy'))
    map_tr = np.load(os.path.join('resImage','mapTrTruncation.npy'))

    return map_s0, map_t2, map_tr

def load_res_fit_image_ncexp():
    print('Load the saved map results calculated by NCEXP model ...')
    map_t2 = np.load(os.path.join('resImage','mapT2NCEXP.npy'))
    map_s0 = np.load(os.path.join('resImage','mapS0NCEXP.npy'))
    map_sg = np.load(os.path.join('resImage','mapSgNCEXP.npy'))

    return map_s0, map_t2, map_sg

def load_res_fit_image_sqexp():
    print('Load the saved map results calculated by SQEXP model ...')
    map_t2 = np.load(os.path.join('resImage','mapT2SQEXP.npy'))
    map_s0 = np.load(os.path.join('resImage','mapS0SQEXP.npy'))
    map_sg = np.load(os.path.join('resImage','mapSgSQEXP.npy'))

    return map_s0, map_t2, map_sg
    
def fit_image_net(data_dicom,TEs_dicom,model,version):
    num_row = data_dicom.shape[0]
    num_col = data_dicom.shape[1]
    
    map_t2_net  = np.zeros([num_row,num_col])
    
    print('Calculate T2 map using deep learning model...')
    if version==6:
        data_dicom = np.reshape(data_dicom,[num_row*num_col,-1])
        maps = model.predict(data_dicom)
        map_t2_net = np.reshape(1000.0/maps[:,1],[num_row,num_col]) # r2* convert to t2*
        map_s0_net = np.reshape(maps[:,0],[num_row,num_col])
        map_sg_net = np.reshape(maps[:,2],[num_row,num_col])
    elif version==7:
        data_dicom = np.reshape(data_dicom,[num_row*num_col,-1,1])
        data_dicom = np.insert(data_dicom,1,TEs_dicom,axis=2)
        data_dicom = np.reshape(data_dicom, [num_row*num_col,-1])
        maps = model.predict(data_dicom)
        map_t2_net = np.reshape(1000.0/maps[:,1],[num_row,num_col]) # r2* convert to t2*
        map_s0_net = np.reshape(maps[:,0],[num_row,num_col])
        map_sg_net = np.reshape(maps[:,2],[num_row,num_col])
    else:
        data_dicom_reshape = np.reshape(data_dicom,[num_row,num_col,data_dicom.shape[-1],1])
        data_dicom_te = np.insert(data_dicom_reshape,1,TEs_dicom,axis=3)
        if version == 4:
            data_dicom_te = np.reshape(data_dicom_te,[num_row,num_col,data_dicom_te.shape[-2]*data_dicom_te.shape[-1]])
            data_dicom_te = np.reshape(data_dicom_te,[num_row*num_col,data_dicom_te.shape[-1]])
        else:
            data_dicom_te = np.reshape(data_dicom_te,[num_row*num_col,data_dicom_te.shape[-2],data_dicom_te.shape[-1]])
            
        maps = model.predict(data_dicom_te)
        map_t2_net = np.reshape(1000.0/maps[:,1],[num_row,num_col])
        map_s0_net = np.reshape(maps[:,0],[num_row,num_col])
        map_sg_net = np.reshape(maps[:,2],[num_row,num_col])

    np.save(os.path.join('resImage','mapT2Net'),map_t2_net)
    np.save(os.path.join('resImage','mapS0Net'),map_s0_net)
    np.save(os.path.join('resImage','mapSgNet'),map_sg_net)
    
def load_res_fit_image_net():
    print('Load the saved map results calculated by deep learning model ...')
    map_t2 = np.load(os.path.join('resImage','mapT2Net.npy'))
    map_s0 = np.load(os.path.join('resImage','mapS0Net.npy'))
    map_sg = np.load(os.path.join('resImage','mapSgNet.npy'))

    return map_s0, map_t2,map_sg