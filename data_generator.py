#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:22:34 2020

@author: luqiqi
"""
import cv2.cv2 as cv2
import skimage
import glob
import pydicom
import numpy as np

import os
import SimpleITK as sitk
import sys
import time
import tqdm
from curvefitsimulation import calc_T2_trunc


from skimage import filters
from skimage import segmentation
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from preprocessing import denoise_nlm_multi


patch_size, stride = 32, 8
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

def read_dicom(file_path):
    """ 
    Read all .dcm file in the folder into a array and their TE value
    """
    file_names = glob.glob(file_path)
    file_names = sorted(file_names,key=os.path.getmtime,reverse=True)
    num_files = np.array(file_names).shape[0]
    image_multite = []
    tes = []
    # read data and info
    for file,i in zip(file_names, range(0,num_files)):
        image_multite.append(pydicom.dcmread(file).pixel_array)
        tes.append(pydicom.dcmread(file).EchoTime)
        
    image_multite = np.array(image_multite).transpose((1,2,0))
    return image_multite,np.array(tes)

def get_data_vivo(file_dir):
    """
    Read data of all study
    """
    print(':: Get vivo data...')
    filepath = os.path.join(file_dir, 'Study2*')
    # get all study name
    folder_names = sorted(glob.glob(filepath),key=os.path.getmtime,reverse=True)
    num_study = np.array(folder_names).shape[0]
    data_vivo = []
    for i in range(num_study):
        data,_=read_dicom(os.path.join(folder_names[i],'*.dcm'))
        data_vivo.append(data)
    return data_vivo

def clearup_clinical_data(file_dir):
    """
    Clean up all raw clinical data and reshape to the same size
    """
    print(':: Clear up clinical data...')
    filepath = os.path.join(file_dir, 'Study2*')
    # get all study name
    name_folders = sorted(glob.glob(filepath),key=os.path.getmtime,reverse=True)
    num_study  = np.array(name_folders).shape[0]
    data_study = np.zeros([num_study,64,128,12])

    # read data and info, and reshape to same size
    for id_study in range(num_study):
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(name_folders[id_study])
         
        if not series_ids:
            print("ERROR: given directory dose not a DICOM series.")
            sys.exit(1)
         
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(name_folders[id_study],series_ids[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        images = series_reader.Execute()
        # resahpe image into same size
        images_resample = resample_sitk_image_series([data_study.shape[-2],data_study.shape[-3]], images)
        images_array = sitk.GetArrayFromImage(images_resample)
        for id_image in range(data_study.shape[-1]):
            data_study[id_study,:,:,id_image] = images_array[id_image,:,:]   
    
    np.save(os.path.join('data_clinical', 'clinical_data'),data_study)
    return data_study

def prep(data_study):
    """Preprocessing of the clinical data
    """
    print('Nonlocal mean...')
    num_study = data_study.shape[0]
    
    # nonlocal mean to all t2s weighted images
    data_study_denoised = []
    for i in range(0,num_study):
        temp = denoise_nlm_multi(data_study[i,:,:,:])
        data_study_denoised.append(temp)
        
    return np.array(data_study_denoised)

def get_clinical_data():
    print('Get the clinical data array...')
    data_train = np.load(os.path.join('data_clinical','clinical_data.npy'))
    return data_train
    
def resample_sitk_image_series(out_size,images_sitk):
    """resample image to specific size
    """
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

def gen_map(image_multite,tes,filtering=False):
    print('Calculate s0 and R2s maps...')
    num_study,h,w,c = image_multite.shape
    num_pixel = num_study*h*w
    
    # filtering befor fitting
    if filtering:
        # filtering using fixed parameters
        image_multite = prep(image_multite)
        np.save(os.path.join('data_clinical', 'clinical_data_denoised_nlm_fix'),image_multite)
        
        # filtering using background noise sigma based parameters @ChangqingWang # may be too much
        # image_multite = np.load(os.path.join('data_clinical','clinical_data_denoised_nlm.npy'))

    data_pixel = np.reshape(image_multite, [-1,c])
    
    s0 = np.zeros(num_pixel)
    t2s = np.zeros(num_pixel)
    
    # use conventional model to fit (Truncation model)
    time.sleep(1.0)
    pbar = tqdm.tqdm(total=num_pixel,desc='Trunc')
    for id_pixel in range(0,num_pixel):
        pbar.update(1)
        s0[id_pixel],t2s[id_pixel],_=calc_T2_trunc(tes,data_pixel[id_pixel])
    pbar.close()
    
    r2s = 1000.0/t2s
    map_s0  = np.reshape(s0,  [num_study,h,w])
    map_r2s = np.reshape(r2s, [num_study,h,w])
    
    # save results
    if filtering:
        np.save(os.path.join('data_clinical', 'clinical_data_map_s0_nlm_fix_autotrunc'),map_s0)
        np.save(os.path.join('data_clinical', 'clinical_data_map_r2s_nlm_fix_autotrunc'),map_r2s)
    else:
        np.save(os.path.join('data_clinical', 'clinical_data_map_s0'),map_s0)
        np.save(os.path.join('data_clinical', 'clinical_data_map_r2s'),map_r2s)

def get_map(filtering=False):
    print('Get clinical data map array...')
    if filtering:
        map_s0  = np.load(os.path.join('data_clinical','clinical_data_map_s0_nlm_fix_autotrunc.npy'))
        map_r2s = np.load(os.path.join('data_clinical','clinical_data_map_r2s_nlm_fix_autotrunc.npy'))
    else:
        map_s0  = np.load(os.path.join('data_clinical','clinical_data_map_s0.npy'))
        map_r2s = np.load(os.path.join('data_clinical','clinical_data_map_r2s.npy'))
    return map_s0, map_r2s

def gen_bkg_mask(image_multite):
    print('Gen background mask of each study...')
    num_study,_,_,_ = image_multite.shape
    masks = []
    # use conventional method to make mask
    for id_study in range(0,num_study):
        img = image_multite[id_study,:,:,0]
        # img=img**2
        elevation_map = skimage.filters.sobel(img)
        markers = np.zeros_like(img)
        markers[img<15]=1
        markers[img>30]=2
        mask = skimage.segmentation.watershed(elevation_map,markers)
        # mask = skimage.segmentation.random_walker(img, markers)
        mask = ndi.binary_fill_holes(mask-1)
        masks.append(mask)
    masks = np.array(masks)
    
    np.save(os.path.join('data_clinical', 'clinical_data_bkg_mask'),masks)
    
def gen_bkg_mask_manual():
    import re
    import struct
    print('clean up all manual mask...')
    mask_manual_path = os.path.join('data_clinical','clinical_data_mask_body','Mask*.raw')
    mask_names = glob.glob(mask_manual_path)
    mask_names.sort(key=lambda x:int(re.findall('Mask(\d+).raw',x)[0]))
    masks=[]
    for i in range(len(mask_names)):
        size = os.path.getsize(mask_names[i])
        with open(mask_names[i],mode='rb') as f:
            mask = f.read(size)
            mask = struct.unpack('B'*size, mask)
            mask = np.reshape(np.array(mask)/255, [64,128])
            masks.append(mask)
    masks = np.array(masks)
    np.save(os.path.join('data_clinical', 'clinical_data_bkg_mask_manual'),masks)
            
    
    
def get_bkg_mask(manual=False):
    print('Get clinical data background mask...')
    if manual:
        masks = np.load(os.path.join('data_clinical','clinical_data_bkg_mask_manual.npy'))
    else:      
        masks = np.load(os.path.join('data_clinical','clinical_data_bkg_mask.npy'))
    return masks


def gen_data_noise_free(s0_map,r2s_map,tes):
    print('Gen simulated noise-free data...')
    num_study,h,w = s0_map.shape
    te = np.reshape(tes,[1,-1])
    data_noise_free = []
    for id_study in range(0,num_study):
        s0 = np.reshape(s0_map[id_study,:,:],[-1,1])
        r2s = np.reshape(r2s_map[id_study,:,:],[-1,1])
        # use exp model to recreate weighted image
        noise_free = s0*np.exp(-1.0*r2s*te/1000.0)
        noise_free = np.reshape(noise_free,[h,w,-1])
        data_noise_free.append(noise_free)
    data_noise_free = np.array(data_noise_free)
    np.save(os.path.join('data_simulated', 'simulated_data_noise_free'),data_noise_free)
    
def get_data_noise_free():
    print('Get simulated noise-free data...')
    data = np.load(os.path.join('data_simulated','simulated_data_noise_free.npy'))
    return data

def gen_data_noise(data_noise_free,sigma_g=5):
    print('Gen noise data (sigma = '+str(sigma_g)+') from noise-free data...')
    num_study,h,w,c=data_noise_free.shape

    if sigma_g == 0:
        print('mixed sigma noise.')
        data_noise = []
        num_p = h*w*c
        for i in range(num_study):
            sg = np.random.choice([1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0])
            images_nf = np.reshape(data_noise_free[i], [-1,1])
            r = np.reshape(sg*np.random.standard_normal(num_p),[-1,1]) + images_nf
            i = np.reshape(sg*np.random.standard_normal(num_p),[-1,1])
            s = np.sqrt(r**2+i**2)
            images_n = np.reshape(s,[h,w,c])
            data_noise.append(images_n)
        data_noise = np.array(data_noise)
    else:    
        num_p = num_study*h*w*c
        data_noise_free = np.reshape(data_noise_free,[-1,1])
        r = np.reshape(sigma_g*np.random.standard_normal(num_p),[-1,1]) + data_noise_free
        i = np.reshape(sigma_g*np.random.standard_normal(num_p),[-1,1])
        s = np.sqrt(r**2+i**2)
        data_noise = np.reshape(s,[num_study,h,w,c])
    
    np.save(os.path.join('data_simulated', '%s%d'%('simulated_data_noise_',sigma_g)),data_noise)
    
def get_data_noise(sigma_g=5):
    print('Get simulated noise data...')
    data = np.load(os.path.join('data_simulated','%s%d%s'%('simulated_data_noise_',sigma_g,'.npy')))
    return data

def gen_data_train(data_noise_free,data_noise,map_s0,map_r2s,sigma_g):
    print('Connect the data for train..')
    num_study,h,w,c=data_noise_free.shape
#    map_s0 = np.reshape(map_s0,[-1,1])
#    map_r2s = np.reshape(map_r2s,[-1,1])
    data_train = {'num of study':num_study,'h':h,'w':w,'c':c,'num of map':2,'sigma_g':sigma_g,
                  'noise free data':data_noise_free,
                  'noise data':data_noise,
                  's0 map':map_s0,
                  'r2s map':map_r2s}
    np.save(os.path.join('data_simulated', '%s%d'%('simulated_data_train_',sigma_g)),data_train)
      
def get_data_train(sigma_g):
    print('>> Get training data ('+'sigma='+str(sigma_g)+')...')
    data = np.load(os.path.join('data_simulated','%s%d%s'%('simulated_data_train_',sigma_g,'.npy')),allow_pickle=True).item()  
    return data

def gen_data_test(data_train):
    print('Gen test data...')
    data_noise_free = data_train['noise free data']
    data_noise = data_train['noise data']
    map_r2s = data_train['r2s map']
    map_s0 = data_train['s0 map']
    sigma_g = data_train['sigma_g']
    
    num_study,h,w,c=data_noise_free.shape
    # num_train = int(num_study*0.8)
    num_train = 100
    
    data_test = {'num of study':num_study,'h':h,'w':w,'c':c,'num of map':2,'sigma_g':sigma_g,
                  'noise free data':data_noise_free[num_train:],
                  'noise data':data_noise[num_train:],
                  's0 map':map_s0[num_train:],
                  'r2s map':map_r2s[num_train:]}
    
    num_study = num_study-num_train
    save_dir = 'dataTest'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join('data_test_simulated', '%s%d'%('simulated_data_test_',sigma_g)),data_test)
    
def gen_data_test_noise(data_train):
    print('Gen test data (noise)...')
    data_noise = data_train['noise data']
    sigma_g = data_train['sigma_g']
    
    num_study,h,w,c=data_noise.shape
    num_train = 100
    
    data_test_noise = data_noise[num_train:]
    
    num_study = num_study-num_train
    np.save(os.path.join('data_test_simulated', '%s%d'%('simulated_data_test_noise_',sigma_g)),data_test_noise)
    
def get_data_test(sigma_g):
    print('Get test data...')
    data = np.load(os.path.join('data_test_simulated','%s%d%s'%('simulated_data_test_',sigma_g,'.npy')),allow_pickle=True).item()  
    return data

def gen_mask_test(masks,train_size=100):
    masks_test = masks[train_size:]
    np.save(os.path.join('data_test_simulated','simulated_data_test_mask'),masks_test)
    
def get_mask_test():
    masks = np.load(os.path.join('data_test_simulated', 'simulated_data_test_mask.npy'))
    return masks

def cleanup_mask_liver_manual(folderpath_in,folderpath_out=None):
    import re
    import struct
    print('>> Clean up all manual liver and liver parenchyma mask...')
    # mask_dir =  os.path.join('data_clinical','clinical_data_mask_liver_parenchyma')
    mask_dir = folderpath_in
    mask_names = glob.glob(os.path.join(mask_dir,'Mask*.raw'))
    mask_names.sort(key=lambda x:int(re.findall('Mask(\d+).raw',x)[0]))
    masks=[]
    for i in range(len(mask_names)):
        size = os.path.getsize(mask_names[i])
        with open(mask_names[i],mode='rb') as f:
            mask = f.read(size)
            mask = struct.unpack('B'*size, mask)
            mask = np.reshape(np.array(mask)/255, [64,128])
            masks.append(mask)
    masks = np.array(masks)

    # split all masks into whole liver masks and liver parenchyma masks
    mask_liver_whole = masks[1::2] 
    mask_liver_parenchyma = masks[::2]

    if folderpath_out is None:
        folderpath_out = folderpath_in
    
    # show mask information
    print('>> (INFO) mask size: ',str(np.shape(masks)))

    np.save(os.path.join(folderpath_out, 'mask_liver_whole_manual'),mask_liver_whole)
    np.save(os.path.join(folderpath_out, 'mask_liver_parenchyma_manual'),mask_liver_parenchyma)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,7.5))
    plt.subplot(1,2,1),plt.imshow(mask_liver_whole[-1],cmap='jet'),plt.title('Liver')
    plt.subplot(1,2,2),plt.imshow(mask_liver_parenchyma[-1],cmap='jet'),plt.title('Liver parenchyma')
    plt.savefig(os.path.join(folderpath_out,'mask_example.png'))


def get_mask_liver(filepath='data_clinical'):
    masks_liver_whole = np.load(os.path.join(filepath, 'mask_liver_whole_manual.npy'))
    masks_liver_parenchyma = np.load(os.path.join(filepath, 'mask_liver_parenchyma_manual.npy'))
    return masks_liver_whole,masks_liver_parenchyma

def get_mask_test_clinical():
    import re
    import struct
    print('>> Clean up all manual liver parenchyma mask (clinical test data)...')
    mask_dir =  os.path.join('data_test_clinical','mask')
    hic_level = ['normal','mild','moderate','severe']

    masks_body  = []
    masks_liver = []
    masks_parenchyma = []

    for i in range(4):
        mask_body = get_raw_data(os.path.join(mask_dir,hic_level[i]+'_body.raw'))
        mask_liver = get_raw_data(os.path.join(mask_dir,hic_level[i]+'_liver.raw')) 
        mask_parenchyma = get_raw_data(os.path.join(mask_dir,hic_level[i]+'_liver_parenchyma.raw'))

        masks_body.append(mask_body)
        masks_liver.append(mask_liver)
        masks_parenchyma.append(mask_parenchyma)

    masks_body = np.array(masks_body)
    masks_liver = np.array(masks_liver)
    masks_parenchyma = np.array(masks_parenchyma)
    return masks_body,masks_liver,masks_parenchyma

def get_raw_data(file_name):
    import struct
    size = os.path.getsize(file_name)
    with open(file_name,mode='rb') as f:
        mask = f.read(size)
        mask = struct.unpack('B'*size, mask)
        mask = np.reshape(np.array(mask)/255, [64,128])
    return mask

def data_aug(img, mode=0):
    # aug data size
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(image_multite):
    """generate patches for single study with multiple weighted images
    """
    h,w,c=image_multite.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        image_multite_scaled = cv2.resize(image_multite, (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = image_multite_scaled[i:i+patch_size, j:j+patch_size,:]
                # patches.append(x)        
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    patches.append(x_aug)
                    
    return np.array(patches)

def gen_data_train_patches(data_train):
    data_noise_free = data_train['noise free data']
    data_noise = data_train['noise data']
    map_r2s = data_train['r2s map']
    map_s0 = data_train['s0 map']
    sigma_g = data_train['sigma_g']
    
    num_study,h,w,c=data_noise_free.shape
    
    # num_study = int(num_study*0.8)
    num_study = 100
    
    data_noise_free_patches = []
    data_noise_patches = []
    map_s0_patches = []
    map_r2s_patches = []
    
    for id_study in range(0,num_study):
        for s in scales:
            h_scaled, w_scaled = int(h*s),int(w*s)
            img_nf_scaled  = cv2.resize(data_noise_free[id_study,:,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            img_n_scaled   = cv2.resize(data_noise[id_study,:,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            map_s0_scaled  = cv2.resize(map_s0[id_study,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            map_r2s_scaled = cv2.resize(map_r2s[id_study,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            
            # extract patches
            for i in range(0, h_scaled-patch_size+1, stride):
                for j in range(0, w_scaled-patch_size+1, stride):
                    x  = img_nf_scaled[i:i+patch_size, j:j+patch_size,:]
                    xn = img_n_scaled[i:i+patch_size, j:j+patch_size,:]
                    ms = map_s0_scaled[i:i+patch_size, j:j+patch_size]
                    mr = map_r2s_scaled[i:i+patch_size, j:j+patch_size]
                    #patches.append(x)        
                    # data aug
                    for k in range(0, aug_times):
                        mode=np.random.randint(0,8)
                        x_aug = data_aug(x, mode=mode)
                        xn_aug = data_aug(xn, mode=mode)
                        ms_aug = data_aug(ms, mode=mode)
                        mr_aug = data_aug(mr, mode=mode)
                        
                        data_noise_free_patches.append(x_aug)
                        data_noise_patches.append(xn_aug)
                        map_s0_patches.append(ms_aug)
                        map_r2s_patches.append(mr_aug)
                        
    data_train_patches = {'noise free data':np.array(data_noise_free_patches),
                          'noise data':np.array(data_noise_patches),
                          's0 map':np.array(map_s0_patches),
                          'r2s map':np.array(map_r2s_patches),
                          'sigma_g':sigma_g}
    
    np.save(os.path.join('data_train_simulated', '%s%d'%('simulated_data_train_patches_',sigma_g)),data_train_patches)
    
def get_data_train_patches(sigma_g):
    # print('Get training data patches...')
    data = np.load(os.path.join('data_train_simulated','%s%d%s'%('simulated_data_train_patches_',sigma_g,'.npy')),allow_pickle=True).item()  
    return data

def gen_clinical_data_patches(data_study,sigma_g):
    print('Add noise to clinical data...')
    num_study,h,w,c=data_study.shape
    num_p = num_study*h*w*c
    
    data = np.reshape(data_study,[-1,1])
    r = np.reshape(sigma_g*np.random.standard_normal(num_p),[-1,1]) + data
    i = np.reshape(sigma_g*np.random.standard_normal(num_p),[-1,1])
    s = np.sqrt(r**2+i**2)
    data_study_noise = np.reshape(s,[num_study,h,w,c])
    
    print('Gen patches...')
    data_patches = []
    data_noise_patches = []
    # set train data size
    num_study = 100
    for id_study in range(0,num_study):
        for s in scales:
            h_scaled, w_scaled = int(h*s),int(w*s)
            img_scaled   = cv2.resize(data_study[id_study,:,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            img_n_scaled = cv2.resize(data_study_noise[id_study,:,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches
            for i in range(0, h_scaled-patch_size+1, stride):
                for j in range(0, w_scaled-patch_size+1, stride):
                    x = img_scaled[i:i+patch_size, j:j+patch_size,:]
                    xn = img_n_scaled[i:i+patch_size, j:j+patch_size,:]
                    #patches.append(x)        
                    # data aug
                    for k in range(0, aug_times):
                        mode=np.random.randint(0,8)
                        x_aug = data_aug(x, mode=mode)
                        xn_aug = data_aug(xn, mode=mode)
                        
                        data_patches.append(x_aug)
                        data_noise_patches.append(xn_aug)
    data_train_patches = {'noise free data':np.array(data_patches),
                          'noise data':np.array(data_noise_patches),
                          'sigma_g':sigma_g}
    np.save(os.path.join('data_train_clinical', '%s%d'%('clinical_data_train_patches_',sigma_g)),data_train_patches)
    
def get_clinical_data_patches(sigma_g):
    print('Get clinical data patches...')
    data = np.load(os.path.join('data_train_clinical','%s%d%s'%('clinical_data_train_patches_',sigma_g,'.npy')),allow_pickle=True).item()  
    return data
    
def datagenerator(sigma_g):
    data = get_data_train_patches(sigma_g)
    data = np.array(data, dtype='float32')
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    return data

def train_datagen(epoch_num=5,batch_size=128,sigma_g=5,du=False):
    while(True):
        n_count = 0
        if n_count == 0:
            #print(n_count)
            data = get_data_train_patches(sigma_g)
            data_noise_free = data['noise free data']
            data_noise = data['noise data']
            
            data_noise_free = np.array(data_noise_free, dtype='float32')
            data_noise = np.array(data_noise, dtype='float32')
            
            discard_n = len(data_noise_free)-len(data_noise_free)//batch_size*batch_size;
            data_noise_free = np.delete(data_noise_free,range(discard_n),axis = 0)
            data_noise = np.delete(data_noise,range(discard_n),axis = 0)
            
            indices = list(range(data_noise_free.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = data_noise[indices[i:i+batch_size]]
                batch_y = data_noise_free[indices[i:i+batch_size]]
                if du:
                    yield batch_x, [batch_y,batch_y]
                else:
                    yield batch_x, batch_y

def clearup_vivo_data():
    # load vivo test data
    file_dir = 'data_test_clinical'
    d_normal,_ = read_dicom(os.path.join(file_dir,'*normal','*.dcm'))
    d_mild,_ = read_dicom(os.path.join(file_dir,'*mild','*.dcm'))
    d_moderate,_ = read_dicom(os.path.join(file_dir,'*moderate','*.dcm'))
    d_severe,_ = read_dicom(os.path.join(file_dir,'*severe','*.dcm'))
    data_vivo = np.array([d_normal,d_mild,d_moderate,d_severe])
    # save into .npy file
    np.save(os.path.join('data_test_clinical','clinical_data_test'),data_vivo)

def get_vivo_data():
    print('Get vivo data...')
    data = np.load(os.path.join('data_test_clinical','clinical_data_test.npy'))
    return data

def gen_data_train_patches_rn(split=0.8):
    # get moise free data
    sigma_g = 111
    data_noise_free = get_data_noise_free()
    map_s0,map_r2s = get_map(filtering=True)
    gen_data_train(data_noise_free,data_noise_free,map_s0,map_r2s,sigma_g=111)
    data_train = get_data_train(sigma_g=111)
    gen_data_train_patches(data_train)
    data_train_patches = get_data_train_patches(sigma_g=111)
    # data_train_patches_n  = data_train_patches['noise data']
    data_train_patches_nf = data_train_patches['noise free data']
    data_train_patches_r2s= data_train_patches['r2s map']
    data_train_patches_s0 = data_train_patches['s0 map']
    num_patch,h,w,c = data_train_patches_nf.shape

    # add noise to patches
    print('add mixed sigma noise...')
    data_noise = []
    num_p = h*w*c
    for i in range(num_patch):
        sg = np.random.choice([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0])
        patch_nf = np.reshape(data_train_patches_nf[i], [-1,1])
        r = np.reshape(sg*np.random.standard_normal(num_p),[-1,1]) + patch_nf
        i = np.reshape(sg*np.random.standard_normal(num_p),[-1,1])
        s = np.sqrt(r**2+i**2)
        patch_n = np.reshape(s,[h,w,c])
        data_noise.append(patch_n)
    data_train_patches_n = np.array(data_noise)

    # random permutation
    index_pixel = np.array(range(0,num_patch))
    index_pixel_random = np.random.permutation(index_pixel)
    
    # split data into train data and validation data
    split_boundary = int(num_patch * split)  
    data_train_patches_nf_train  = data_train_patches_nf[index_pixel_random[:split_boundary]]
    data_train_patches_nf_valid  = data_train_patches_nf[index_pixel_random[split_boundary:]]

    data_train_patches_n_train   = data_train_patches_n[index_pixel_random[:split_boundary]]
    data_train_patches_n_valid   = data_train_patches_n[index_pixel_random[split_boundary:]]

    data_train_patches_r2s_train = data_train_patches_r2s[index_pixel_random[:split_boundary]]
    data_train_patches_r2s_valid = data_train_patches_r2s[index_pixel_random[split_boundary:]]

    data_train_patches_s0_train  = data_train_patches_s0[index_pixel_random[:split_boundary]]
    data_train_patches_s0_valid  = data_train_patches_s0[index_pixel_random[split_boundary:]]

    # save train data and validation data
    data_train_patches_train = {'noise free data':np.array(data_train_patches_nf_train),
                                'noise data':np.array(data_train_patches_n_train),
                                's0 map':np.array(data_train_patches_s0_train),
                                'r2s map':np.array(data_train_patches_r2s_train),
                                'sigma_g':sigma_g}
    
    np.save(os.path.join('data_train_simulated', 'simulated_data_train_patches_'+str(sigma_g)),data_train_patches_train)
    
    data_train_patches_valid = {'noise free data':np.array(data_train_patches_nf_valid),
                                'noise data':np.array(data_train_patches_n_valid),
                                's0 map':np.array(data_train_patches_s0_valid),
                                'r2s map':np.array(data_train_patches_r2s_valid),
                                'sigma_g':sigma_g}
    
    np.save(os.path.join('data_validation_simulated', 'simulated_data_valid_patches_'+str(sigma_g)),data_train_patches_valid)

def split_data_trian_patches(sigma_g=7,split=0.8):
    data_train_patches = get_data_train_patches(sigma_g)
    data_train_patches_n  = data_train_patches['noise data']
    data_train_patches_nf = data_train_patches['noise free data']
    data_train_patches_r2s= data_train_patches['r2s map']
    data_train_patches_s0 = data_train_patches['s0 map']
    num_patch,h,w,c = data_train_patches_nf.shape

    if num_patch == 15200 :
        # random permutation
        index_pixel = np.array(range(0,num_patch))
        index_pixel_random = np.random.permutation(index_pixel)
        print('>> Split start...')
        
        # split data into train data and validation data
        split_boundary = int(num_patch * split)  
        data_train_patches_nf_train  = data_train_patches_nf[index_pixel_random[:split_boundary]]
        data_train_patches_nf_valid  = data_train_patches_nf[index_pixel_random[split_boundary:]]

        data_train_patches_n_train   = data_train_patches_n[index_pixel_random[:split_boundary]]
        data_train_patches_n_valid   = data_train_patches_n[index_pixel_random[split_boundary:]]

        data_train_patches_r2s_train = data_train_patches_r2s[index_pixel_random[:split_boundary]]
        data_train_patches_r2s_valid = data_train_patches_r2s[index_pixel_random[split_boundary:]]

        data_train_patches_s0_train  = data_train_patches_s0[index_pixel_random[:split_boundary]]
        data_train_patches_s0_valid  = data_train_patches_s0[index_pixel_random[split_boundary:]]

        # save train data and validation data
        data_train_patches_train = {'noise free data':np.array(data_train_patches_nf_train),
                                    'noise data':np.array(data_train_patches_n_train),
                                    's0 map':np.array(data_train_patches_s0_train),
                                    'r2s map':np.array(data_train_patches_r2s_train),
                                    'sigma_g':sigma_g}
        
        np.save(os.path.join('data_train_simulated', 'simulated_data_train_patches_'+str(sigma_g)),data_train_patches_train)
        
        data_train_patches_valid = {'noise free data':np.array(data_train_patches_nf_valid),
                                    'noise data':np.array(data_train_patches_n_valid),
                                    's0 map':np.array(data_train_patches_s0_valid),
                                    'r2s map':np.array(data_train_patches_r2s_valid),
                                    'sigma_g':sigma_g}
        
        np.save(os.path.join('data_validation_simulated', 'simulated_data_valid_patches_'+str(sigma_g)),data_train_patches_valid)

        print('>> Training data size: '+str(data_train_patches_n_train.shape[0])+' Validation data size: '+str(data_train_patches_n_valid.shape[0])+
        ' (Total= '+str(num_patch)+')')
    else:
        print('>> The data has been splited.'+'Training data size: '+str(num_patch))


def get_data_valid_patches(sigma_g=111):
    # print('Get training data patches...')
    data = np.load(os.path.join('data_validation_simulated','simulated_data_valid_patches_'+str(sigma_g)+'.npy'),allow_pickle=True).item()  
    return data

def print_line():
    print('='*50)


    
if __name__ == '__main__':
    tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    
# =============================================================================
#     Generate clinical data
# =============================================================================
    # clean up and resample all clinical data into .npy file.
    file_dir = 'data_liver_mr_same_te'
    # clearup_clinical_data(file_dir)
    data_clinical = get_clinical_data().astype('float32')
    print('All clinical data shape : ',data_clinical.shape)
    num_study,h,w,c = data_clinical.shape
    
    # # show clinical data
    # for i in range(num_study):
    #     plt.figure()
    #     plt.imshow(data_clinical[i,:,:,0], cmap='gray', interpolation='none'),plt.title('Study '+str(i))
    
# =============================================================================
#     generate clinical data maps 
# =============================================================================
    print_line()
    # generate maps not using denoised weighted images
    # gen_map(data_clinical, tes)
    # map_s0,map_r2s = get_map()
    
    # generate maps using denoised weighted images
    # gen_map(data_clinical, tes, filtering=True)
    map_s0_nlm_fix_autotrunc,map_r2s_nlm_fix_autotrunc = get_map(filtering=True)
    

    # show denoising results
    # data_clinical_denoised_nlm = np.load(os.path.join('data_clinical','clinical_data_denoised_nlm.npy'))
    # data_clinical_denoised_nlm_fix = np.load(os.path.join('data_clinical','clinical_data_denoised_nlm_fix.npy'))
    # j=0
    # for i in range(num_study):
    #     plt.figure()
    #     plt.subplot(221)
    #     plt.imshow(data_clinical[i,:,:,j],cmap='jet',interpolation='none'),plt.title('Study '+str(i))
    #     plt.subplot(222)
    #     plt.imshow(data_clinical_denoised_nlm[i,:,:,j],cmap='jet',interpolation='none'),plt.title('Study '+str(i)+' denoised')
    #     plt.subplot(223)
    #     plt.imshow(data_clinical_denoised_nlm_fix[i,:,:,j],cmap='jet',interpolation='none'),plt.title('Study '+str(i)+' denoised(fix)')
    
    # show mapping results
    # map_s0_nlm_autotrunc = np.load(os.path.join('data_clinical', 'clinical_data_map_s0_nlm_autotrunc.npy'))
    # map_r2s_nlm_autotrunc = np.load(os.path.join('data_clinical', 'clinical_data_map_r2s_nlm_autotrunc.npy'))
    # for i in range(num_study):
    #     plt.figure()
    #     plt.subplot(221)
    #     plt.imshow(map_r2s_nlm_autotrunc[i,:,:],cmap='jet',vmax=1000,vmin=0,interpolation='none'),plt.title('R2s map (nlm) '+str(i)),plt.colorbar(fraction=0.022)
    #     plt.subplot(222)
    #     plt.imshow(map_r2s_nlm_fix_autotrunc[i,:,:],cmap='jet',vmax=1000,vmin=0,interpolation='none'),plt.title('R2s map (nlm fix) '+str(i)),plt.colorbar(fraction=0.022)
    #     plt.subplot(223)
    #     plt.imshow(map_s0[i,:,:],cmap='jet',vmax=300,vmin=0,interpolation='none'),plt.title('S0 map (raw) '+str(i)),plt.colorbar(fraction=0.022)
    #     plt.subplot(224)
    #     plt.imshow(map_r2s[i,:,:],cmap='jet',vmax=1000,vmin=0,interpolation='none'),plt.title('R2s map (raw) '+str(i)),plt.colorbar(fraction=0.022)
    
# =============================================================================
#     generate background mask
# =============================================================================
    # generate mask from clincial image data
    # gen_bkg_mask(data_clinical)
    
    # generate mask from clinical image data after denoising
    # gen_bkg_mask(prep(data_clinical))
    
    # get automatic created masks
    # masks = get_bkg_mask(manual=False)
    
    # get manual created masks
    masks = get_bkg_mask(manual=True)
    
    # generate test data masks
    gen_mask_test(masks,train_size=100)
    
    # get test data masks
    # mask_test = get_mask_test()
    
    # show background mask results
    # j=0
    # for i in range(0,num_study):
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(masks[i,:,:]),plt.title('mask '+str(i))
    #     plt.subplot(122)
    #     plt.imshow(data_clinical[i,:,:,j],cmap='gray',vmax=100.0),plt.title('Study '+str(i)+' TE '+str(j))
    
    # mask out map background region
    map_s0_masked = map_s0_nlm_fix_autotrunc*masks
    map_r2s_masked = map_r2s_nlm_fix_autotrunc*masks
    
    # show masked maps
    # for i in range(num_study):
    #     plt.figure()
    #     plt.subplot(221)
    #     plt.imshow(map_r2s_nlm_fix_autotrunc[i],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.title('R2s map '+str(i))
    #     plt.subplot(222)
    #     plt.imshow(map_r2s_masked[i],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.title('R2s map (masked) '+str(i))
    #     plt.subplot(223)
    #     plt.imshow(map_s0_nlm_fix_autotrunc[i],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.title('S0 map '+str(i))
    #     plt.subplot(224)
    #     plt.imshow(map_s0_masked[i],cmap='jet',interpolation='none',vmin=0,vmax=1000),plt.title('S0 map (masked) '+str(i))
    
# =============================================================================
#     generate simulated noise and noise-free data
# =============================================================================
    # generate simulated noise-free data
    # gen_data_noise_free(map_s0_masked,map_r2s_masked,tes)
    data_noise_free = get_data_noise_free()
    
    # show noise-free data
    # j=1
    # k=10
    # for i in range(num_study):
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(data_noise_free[i,:,:,j],cmap='jet',interpolation='none'),plt.title('Study '+str(i)+' TE '+str(j))
    #     plt.subplot(122)
    #     plt.imshow(data_noise_free[i,:,:,k],cmap='jet',interpolation='none'),plt.title('Study '+str(i)+' TE '+str(k))
        
    # generate simulated noise data in different noise
    sigmas = [0.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0]

    for sigma in sigmas:
        gen_data_noise(data_noise_free,sigma)
    
    # get simulated noise data 
    # data_noise = get_data_noise(sigma_g=0.0)
    
    # show noise-free and noisy data
    # j=1
    # k=10
    # for i in range(num_study):
    #     plt.figure()
    #     plt.subplot(221)
    #     plt.imshow(data_noise_free[i,:,:,j],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('Study '+str(i)+' TE '+str(j))
    #     plt.subplot(222)
    #     plt.imshow(data_noise_free[i,:,:,k],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('Study '+str(i)+' TE '+str(k))
    #     plt.subplot(223)
    #     plt.imshow(data_noise[i,:,:,j],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('Study '+str(i)+' TE '+str(j))
    #     plt.subplot(224)
    #     plt.imshow(data_noise[i,:,:,k],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('Study '+str(i)+' TE '+str(k))
    
    # show same study at different noise level
    # p = 100
    # stu = []
    # for sigma in sigmas:
    #     data = get_data_noise(sigma)
    #     stu.append(data[p,:,:,:])
    # j=1
    # k=11
    # for i in range(0,np.array(stu).shape[0]):
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(stu[i][:,:,j],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('Study '+str(p)+' TE '+str(j))
    #     plt.subplot(122)
    #     plt.imshow(stu[i][:,:,k],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('Study '+str(p)+' TE '+str(k))
        
# =============================================================================
#     generate simulated data for training
# =============================================================================
    # generate simulated data (containing traning data and testing data)
    for sigma in sigmas:
        data_noise = get_data_noise(sigma)
        gen_data_train(data_noise_free,data_noise,map_s0_masked,map_r2s_masked,sigma)
    
    # get simulated data
    # data_train=get_data_train(sigma_g=0.0)
    
# =============================================================================
#     generate simulated data for testing
# =============================================================================
    for sigma in sigmas:
        data_train=get_data_train(sigma)
        gen_data_test(data_train) # for model testing
        gen_data_test_noise(data_train) # for map calculation using other methods
    
    # data_test = get_data_test(sigma_g=5)
    
    # show test data
    # map_r2s_test = data_test['r2s map']
    # data_test_noise = data_test['noise data']
    # for i in range(map_r2s_test.shape[0]):
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(map_r2s_test[i],cmap='jet',interpolation='none',vmax=1000,vmin=0),plt.title('R2s map test '+str(i))
    #     plt.subplot(122)
    #     plt.imshow(data_test_noise[i,:,:,1],cmap='jet',interpolation='none',vmax=300,vmin=0),plt.title('data test (noisy) '+str(i))
    
    
# =============================================================================
#     generate simulated data patches for training
# =============================================================================
    for sigma in sigmas:
        data = get_data_train(sigma)
        gen_data_train_patches(data)

    # get patches for training
    # data_train_patches = get_data_train_patches(sigma_g=0.0)
    
    # show patches generation results
    # data_noise_free_patches = data_train_patches['noise free data']
    # data_noise_patches = data_train_patches['noise data']
    # map_r2s_patches = data_train_patches['r2s map']
    # map_s0_patches = data_train_patches['s0 map']
    # plt.figure()
    # k = 250
    # plt.subplot(221)
    # plt.imshow(data_noise_free_patches[k,:,:,0],cmap='gray',vmax=200),plt.title('image (noise-free) patch '+str(k)),plt.colorbar(fraction=0.022)
    # plt.subplot(222)
    # plt.imshow(data_noise_patches[k,:,:,0],cmap='gray',vmax=200),plt.title('image (noise) patch '+str(k)),plt.colorbar(fraction=0.022)
    # plt.subplot(223)
    # plt.imshow(map_s0_patches[k,:,:],cmap='jet'),plt.title('S0 map patch '+str(k)),plt.colorbar(fraction=0.022)
    # plt.subplot(224)
    # plt.imshow(map_r2s_patches[k,:,:],cmap='jet'),plt.title('R2s map patch '+str(k)),plt.colorbar(fraction=0.022)
    
# =============================================================================
#     generate clinical data patches for training
# =============================================================================
    for sigma in sigmas:
        gen_clinical_data_patches(data_clinical,sigma)
    
    # get clinical data patches
    # data_clinical_patches = get_clinical_data_patches(sigma_g=17)
    
    # show patches generation results
    # data_clinical_nf__patches = data_clinical_patches['noise free data']
    # data_clinical_n_patches = data_clinical_patches['noise data']
    # plt.figure()
    # p = 120
    # plt.subplot(221),plt.axis('off')
    # plt.imshow(data_clinical_nf__patches[p,:,:,0],cmap='gray',vmax=200),plt.title('clinical img (nf) patch'+str(p))
    # plt.subplot(222),plt.axis('off')
    # plt.imshow(data_clinical_n_patches[p,:,:,0],cmap='gray',vmax=200),plt.title('clinical img (n) patch'+str(p))
    # plt.subplot(223),plt.axis('off')
    # plt.imshow(data_noise_free_patches[p,:,:,0],cmap='gray',vmax=200),plt.title('simulated img (nf) patch'+str(p))
    # plt.subplot(224),plt.axis('off')
    # plt.imshow(data_noise_patches[p,:,:,0],cmap='gray',vmax=200),plt.title('simulated img (n) patch'+str(p))
    
# =============================================================================
#     test training data generator
# =============================================================================
    # batch_iter=train_datagen()
    
    # batch = next(batch_iter)
    # batch_x = batch[0]
    # batch_y = batch[1][0]
    
    # # show batch henerator results
    # plt.figure()
    # n=64
    # m=72
    # plt.subplot(221)
    # plt.imshow(batch_x[n,:,:,0],cmap='jet',vmax=200),plt.title('input image patch'+str(n))
    # plt.subplot(222)
    # plt.imshow(batch_y[n,:,:,0],cmap='jet',vmax=200),plt.title('output image patch'+str(n))
    # plt.subplot(223)
    # plt.imshow(batch_x[m,:,:,0],cmap='jet',vmax=200),plt.title('input image patch'+str(m))
    # plt.subplot(224)
    # plt.imshow(batch_y[m,:,:,0],cmap='jet',vmax=200),plt.title('output image patch'+str(m))

# =============================================================================
#     clean up vivo test data
# =============================================================================
    # clearup_vivo_data()
    # data_vivo = get_vivo_data()
    
    # # show vivo data
    # plt.figure()
    # i=11
    # up=250
    # plt.subplot(221),plt.axis('off')
    # plt.imshow(data_vivo[0,:,:,i],cmap='jet',interpolation='none',vmin=0,vmax=250),plt.title('vivo data (normal) TE '+str(i))
    # plt.subplot(222),plt.axis('off')
    # plt.imshow(data_vivo[1,:,:,i],cmap='jet',interpolation='none',vmin=0,vmax=200),plt.title('vivo data (mild) TE '+str(i))
    # plt.subplot(223),plt.axis('off')
    # plt.imshow(data_vivo[2,:,:,i],cmap='jet',interpolation='none',vmin=0,vmax=150),plt.title('vivo data (moderate) TE '+str(i))
    # plt.subplot(224),plt.axis('off')
    # plt.imshow(data_vivo[3,:,:,i],cmap='jet',interpolation='none',vmin=0,vmax=100),plt.title('vivo data (severe) TE '+str(i))
