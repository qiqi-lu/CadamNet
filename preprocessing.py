#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 00:21:19 2020

@author: luqiqi
"""
import cv2 as cv

def denoise_nlm_multi(data_multi):
    data_multi = data_multi.astype('uint16')
    for i in range(data_multi.shape[2]):
        img = data_multi[:,:,i]
        dst=cv.fastNlMeansDenoising(src=img, dst=None, h=[5.0], templateWindowSize=3, searchWindowSize=11, normType = cv.NORM_L1)
        data_multi[:,:,i] = dst
    data_multi = data_multi.astype('float32')
    return data_multi

def read_raw(file_name,data_type='uint8',shape=None):
    # read binary files
    import numpy as np
    data = np.fromfile(file_name,dtype=data_type)
    data = np.reshape(data,shape)
    return data

def read_dcm(file_path):
    """ 
    Read all .dcm file in the folder into a array and their TE value
    """
    import glob
    import pydicom
    import numpy as np
    import os

    file_names = glob.glob(file_path)
    file_names = sorted(file_names,key=os.path.getmtime,reverse=True)
    num_files = np.array(file_names).shape[0]
    if num_files==0: print(':: Error: dicom file not found.')
    image_multite = []
    tes = []
    # read data and info
    for file,i in zip(file_names, range(0,num_files)):
        image_multite.append(pydicom.dcmread(file).pixel_array)
        tes.append(pydicom.dcmread(file).EchoTime)
        
    image_multite = np.array(image_multite).transpose((1,2,0))
    return image_multite,np.array(tes)

if __name__ == '__main__':
    import os
    file = os.path.join('data_test_clinical','mask','mild_body.raw')
    image = read_raw(file,shape=[64,128])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image)
    plt.savefig(os.path.join('figure','z.png'))

    file_path = os.path.join('data_test_clinical','Study20081016_151327_156000_moderate_new','*.dcm')
    # file_path = os.path.join('data_test_clinical','Study20081127_133901_218000_severe_new','*.dcm')
    data,_=read_dcm(file_path)
    print(data.shape)

    import numpy as np
    data_clinical = np.load(os.path.join('data_clinical','clinical_data.npy'))
    print(data_clinical.shape)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(data_clinical[100,:,:,0])
    plt.subplot(1,2,2)
    plt.imshow(data_clinical[111,:,:,0])
    plt.savefig(os.path.join('figure','z.png'))

