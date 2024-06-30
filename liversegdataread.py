# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:09:25 2020

@author: Loo
"""

import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import time

def save_data_as_npy():
    """Convert nii data to npy data
    Only include the slices containing the liver
    """
    time_start = time.time()
    print('Save the raw data to npy data')
    # get the mask and image name
    filenames_mask  = glob.glob(r'D:\MRI\R2Star\Datasets\liversegCT\seg*.nii')
    filenames_image = glob.glob(r'D:\MRI\R2Star\Datasets\liversegCT\vol*.nii')
    num_patient = len(filenames_image)
    
    # truncate the raw data, only use the slices containing liver
    index_slice = [[51,  71],
                   [51,  71],[311,471],[296,436],[281,451],[308,408],[278,398],[253,373],[301,451],[350,520],[385,535],[329,509],
                   [357,467],[366,546],[373,433],[115,145],[276,376],[156,256],[406,566],[278,438],[349,549],
                   [340,460],
                   [383,583],
                   [349,469],
                   [336,486],
                   [347,497],
                   [367,507],
                   [362,512],]
    
    # save as npy formate data
    for index_patient in range(0,num_patient):
        # all mask slices        
        mask = nib.load(filenames_mask[index_patient])
        # liver mask slices
        mask_liver = mask.dataobj[:,:,index_slice[index_patient][0]:index_slice[index_patient][1]]
        # save the liver mask slices to npy data, each file contains one patient's data
        file_mask = "%s%d"%('Datanpy\mask',index_patient)
        np.save(file_mask,mask_liver)
        
        # all image slices
        image = nib.load(filenames_image[index_patient])
        # liver image slices
        image_liver = image.dataobj[:,:,index_slice[index_patient][0]:index_slice[index_patient][1]]
        # save the liver image slices to npy data, each file cantains one patient's data
        file_name = "%s%d"%('Datanpy\image',index_patient)
        np.save(file_name,image_liver)
        
        print("process[",index_patient,"]: shape:",mask.shape,'=>',
              index_slice[index_patient][1]-index_slice[index_patient][0],'=',
              mask_liver.shape[2])
    
    time_end = time.time() 
    print('Saving done, consuming:',time_end-time_start)
    
def load_npy_data(num_patient=0):
    """load npy format data saved
    
    # Arguments
        num_patient (int): the number of patients wanted
        
    # Returns
        mask  (tensor): shape = (num_row, num_col, num_slice)
        image (tensor): shape = (num_row, num_col, num_slice)
    """
    
    time_start = time.time()
    # get the mask and image file names
    filenames_mask  = glob.glob(r'D:\MRI\R2Star\CodePython\Datanpy\mask*.npy')
    filenames_image = glob.glob(r'D:\MRI\R2Star\CodePython\Datanpy\image*.npy')
    
    # load all data in the files
    if num_patient == 0:
        num_patient = len(filenames_mask) 
 
    # load the first patient's mask and data
    mask  = np.load(filenames_mask[0])
    image = np.load(filenames_image[0])
    
    # append num_patient patients' mask and image data into one mask array and 
    # one image array
    if num_patient > 1:
        for index_patient in range(1,num_patient):
            mask = np.append(mask,np.load(filenames_mask[index_patient]),2)
            image = np.append(image,np.load(filenames_image[index_patient]),2)     
    
    # count the image loaded        
    num_image = mask.shape[-1]
    
    time_end = time.time()
    print('num of patient:',num_patient,'num of image:',num_image)
    print('Load npy data done:',time_end-time_start)
        
    return mask,image

def create_train_data(mask,image,split = 0.8):
    mask_reshape = np.zeros([mask.shape[-1],mask.shape[0],mask.shape[1]])
    image_reshape = np.zeros([image.shape[-1],image.shape[0],image.shape[1]])
    for index_slice in range(0,mask.shape[-1]):
        mask_reshape[index_slice,:,:] = mask[:,:,index_slice]
        image_reshape[index_slice,:,:] = image[:,:,index_slice]
        
    split_boundary = int(mask_reshape.shape[0] * split)
    mask_train = mask_reshape[:split_boundary,:,:]
    mask_test  = mask_reshape[split_boundary:,:,:]
    image_train = image_reshape[:split_boundary,:,:]
    image_test = image_reshape[split_boundary:,:,:]
    
    mask_train  = np.reshape(mask_train, [mask_train.shape[0], mask_train.shape[1],mask_train.shape[2],1])
    image_train = np.reshape(image_train,[image_train.shape[0],image_train.shape[1],image_train.shape[2],1])
    mask_test   = np.reshape(mask_test,  [mask_test.shape[0],  mask_test.shape[1],mask_test.shape[2],1])
    image_test  = np.reshape(image_test, [image_test.shape[0], image_test.shape[1],image_test.shape[2],1])
        
    return mask_train, image_train, mask_test, image_test
        



if __name__ == '__main__':
#    save_data_as_npy()
#    mask,image = load_npy_data(4)
#    mask_train, image_train, mask_test, image_test = create_train_data(mask,image)
    
    


#mask0 = nib.load(filenames_mask[0])
#print('shape of the image',mask0.shape)
#
#data = mask0.dataobj[:,:,index_slice[0][0]:index_slice[0][1]]
#
#plt.figure(0)
#plt.imshow(data[:,:,-1].transpose(),origin='lower')
#
#image0 = nib.load(filenames_image[0])
#datai = image0.dataobj[:,:,71]
#plt.figure(1)
#plt.imshow(datai.transpose(),origin='lower',cmap='Greys')





