# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:35:03 2020

@author: Recklexx
Refference: 1.https://github.com/yihui-he/u-net/blob/master/train.py
"""

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
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
        file_image = "%s%d"%('Datanpy\image',index_patient)
        np.save(file_image,image_liver)
        
        print("process[",index_patient,"]: shape:",mask.shape,'=>',
              index_slice[index_patient][1]-index_slice[index_patient][0],'=',
              mask_liver.shape[2])
    
    time_end = time.time() 
    print('Saving done, consuming:',time_end-time_start)
    
def load_npy_data(num_patient = 0):
    """load npy format data saved
    
    # Arguments
        num_patient (int): the number of patients wanted
        
    # Returns
        mask  (tensor): shape = (num_row, num_col, num_slice)
        image (tensor): shape = (num_row, num_col, num_slice)
    """
    print('Load npy data...')
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
    print('Num of patient:',num_patient,',Num of image:',num_image)
    print('Load done:',time_end-time_start,'s.')
        
    return mask,image

def create_train_data(mask,image,split = 0.8):
    """Create train data and test data.
    Reshape the raw data and split it into train data and test data.
    
    # Arguments
        mask  (tensor): mask data, shape = (num_row, num_col, num_image)
        image (tensor): image data, shape = (num_row, num_col, num_image)
        split  (float): the percentage of the data for training model
        
    # Returns
        mask_train  (tensor): mask data for training, shape = (num_image, num_row, num_col, num_channel)
        mask_test   (tensor): mask data for testing, shape = (num_image, num_row, num_col, num_channel)
        image_train (tensor): image data for training, shape = (num_image, num_row, num_col, num_channel)
        image_test  (tensor): image data for testing, shape = (num_image, num_row, num_col, num_channel)
    """
    print('Create train data and test data...')
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
    
    print('Create done.')
    return mask_train, image_train, mask_test, image_test

def U_Net(input_shape):
    """U-Net
    
    # Arguments
        input_shape (tensor): shape = (num_row, num_col, num_channel)
        
    # Returns
        model (Model): Keras model instance
    """
    inputs = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(filters=64,
                                kernel_size=(3,3),
                                activation='relu',
                                padding='same')(inputs)
    conv1 = keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv1)
    pool1 = keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
    
    conv2 = keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool1)
    conv2 = keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv2)
    pool2 = keras.layers.MaxPool2D(pool_size=(2,2))(conv2)
    
    conv3 = keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool2)
    conv3 = keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv3)
    pool3 = keras.layers.MaxPool2D(pool_size=(2,2))(conv3)
    
    conv4 = keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool3)
    conv4 = keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv4)
    pool4 = keras.layers.MaxPool2D(pool_size=(2,2))(conv4)
    
    convbase = keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)
    convbase = keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(convbase)
    
    conc5 = keras.layers.Concatenate()([keras.layers.Conv2D(512,(3,3),padding='same')(keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4])
    conv5 = keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conc5)
    conv5 = keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv5)
    
    conc6 = keras.layers.Concatenate()([keras.layers.Conv2D(256,(3,3),padding='same')(keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3])
    conv6 = keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc6)
    conv6 = keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv6)
    
    conc7 = keras.layers.Concatenate()([keras.layers.Conv2D(128,(3,3),padding='same')(keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2])
    conv7 = keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc7)
    conv7 = keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv7)
    
    conc8 = keras.layers.Concatenate()([keras.layers.Conv2D(64, (3,3),padding='same')(keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1])
    conv8 = keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc8)
    conv8 = keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv8)
    
    convone = keras.layers.Conv2D(1,(1,1),activation='relu')(conv8)
    
    model = keras.Model(inputs=inputs,outputs=convone)
    
    return model

def dice_coef(y_true,y_predict):
    """Dice coefficient
    
    # Arguments
        y_true (tensor): true mask
        y_predict (tensor): predicted mask
    
    # Returns
        Dice coefficient (float)
    """
    y_true_f = keras.backend.flatten(y_true)
    y_predict_f = keras.backend.flatten(y_predict)
    
    smooth = 1.
    
    # overlap area of the predict and ground-truth maps
    intersection = keras.backend.sum(y_true_f*y_predict_f)
    
    # Dice = twice the overlap area of predictred and ground-truth maps, 
    #        divided by the  total number of pixels in both images.
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f*y_true_f)
            + keras.backend.sum(y_predict_f*y_predict_f) + smooth)

def dice_loss(y_true,y_predict):
    """Dice coefficent loss
    
    # Arguments
        y_true (tensor): true mask
        y_predict (tensor): predicted mask
    
    # Returns
        Dice coefficient loss (float)
    """
    return 1.-dice_coef(y_true,y_predict)


def train_model(image_train, mask_train, image_test, mask_test):
    """model train function
    
    # Arguments
        image_train (tensor): images for training model
        mask_train  (tensor): masks of the train images
        image_test  (tensor): images for testing model
        mask_test   (tensor): masks of the test images
    
    # Returns
        model    (Model): trained Keras model
        predict (tensor): predict mask      
    """
    input_shape = (image_train.shape[-3],image_train.shape[-2],image_train.shape[-1])
    model = U_Net(input_shape)
    model.compile(loss=dice_loss, optimizer=keras.optimizers.Adam(lr=1e-5), netrics = dice_coef)
    model.summary()
    
    model.fit(image_train,mask_train,batch_size=1,epochs=20)
    predict = model.predict(image_test)
    return model, predict
    
if __name__ == '__main__':
    print('-'*60)
    print('Loading and processing train data and test data...')
    print('-'*60)
#    save_data_as_npy()
    mask,image = load_npy_data(2)
    mask_train, image_train, mask_test, image_test = create_train_data(mask,image)    
    
    print('-'*60)
    print('Creating and fitting model...')
    print('-'*60)
#    model, predict = train_model(image_train=image_train,
#                                 mask_train=mask_train,
#                                 image_test=image_test,
#                                 mask_test=mask_test)
    
    
    
    
    
    