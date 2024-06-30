# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:25:01 2020

@author: QiqiLu
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from data import create_ideal_and_simu_data, read_dicom
import numpy as np
import os


def loss_model_sqexp_image(y_true,y_pred):
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    recon = tf.reshape(tf.square(y_pred[:,0]),[-1,1])*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])/1000.0*tes)+2.0*tf.square(tf.reshape(y_pred[:,2],[-1,1]))
    recon = tf.reshape(recon,[-1,64,128,12])
    loss = tf.keras.losses.MAE(tf.square(y_true),recon)
    # loss = tf.keras.losses.MSE(y_true,tf.sqrt(recon))
    return loss

def mse_r2(y_true,y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred),axis=-1)

def loss_model_sqexp_trueplus(y_true,y_pred):
    y_true_data = y_true[:,0:12]
    y_true_true = y_true[:,12:]
    
    # loss1 = tf.keras.losses.MAE(y_true_true,y_pred)
    loss1 = tf.keras.losses.MSE(y_true_true,y_pred)
    loss2 = loss_model_sqexp(y_true_data, y_pred)
    
    lambda1 = 0.1
    lambda2 = 0.9
    loss = lambda1*loss1 + lambda2*loss2
    return loss

def loss_model_ncexp_trueplus(y_true,y_pred):
    y_true_data = y_true[:,0:12]
    y_true_true = y_true[:,12:]
    
    # loss1 = tf.keras.losses.MAE(y_true_true,y_pred)
    # loss1 = tf.keras.losses.MSLE(y_true_true,y_pred)
    loss1 = tf.keras.losses.MSE(y_true_true,y_pred)
    loss2 = loss_model_ncexp(y_true_data, y_pred)
    
    lambda1 = 1.0
    lambda2 = 0.1
    loss = lambda1*loss1 + lambda2*loss2
    return loss
    

def loss_model_sqexp(y_true,y_pred):
    # y_pred: [batch_size, outputs_size]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    recon = tf.reshape(tf.square(y_pred[:,0]),[-1,1])*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])/1000.0*tes)+2.0*tf.square(tf.reshape(y_pred[:,2],[-1,1]))
    loss = tf.keras.losses.MAE(tf.square(y_true),recon)
    # loss = tf.keras.losses.MAE(y_true,tf.sqrt(recon))
    return loss

def loss_model_sqexp_teplus(y_true,y_pred):
    # y_pred: [batch_size, outputs_size]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    recon = tf.reshape(tf.square(y_pred[:,0]),[-1,1])*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])/1000.0*tes)+2.0*tf.square(tf.reshape(y_pred[:,2],[-1,1]))
    
    y_true = tf.reshape(y_true,[-1,12,2])
    y_true = y_true[:,:,0]
    
    loss = tf.keras.losses.MAE(tf.square(y_true),recon)
    # loss = tf.keras.losses.MAE(y_true,tf.sqrt(recon))
    return loss


def loss_model_ncexp(y_true,y_pred):
    # y_pred: [batch_size, outputs_size]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    
    y_pred_c = y_pred+0.000001

    s = tf.reshape(y_pred_c[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred_c[:,1],[-1,1])/1000.0*tes)
    alpha = tf.square(0.5*s/tf.reshape(y_pred_c[:,2],[-1,1]))
    # alpha = tf.square(0.5*s/(tf.reshape(y_pred[:,2],[-1,1])+tf.exp(-8.0)))
    tempw = (1.0+2.0*alpha)*tf.math.bessel_i0e(alpha)+2.0*alpha*tf.math.bessel_i1e(alpha)
    recon = tf.sqrt(0.5*np.pi*tf.square(tf.reshape(y_pred_c[:,2],[-1,1])))*tempw
    
    # loss = tf.keras.losses.MAE(y_true,recon) # more on s0
    # loss = tf.keras.losses.MAE(y_true/(tf.reshape(y_true[:,0],[-1,1])+0.00001),recon/(tf.reshape(recon[:,0],[-1,1])+0.00001)) # more on s0
    # loss = tf.keras.losses.MSLE(y_true,recon) # pay more attention to the sigma estimation
    # loss = tf.keras.losses.MAPE(y_true,recon) # can't
    loss = tf.keras.losses.MSE(y_true,recon)
    # loss = tf.keras.losses.MAE(tf.square(y_true),tf.square(recon)) # maybe
    # loss = tf.keras.losses.MSE(tf.math.log(y_true+1),tf.math.log(recon+1)) # no
    return loss

def loss_model_ncexp_teplus(y_true,y_pred):
    # y_pred: [batch_size, outputs_size]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred_c = y_pred+0.000001
    
    s = tf.reshape(y_pred_c[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred_c[:,1],[-1,1])/1000.0*tes)
    alpha = tf.square(0.5*s/tf.reshape(y_pred_c[:,2],[-1,1]))
    tempw = (1.0+2.0*alpha)*tf.math.bessel_i0e(alpha)+2.0*alpha*tf.math.bessel_i1e(alpha)
    recon = tf.sqrt(0.5*np.pi*tf.square(tf.reshape(y_pred_c[:,2],[-1,1])))*tempw
    
    # recon = tf.reshape(recon,[-1,12,1])
    # recon = np.insert(recon,1,tes,axis=2)
    # recon = tf.reshape(recon,[-1,recon.shape[-2]*recon.shape[-1]])
    
    y_true = tf.reshape(y_true,[-1,12,2])
    y_true_c = y_true[:,:,0]
    
    # loss = tf.keras.losses.MAE(y_true,recon) # more on s0
    # loss = tf.keras.losses.MAE(y_true/(tf.reshape(y_true[:,0],[-1,1])+0.00001),recon/(tf.reshape(recon[:,0],[-1,1])+0.00001)) # more on s0
    # loss = tf.keras.losses.MSLE(y_true_c,recon) # pay more attention to the sigma estimation
    # loss = tf.keras.losses.MAPE(y_true,recon) # can't
    loss = tf.keras.losses.MSE(y_true_c,recon)
    # loss = tf.keras.losses.MAE(tf.square(y_true_c),tf.square(recon)) # maybe
    # loss = tf.keras.losses.MSE(tf.math.log(y_true+1),tf.math.log(recon+1)) # no
    return loss

def loss_sigma(y_true,y_pred):
    mad = np.reshape(np.median(y_true,axis=-1), [-1,1])
    mad_mad = np.abs(y_true-mad)
    K=1.4826
    sg_mad = np.reshape(K*np.median(mad_mad,axis=-1),[-1,1])
    loss = np.square(y_pred[:,-1]-sg_mad)
    return loss

if __name__ == '__main__':
    # # load the training data and test data from txt file
    # train_x,train_y,test_x,test_y,TEs = load_txt_data()
    
    # # reshape train and test x data
    # train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
    # train_y = np.reshape(train_y,(train_y.shape[0],1))
    # test_x  = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))
    # test_y  = np.reshape(test_y, (test_y.shape[0],1))

    # # add TEs informatio to the data
    # train_x_with_te = np.insert(train_x,1,TEs,axis=2)
    # test_x_with_te = np.insert(test_x,1,TEs,axis=2)
    
    # # set the training parameters
    # version = 5
    
    # # start training
    # predict_y,model = train_model(train_x_with_te,train_y,test_x_with_te,test_y,version)
    
    # # read dicom image data
    # filepath = 'D:\MRI\R2Star\CodePython\datadcm\*.dcm'
    # data,TE = read_dicom(filepath=filepath)
    
    # # use trained model to predict the T2s image
    # timeStart = time.time()
    # ImageT2s = model_predict_map(model,data,TE,version)
    # timeEnd = time.time()
    
    # # plot the result
    # fig = plt.figure(1)
    # plt.plot(predict_y,'g:')    
    # plt.plot(test_y,'r-')
    # plt.show()
    
    # plt.figure(2)
    # plt.plot(predict_y,test_y,'*')
    # plt.plot([1,12],[1,12])
    # plt.show()
    
    # plt.imshow(ImageT2s)
    
    # print("The time used for prediction one T2s image:",timeEnd-timeStart)

    flag_recreate_train_data = 0
    flag_retrain = 0
    
    # create simulated train data
    # train data parameters
    noise_level = np.linspace(15,60,10)
    S0 = np.linspace(0,400,80)
    T2 = 1000.0/np.linspace(0,2000,200)
    num_repeat = 1
    num_pixel = 1
    channel = 1
    
    image_test,TEs = read_dicom(os.path.join('datadcm', '*.dcm'))
     
    if flag_recreate_train_data:
        print('Recreate simulation data...')
        # create simulated data with noise and t2 value for each data
        _, data_train_noise, data_train_t2 = create_ideal_and_simu_data(noise_level=noise_level,
                                                            S0=S0,T2=T2,
                                                            num_repeat=num_repeat,
                                                            TEs=TEs,num_pixel=num_pixel,
                                                            num_channel=channel)

        # save the simulated train data
        np.save(os.path.join('datatrain', 'dataTrainNoise'), data_train_noise)
        np.save(os.path.join('datatrain', 'dataTrainT2'), data_train_t2)
    else:
        # load saved simulated data
        print('Load the saved simulated train data ...')
        data_train_noise = np.load(os.path.join('datatrain', 'dataTrainNoise.npy'))
        data_train_t2 = np.load(os.path.join('datatrain', 'dataTrainT2.npy'))
    
    if flag_retrain:
        print('Retrain the deep learning model ...')
        # reshape train and test data
        x = np.reshape(data_train_noise,(-1,data_train_noise.shape[-1],1))
        y = np.reshape(data_train_t2,(-1,1))
        
        # random permutation
        index_pixel = np.array(range(0,x.shape[0]))
        index_pixel_random = np.random.permutation(index_pixel)
        
        # split data into train data and test data
        split=0.8
        split_boundary = int(x.shape[0] * split)  
        train_x = x[index_pixel_random[:split_boundary],:]
        test_x  = x[index_pixel_random[split_boundary:],:]
        train_y = y[index_pixel_random[:split_boundary],:]
        test_y  = y[index_pixel_random[split_boundary:],:]
        
        # add TEs informatio to the data
        train_x_with_te = np.insert(train_x,1,TEs,axis=2)
        test_x_with_te = np.insert(test_x,1,TEs,axis=2)
    
 
    

    
    