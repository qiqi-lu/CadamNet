# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:50:49 2020

@author: Loo
"""
import skimage
import tensorflow as tf
import os
import argparse

# Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DeepT2s(2)', type=str, help='model name')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data_train_simulated', type=str, help='path of train data')
parser.add_argument('--sigma', default=111, type=int, help='sigma of Gaussian noise')
parser.add_argument('--epoch', default=600, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
args = parser.parse_args()

save_dir = os.path.join('model',args.model+'_'+'sigma'+str(args.sigma)) 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def SeparableCNN(depth=8,depth_multi=10,filters=40, image_channels=12,dilation=1):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))
    # 1st layer, Separable Conv+relu
    x = tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=(3,3),
                                        strides=(1,1),
                                        kernel_initializer='Orthogonal',
                                        padding='same',
                                        dilation_rate=(dilation,dilation),
                                        depth_multiplier=depth_multi)(inpt)
    x = tf.keras.layers.Activation('relu')(x)
    # depth-2 layers, Separable Conv+BN+relu
    for i in range(depth-2):
        x = tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=(3,3),
                                        strides=(1,1),
                                        kernel_initializer='Orthogonal',
                                        padding='same',
                                        dilation_rate=(dilation,dilation),
                                        depth_multiplier=depth_multi)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
        x = tf.keras.layers.Activation('relu')(x)  
    # last layer, Conv
    x = tf.keras.layers.SeparableConv2D(filters=12,
                                        kernel_size=(3,3),
                                        strides=(1,1),
                                        kernel_initializer='Orthogonal',
                                        padding='same',
                                        dilation_rate=(dilation,dilation),
                                        depth_multiplier=depth_multi)(x)
    # ResNet architecture
    x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model

def UNet(image_channels=12):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))
    
    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(inpt) # 64*128
    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv1)  # 64*128
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1) # 32*64
    
    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2) # 16*32
    
    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv3)
    
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3) # 8*16
    
    conv4 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4) # 4*8
    
    convbase = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool4)    # 4*8
    convbase = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(convbase) # 4*8
    
    conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) # 8*16*1024
    
    conv5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc5) # 8*16*512
    conv5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv5) # 8*16*512
    
    conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3]) # 16*32*512
    
    conv6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc6) # 16*32*256
    conv6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv6) # 16*32*256
    
    conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2]) # 32*64*256
    
    conv7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc7) # 32*64*128
    conv7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv7) # 32*64*128
    
    conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(32, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1]) # 64*128*128
    
    conv8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conc8) # 64*128*64
    conv8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv8) # 64*128*64
    
    convone = tf.keras.layers.Conv2D(2,(1,1),activation='relu')(conv8) # 64*128*3
    
    model = tf.keras.Model(inputs=inpt,outputs=convone)
    
    return model

def DeepT2s(image_channels=12):
    
    model_denoise = SeparableCNN(depth=8,depth_multi=10,filters=40,image_channels=image_channels,dilation=1)
    model_mapping = UNet(image_channels=image_channels)
    
    inpts = tf.keras.Input(shape=(None,None,image_channels))
    denoise_pred = model_denoise(inpts)
    mapping_pred = model_mapping(denoise_pred)
    
    model = tf.keras.Model(inputs=inpts,
                           outputs=[denoise_pred,mapping_pred],
                           )
    return model

def findLastCheckpoint(save_dir):
    import glob, re
    file_list = glob.glob(os.path.join(save_dir,'model_*.h5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    return lr
   
# define loss
def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true))/2

def loss_model_exp_image(y_true,y_pred):
    # y_pred: [batch_size, 64, 128,2]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # y_pred =y_pred
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    y_true = tf.reshape(y_true,[-1,12])
    recon = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])/1000.0*tes)
    # recon = tf.reshape(recon,[-1,32,32,12])
    loss = tf.keras.backend.sum(tf.keras.backend.square(recon - y_true))/2
    return loss
    
if __name__ == '__main__':
    import config
    config.config_gpu(3)
    model_type = 2
    
    from data_generator import train_datagen,get_data_valid_patches

    # load validation data
    data_valid = get_data_valid_patches(sigma_g=111)
    data_valid_nf = data_valid['noise free data']
    data_valid_n  = data_valid['noise data']
    if model_type==2:
        validation_data = (data_valid_n,[data_valid_nf,data_valid_nf])
    else:
        validation_data = (data_valid_n,data_valid_nf)

    # create model
    model = DeepT2s()
    model.summary()
    
    # load the last model parameters
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = tf.keras.models.load_model(os.path.join(save_dir,'model_%03d.h5'%initial_epoch), compile=False)
    
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                  loss=[sum_squared_error,
                        loss_model_exp_image],
                  loss_weights = [1.0,1.0],
                  )
    
    # use call back functions
    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.h5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    history = model.fit_generator(train_datagen(batch_size=args.batch_size,sigma_g=args.sigma,du=True),
                steps_per_epoch=2000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler],validation_data=validation_data)