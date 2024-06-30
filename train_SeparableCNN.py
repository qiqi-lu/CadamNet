# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:50:49 2020
Train SeparableCNN for denoising.
@author: luqiqi
"""
import skimage
import tensorflow as tf
import os
import argparse

# Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SeparableCNN', type=str, help='model name')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data_train_simulated', type=str, help='path of train data')
parser.add_argument('--sigma', default=3, type=int, help='sigma of Gaussian noise')
parser.add_argument('--epoch', default=300, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=10, type=int, help='save model at every x epoches')
args = parser.parse_args()

save_dir = os.path.join('model',args.model+'_'+'sigma'+str(args.sigma)) 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def SeparableCNN(depth=8,depth_nulti=10,filters=40, image_channels=12,dilation=1):
    layer_count = 0
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))

    # 1st layer, Separable Conv+relu
    layer_count += 1
    x = tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=(3,3),
                                        strides=(1,1),
                                        kernel_initializer='Orthogonal',
                                        padding='same',
                                        dilation_rate=(dilation,dilation),
                                        depth_multiplier=depth_nulti,
                                        name = 'conv'+str(layer_count))(inpt)
    x = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x)

    # depth-2 layers, Separable Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),kernel_initializer='Orthogonal',padding='same',dilation_rate=(dilation,dilation),
                                            depth_multiplier=depth_nulti,name = 'conv'+str(layer_count))(x)
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        x = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x)

    # last layer, Conv
    layer_count += 1
    x = tf.keras.layers.SeparableConv2D(filters=12,kernel_size=(3,3),strides=(1,1),kernel_initializer='Orthogonal',padding='same',dilation_rate=(dilation,dilation),
                                        depth_multiplier=depth_nulti,name = 'conv'+str(layer_count))(x)
                                        
    # ResNet architecture
    x = tf.keras.layers.Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise

    model = tf.keras.Model(inputs=inpt, outputs=x)
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
    
if __name__ == '__main__':
    import config
    config.config_gpu(4)
    
    from data_generator import train_datagen
    
    model = SeparableCNN(depth=8,depth_nulti=10,filters=40,image_channels=12,dilation=1)
    model.summary()
    
    # load the last model
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = tf.keras.models.load_model(os.path.join(save_dir,'model_%03d.h5'%initial_epoch), compile=False)
    
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=sum_squared_error)
    
    # use call back functions
    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.h5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    history = model.fit_generator(train_datagen(batch_size=args.batch_size,sigma_g=args.sigma,du=False),
                steps_per_epoch=2000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])