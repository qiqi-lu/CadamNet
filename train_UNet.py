import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import models as mod
import helper
import losses
import config

id = 7
config.config_gpu(id)

model_name = 'UNetUS3tv6_035'
# model_name = 'UNetUS3tv5_002'
# model_name = 'DnCNN3_012'
# model_name = 'DnCNN3'
weight = 0.035

# batch_size = 256
batch_size = 128
epochs     = 300
save_every = 20
sigma      = 0

# noise_type='Gaussian'
# noise_type='Rician'
noise_type='InVivo'

m=1.0

tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
data_dir = os.path.join('data','liver',noise_type)

# wImg    = np.load(os.path.join(data_dir,'wImg_'+str(m)+'.npy')).astype(np.float32)
# wImgN   = np.load(os.path.join(data_dir,'wImgN'+str(sigma)+'_'+str(m)+'.npy')).astype(np.float32)

wImgN   = np.load(os.path.join(data_dir,'wImg121.npy')).astype(np.float32)

# maskBody= np.load(os.path.join(data_dir,'maskBody.npy')).astype(np.float32)
# maskBody= np.repeat(maskBody,6,axis=-1)

# wImg   = wImg*maskBody
# wImgN  = wImgN*maskBody

# wPatch  = helper.makePatch(wImg[0:100],patch_size=32,stride=8,rescale=True,aug_times=4)
wPatchN = helper.makePatch(wImgN[0:100],patch_size=32,stride=8,rescale=True,aug_times=4)
# wPatchN = np.load(os.path.join(data_dir,'wPatchNmix.npy'))

index   = np.random.permutation(wPatchN.shape[0])
# wPatch  = wPatch[index]
wPatchN = wPatchN[index]

print('Training data shape:'+str(wPatchN.shape))

model_dir = os.path.join('model',model_name+'_'+'sigma'+str(sigma))
if not os.path.exists(model_dir): os.mkdir(model_dir)

img_channels  = 12

##### #####
# model_denoise = mod.SeparableCNN(depth=8,depth_multi=10,filters=40,image_channels=img_channels,dilation=1)
# model_mapping = mod.UNetH(image_channels=img_channels)
# model_mapping = mod.UNetDouble(image_channels=img_channels)
# model_mapping = mod.UNet(image_channels=img_channels,output_channel=3,dilation_rate=(2,2))
model_mapping = mod.UNet(image_channels=img_channels,output_channel=3)
# model_mapping = mod.UNet(image_channels=img_channels,output_channel=2)


inpts    = tf.keras.Input(shape=(None,None,img_channels))
outpts_mapping   = model_mapping(inpts)

# model = tf.keras.Model(inputs=inpts,outputs=outpts_mapping)
model = tf.keras.Model(inputs=inpts,outputs=[outpts_mapping,outpts_mapping])
##### ######

model.summary()

initial_epoch = helper.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:  
    print('Resuming by loading epoch %03d'%initial_epoch)
    model = tf.keras.models.load_model(os.path.join(model_dir,'model_%03d.h5'%initial_epoch), compile=False)

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              # loss=[losses.exp],
              # loss=[losses.ncexp],
              # loss=[losses.exp,losses.tv],
              # loss=[losses.exp_square],
              # loss=[losses.exp_square,losses.tv],loss_weights=[1.0,weight],
              # loss=[losses.exp_square,losses.tv2],loss_weights=[1.0,weight],
              # loss=[losses.exp_square,losses.tv3],loss_weights=[1.0,weight],
              # loss=[losses.exp_square,losses.tv4],loss_weights=[1.0,weight],
              # loss=[losses.exp_square,losses.tv5],loss_weights=[1.0,weight],
              loss=[losses.exp_square,losses.tv6],loss_weights=[1.0,weight],
            #   loss=[losses.exp,losses.tv6],loss_weights=[1.0,weight],
              )

checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=False, period=save_every)
csv_logger   = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'log.csv'), append=True, separator=',')
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule)
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule2)
tensorboard  = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)

model.fit(  x=wPatchN,
            # y=[wPatch,wPatch],
            # y=[wPatch],
            # y=[wPatchN],
            y=[wPatchN,wPatchN],
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            validation_split = 0.2,
            shuffle=True,
            callbacks=[checkpointer,csv_logger,lr_scheduler,tensorboard],
            # callbacks=[checkpointer,csv_logger,lr_scheduler,tensorboard,callbacks.LogLoss()],
            )