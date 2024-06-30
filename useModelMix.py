import numpy as np
import os
import config
import tensorflow as tf
import matplotlib.pyplot as plt
import helper

id=1
config.config_gpu(id)

print('='*98+'\nLoad data...')
data_dir = os.path.join('data','liver','Rician')
wPatchN  = np.load(os.path.join(data_dir,'wPatchNmix.npy'))
wImg     = np.load(os.path.join(data_dir,'wImg_1.0.npy')).astype(np.float32)
wPatch   = helper.makePatch(wImg[0:100],patch_size=32,stride=8,rescale=True,aug_times=4)


model_nameA = 'Denoiser'
# model_nameA = 'DeepT2s'
epochA=300
sigmaTrnA=0

model_dirA = os.path.join('model',model_nameA+'_sigma'+str(sigmaTrnA))
modelA     = tf.keras.models.load_model(os.path.join(model_dirA,'model_'+str(epochA)+'.h5'),compile=False)
wPatchDnA  = modelA.predict(wPatchN)
np.save(os.path.join(data_dir,'wPatchNmixDn'),wPatchDnA)

plt.figure(figsize=(30,10))
ind = 100
plt.subplot(3,4,1)
plt.imshow(wPatch[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noise free TE0'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,2)
plt.imshow(wPatch[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noise free TE1'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,3)
plt.imshow(wPatchN[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noisy TE0 '),plt.colorbar(fraction=0.022)
plt.subplot(3,4,4)
plt.imshow(wPatchN[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noisy TE1 '),plt.colorbar(fraction=0.022)
plt.subplot(3,4,5)
plt.imshow(wPatchDnA[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('DN '),plt.colorbar(fraction=0.022)
plt.subplot(3,4,6)
plt.imshow(wPatchDnA[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('DN '),plt.colorbar(fraction=0.022)
plt.savefig(os.path.join('figure','tmp'))
