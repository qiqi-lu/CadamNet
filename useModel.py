import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config
import metricx

id = 7
config.config_gpu(id)

m=1.0
sigmaTst  = 7

sigmaTrnA = 0
sigmaTrnB = 0



model_nameA = 'Denoiser'
# model_nameB = 'DeepT2s'
# model_nameB = 'CadamNet'
# model_nameB = 'Denoiser'
# model_nameB = 'DeepT2s(2)'
model_nameB = 'DeepT2s'

epochA=300
epochB=300

print('='*98+'\nLoad data...')
data_dir = os.path.join('data','liver','Rician')
pImg     = np.load(os.path.join(data_dir,'pImg_'+str(m)+'.npy'))
wImg     = np.load(os.path.join(data_dir,'wImg_'+str(m)+'.npy'))
wImgN    = np.load(os.path.join(data_dir,'wImgN'+str(sigmaTst)+'_'+str(m)+'.npy'))

model_dirA = os.path.join('model',model_nameA+'_sigma'+str(sigmaTrnA))
model_dirB = os.path.join('model',model_nameB+'_sigma'+str(sigmaTrnB))
modelA = tf.keras.models.load_model(os.path.join(model_dirA,'model_'+str(epochA)+'.h5'),compile=False)
modelB = tf.keras.models.load_model(os.path.join(model_dirB,'model_'+str(epochB)+'.h5'),compile=False)
print('Load model...\n'+model_dirA+'\n'+model_dirB)

wImgDnA   = modelA.predict(wImgN)
wImgDnB,_ = modelB.predict(wImgN)

print('Save Denoiser predicted images...')
np.save(os.path.join('data','liver','Rician','wImgDnM'+str(sigmaTst)),wImgDnA)

maskLiver = np.load(os.path.join(data_dir,'maskLiver.npy'))
maskBody  = np.load(os.path.join(data_dir,'maskBody.npy'))
maskParen = np.load(os.path.join(data_dir,'maskParenchyma.npy'))

nrmseA0=metricx.nRMSE(wImg[...,0],wImgDnA[...,0],maskLiver)
nrmseA1=metricx.nRMSE(wImg[...,1],wImgDnA[...,1],maskLiver)
nrmseB0=metricx.nRMSE(wImg[...,0],wImgDnB[...,0],maskLiver)
nrmseB1=metricx.nRMSE(wImg[...,1],wImgDnB[...,1],maskLiver)

p0 = metricx.Pvalue(nrmseA0[100:],nrmseB0[100:],alt='greater')
p1 = metricx.Pvalue(nrmseA1[100:],nrmseB1[100:],alt='greater')
print(p0,p1)

plt.figure(figsize=(30,10))
ind = 100
plt.subplot(3,4,1)
plt.imshow(wImg[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noise free TE0'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,2)
plt.imshow(wImg[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noise free TE1'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,3)
plt.imshow(wImgN[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noisy TE0 ('+str(sigmaTst)+')'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,4)
plt.imshow(wImgN[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title('Noisy TE1 ('+str(sigmaTst)+')'),plt.colorbar(fraction=0.022)

plt.subplot(3,4,5)
plt.imshow(wImgDnA[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title(model_nameA+' TE0'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,6)
plt.imshow(wImgDnA[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title(model_nameA+' TE1'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,7)
plt.imshow(wImgDnB[ind,...,0],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title(model_nameB+' TE0'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,8)
plt.imshow(wImgDnB[ind,...,1],cmap='gray',vmax=400,vmin=0,interpolation='none'),plt.title(model_nameB+' TE1'),plt.colorbar(fraction=0.022)

plt.subplot(3,4,9)
plt.imshow(np.abs(wImg[ind,...,0]-wImgDnA[ind,...,0]),cmap='gray',vmax=50,vmin=0,interpolation='none'),plt.title(model_nameA+' Abs Diff')
plt.colorbar(fraction=0.022),plt.title('NRMSE = '+str(nrmseA0[ind]),loc='left')
plt.subplot(3,4,10)
plt.imshow(np.abs(wImg[ind,...,1]-wImgDnA[ind,...,1]),cmap='gray',vmax=50,vmin=0,interpolation='none'),plt.title(model_nameA+' Abs Diff')
plt.colorbar(fraction=0.022),plt.title('NRMSE = '+str(nrmseA1[ind]),loc='left')
plt.subplot(3,4,11)
plt.imshow(np.abs(wImg[ind,...,0]-wImgDnB[ind,...,0]),cmap='gray',vmax=50,vmin=0,interpolation='none'),plt.title(model_nameB+' Abs Diff')
plt.colorbar(fraction=0.022),plt.title('NRMSE = '+str(nrmseB0[ind]),loc='left'),plt.title('P = '+str(np.round(p0,4)),loc='right')
plt.subplot(3,4,12)
plt.imshow(np.abs(wImg[ind,...,1]-wImgDnB[ind,...,1]),cmap='gray',vmax=50,vmin=0,interpolation='none'),plt.title(model_nameB+' Abs Diff')
plt.colorbar(fraction=0.022),plt.title('NRMSE = '+str(nrmseB1[ind]),loc='left'),plt.title('P = '+str(np.round(p1,4)),loc='right')

plt.savefig(os.path.join('figure','Denoiser_output'))

