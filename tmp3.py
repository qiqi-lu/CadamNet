# import tensorflow as tf
import numpy as np
# import config
import os
import helper
import matplotlib.pyplot as plt


sigma      = 0
noise_type = 'InVivo'
m          = 1.0
aug        = 6
step       = 8
resc       = True

print('loading...')
tes       = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
data_dir  = os.path.join('data','liver',noise_type)
wImgN     = np.load(os.path.join(data_dir,'wImg121.npy')).astype(np.float32)
sigmas    = helper.SigmaG(wImgN)
mask      = np.load(os.path.join(data_dir,'maskBody.npy'))
map_pcanr = np.load(os.path.join(data_dir,'pImgPCANRnew.npy'))
map_pcanr = map_pcanr*mask

n       = 100
wPatchN = helper.makePatch(wImgN[0:n,2:62,14:114,:],patch_size=32,stride=step,rescale=resc,aug_times=aug)
pPatch  = helper.makePatch(map_pcanr[0:n,2:62,14:114,:],patch_size=32,stride=step,rescale=resc,aug_times=aug)
meanR2  = np.mean(pPatch[...,-1],axis=(1,2))
meanS   = np.mean(wPatchN,axis=(1,2,3))

# pPatch     = pPatch[...,0][...,np.newaxis]
pixel_dif1 = pPatch[:, 1:, :, :] - pPatch[:, :-1, :, :] # finite forward difference donw->up
pixel_dif2 = pPatch[:, :, 1:, :] - pPatch[:, :, :-1, :] # right->left
tv = np.sum(np.abs(pixel_dif1),axis=(1,2,3))+np.sum(np.abs(pixel_dif2),axis=(1,2,3))

wei    = 3.0*np.exp(-meanR2/200.0)+1.0
wei    = np.exp(meanS/200.0)


plt.figure()
# plt.plot(meanR2,tv,    '*b',markersize=0.25)
# plt.plot(meanR2,tv*wei,'*r',markersize=0.25)
plt.plot(meanS,tv,    '*b',markersize=0.25)
plt.plot(meanS,tv*wei,'*r',markersize=0.25)
# plt.plot(meanR2,meanS,'*b',markersize=0.25)
plt.savefig('figure/wei.png')

