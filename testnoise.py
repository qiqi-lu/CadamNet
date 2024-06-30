import matplotlib.pyplot as plt
import numpy as np
import os
import helper
data_dir = os.path.join('data','liver','InVivo')
wImgNInVivo = np.load(os.path.join(data_dir,'wImg121.npy'))
mask = np.load(os.path.join(data_dir,'maskBody.npy'))[...,0]
print(mask.shape)
sigmas1 = helper.SigmaG(wImgNInVivo,num_coil=1,mask=mask)
print(np.mean(sigmas1[20:]))
print(np.median(sigmas1[20:]))
sigmas2 = helper.SigmaG(wImgNInVivo,num_coil=1)
print(np.mean(sigmas2[20:]))
print(np.median(sigmas2[20:]))


plt.figure(figsize=(20,10))
plt.bar(np.linspace(0,120,121),sigmas1,width=0.3,label='bkg')
plt.bar(np.linspace(0,120,121)+0.3,sigmas2,width=0.3,label='rect')
plt.legend()
plt.savefig(os.path.join('figure','Sigma,png'))
