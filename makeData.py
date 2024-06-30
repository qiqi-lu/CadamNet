import os
import numpy as np
import matplotlib.pyplot as plt
import helper
import scipy.ndimage


##### Data Information #####
#### Data Parameters
sigma = 6.5
tes   = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
m     = 1.0
# m     = 2.5

#### Noise Type
# noise_type='Gaussian'
# noise_type='Rician'
noise_type='ChiSquare'
NCoils = 1

#### Whether to save data
# save  = False
save  = True

data_dir = os.path.join('data','liver',noise_type)
if not os.path.exists(data_dir): os.makedirs(data_dir)

print('='*98+'\nSimulated data making('+str(sigma)+'), save to '+data_dir+'...')

##### LOAD PARAMETER MAPS (REFFERENCE) #####
# data = np.load(os.path.join('data_simulated','simulated_data_train_'+str(11)+'.npy'),allow_pickle=True).item()
# print('KEYS: ',data.keys())
# pImg = np.stack([data['s0 map'],data['r2s map']],-1)
# np.save(os.path.join(data_dir,'pImg.npy'),pImg)

pImg=np.load(os.path.join(data_dir,'pImg.npy'))
print('Parameter maps shape:', pImg.shape)

##### LOAD MASKS #####
# mask_body = np.load(os.path.join('data_clinical','clinical_data_bkg_mask_manual.npy'))
# mask_whole_liver = np.load(os.path.join('data_clinical','mask_liver_whole_manual.npy'))
# mask_parenchyma  = np.load(os.path.join('data_clinical','mask_liver_parenchyma_manual.npy'))
# mask_body=np.stack([mask_body,mask_body],axis=-1)
# mask_whole_liver=np.stack([mask_whole_liver,mask_whole_liver],axis=-1)
# mask_parenchyma=np.stack([mask_parenchyma,mask_parenchyma],axis=-1)
# np.save(os.path.join(data_dir,'maskBody.npy'),mask_body)
# np.save(os.path.join(data_dir,'maskLiver.npy'),mask_whole_liver)
# np.save(os.path.join(data_dir,'maskParenchyma.npy'),mask_parenchyma)

mask_body        = np.load(os.path.join(data_dir,'maskBody.npy'))
mask_whole_liver = np.load(os.path.join(data_dir,'maskLiver.npy'))
mask_parenchyma  = np.load(os.path.join(data_dir,'maskParenchyma.npy'))
print('Mask shape:', mask_body.shape)

if m > 1.0:
    print('Multiply...')
    a = np.array(mask_body)
    b = np.array(mask_parenchyma)
    c = np.zeros(b.shape)
    for i in range(b.shape[0]): c[i,...,1] = scipy.ndimage.gaussian_filter(b[i,:,:,1]*(m-1),sigma=0.75)
    b[...,1] = c[...,1]
    a[...,1] = a[...,1] + b[...,1]
    pImg = pImg*a
else:
    pImg=pImg*np.array(mask_body)
if save: np.save(os.path.join(data_dir,'pImg'+'_'+str(m)+'.npy'),pImg)

plt.figure(figsize=(40,10))
index = 100
plt.subplot(2,4,1),plt.imshow(mask_body[index,:,:,0],cmap='gray',interpolation='none'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,2),plt.imshow(mask_body[index,:,:,1],cmap='gray',interpolation='none'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,3),plt.imshow(mask_whole_liver[index,:,:,0],cmap='gray',interpolation='none'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,4),plt.imshow(mask_whole_liver[index,:,:,1],cmap='gray',interpolation='none'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,5),plt.imshow(mask_parenchyma[index,:,:,0],cmap='gray',interpolation='none'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,6),plt.imshow(mask_parenchyma[index,:,:,1],cmap='gray',interpolation='none'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,7),plt.imshow(pImg[index,:,:,0],cmap='jet',vmax=500,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('$S_0$')
# plt.subplot(2,4,8),plt.imshow(pImg[index,:,:,1],cmap='jet',vmax=1875,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('$R_2$')
plt.subplot(2,4,8),plt.imshow(pImg[index,:,:,1],cmap='jet',vmax=900,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('$R_2$')
plt.savefig(os.path.join('figure','Mask&Map.png'))

#### SIMULATED WEIGHTED IMAGES #####
wImg,wImgN = helper.makePairedData(pImg,tes=tes,sigma=sigma,noise_type=noise_type,NCoils=NCoils)
print('Weighted images shape:',wImgN.shape)

if save: np.save(os.path.join(data_dir,'wImg'+'_'+str(m)+'.npy'),wImg)
if save: np.save(os.path.join(data_dir,'wImgN'+str(sigma)+'_'+str(m)+'.npy'),wImgN)

plt.figure(figsize=(40,10))
index = 100
plt.subplot(2,4,1),plt.imshow(wImg[index,:,:,0],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024),plt.title('TE1')
plt.subplot(2,4,2),plt.imshow(wImg[index,:,:,1],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024),plt.title('TE2')
plt.subplot(2,4,3),plt.imshow(wImg[index,:,:,2],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024),plt.title('TE3')
plt.subplot(2,4,4),plt.imshow(wImg[index,:,:,3],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024),plt.title('TE4')
plt.subplot(2,4,5),plt.imshow(wImgN[index,:,:,0],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024)
plt.subplot(2,4,6),plt.imshow(wImgN[index,:,:,1],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024)
plt.subplot(2,4,7),plt.imshow(wImgN[index,:,:,2],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024)
plt.subplot(2,4,8),plt.imshow(wImgN[index,:,:,3],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.024)
plt.savefig(os.path.join('figure','wImg&wImgN.png'))


# print('='*98+'\nIn vivo data making...')
# data_dir_invivo =  os.path.join('data','liver','InVivo')
# if not os.path.exists(data_dir_invivo): os.makedirs(data_dir_invivo)

# # wImgInVivo = np.load(os.path.join('data_clinical','clinical_data.npy'))
# # np.save(os.path.join(data_dir_invivo,'wImg'),wImgInVivo)
# np.save(os.path.join(data_dir_invivo,'maskBody'),mask_body)
# np.save(os.path.join(data_dir_invivo,'maskLiver'),mask_whole_liver)
# np.save(os.path.join(data_dir_invivo,'maskParenchyma'),mask_parenchyma)

# wImgInVivo = np.load(os.path.join(data_dir_invivo,'wImg121.npy'))
# print('In vivo data shape: ',wImgInVivo.shape)

# pImg_pcanr_invivo = np.load(os.path.join(data_dir_invivo,'pImgPCANR121.npy'))
# pImg_m1ncm_invivo = np.load(os.path.join(data_dir_invivo,'pImgM1NCM121.npy'))
# print('Parameter map shape (PCANR): ',pImg_pcanr_invivo.shape)
# print('Parameter map shape (M1NCM): ',pImg_m1ncm_invivo.shape)

# plt.figure(figsize=(40,20))
# index = 25
# plt.subplot(2,4,1),plt.imshow(wImgInVivo[index,:,:,0],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('TE1')
# plt.subplot(2,4,2),plt.imshow(wImgInVivo[index,:,:,1],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('TE2')
# plt.subplot(2,4,3),plt.imshow(wImgInVivo[index,:,:,2],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('TE3')
# plt.subplot(2,4,4),plt.imshow(wImgInVivo[index,:,:,3],cmap='gray',vmax=450,vmin=0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('TE4')
# plt.subplot(2,4,5),plt.imshow(pImg_m1ncm_invivo[index,:,:,0],cmap='jet',vmax = 500,vmin =0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('M1NCM S0')
# plt.subplot(2,4,6),plt.imshow(pImg_m1ncm_invivo[index,:,:,1],cmap='jet',vmax = 1100,vmin =0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('M1NCM R2')
# plt.subplot(2,4,7),plt.imshow(pImg_pcanr_invivo[index,:,:,0],cmap='jet',vmax = 500,vmin =0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('PCANR S0')
# plt.subplot(2,4,8),plt.imshow(pImg_pcanr_invivo[index,:,:,1],cmap='jet',vmax = 1100,vmin =0,interpolation='none'),plt.colorbar(fraction=0.022),plt.title('PCANR R2')
# plt.savefig(os.path.join('figure','Invivo.png'))

