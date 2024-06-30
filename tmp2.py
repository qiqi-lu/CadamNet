import numpy as np
import os
import matplotlib.pyplot as plt
import methods
import helper

data_dir_iv  = os.path.join('data','liver','InVivo')
wImg_iv      = np.load(os.path.join(data_dir_iv,'wImg121.npy'))
maskBody_iv  = np.load(os.path.join(data_dir_iv,'maskBody.npy'))
maskParen_iv = np.load(os.path.join(data_dir_iv,'maskParenchyma.npy'))

wImg_iv_n = wImg_iv
sigma_invivo_n = helper.SigmaG(wImg_iv_n)
print(sigma_invivo_n[19])

i=119
tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
map_pcanr_iv=methods.PCANR(imgs=wImg_iv_n[i],tes=tes,sigma=8.0,beta=1.3,f=5,m=0,Ncoils=1,pbar_leave=False)
map_pcanr_iv = np.array(map_pcanr_iv)

m_pcanr,std_pcanr = helper.mean_std_roi(map_pcanr_iv[np.newaxis][...,1],maskParen_iv[i,...,1][np.newaxis])
print(m_pcanr)