import config
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import metricx
import helper

id = 5
config.config_gpu(id)
m=1.0
# m = 2.5 # data multiplication


sigmaTst  = 17
#                 0          1           2          3            4            5      6          7           8        9    
model_name = ['DeepT2s','CadamNet4001','CadamNet','DeepT2s(2)','DeepT2s(1)','UNet','UNetH','CadamNet251','UNetH25','MapNet']
model_type = [2,2,2,2,2,1,1,2,1,1]

sigmaTrnA = 17
epochA = '300'
iA = 5
model_nameA,model_typeA = model_name[iA],model_type[iA]

sigmaTrnB = 17
epochB = '300'
iB = 9
model_nameB,model_typeB = model_name[iB],model_type[iB]

##### SIMULATION TEST #####
print('='*98+'\nLoad simulated testing data...')
data_dir = os.path.join('data','liver','Rician')

pImg     = np.load(os.path.join(data_dir,'pImg_'+str(m)+'.npy'))
wImg     = np.load(os.path.join(data_dir,'wImg_'+str(m)+'.npy'))
wImgN    = np.load(os.path.join(data_dir,'wImgN'+str(sigmaTst)+'_'+str(m)+'.npy'))

# data for testing
pImgTst  = pImg[100:].astype(np.float32)
wImgTst  = wImg[100:].astype(np.float32)
wImgNTst = wImgN[100:].astype(np.float32)

# data after denoising using Denoiser
wImgDn    = np.load(os.path.join(data_dir,'wImgDn'+str(sigmaTst)+'.npy'))
wImgDnTst = wImgDn[100:].astype(np.float32)

print('Simulated testing data shape:',pImgTst.shape,wImgTst.shape,wImgNTst.shape)

# mask for testing
maskLiver = np.load(os.path.join(data_dir,'maskLiver.npy'))
maskLiverTst = maskLiver[100:]
maskBody  = np.load(os.path.join(data_dir,'maskBody.npy'))
maskBodyTst = maskBody[100:]
maskParen = np.load(os.path.join(data_dir,'maskParenchyma.npy'))
maskParenTst = maskParen[100:]

print('='*98+'\nLoad models...')
model_dirA = os.path.join('model',model_nameA+'_sigma'+str(sigmaTrnA))
model_dirB = os.path.join('model',model_nameB+'_sigma'+str(sigmaTrnB))
modelA = tf.keras.models.load_model(os.path.join(model_dirA,'model_'+str(epochA)+'.h5'),compile=False)
modelB = tf.keras.models.load_model(os.path.join(model_dirB,'model_'+str(epochB)+'.h5'),compile=False)
print('Test model:\n'+model_dirA+'/model_'+str(epochA)+'\n'+model_dirB+'/model_'+str(epochB))

print('='*98+'\nPredicting...')
def predict(model,model_type,dataTst):
    timeS = time.time()
    if model_type==2: xp,mapp = model.predict(dataTst)
    if model_type==1: mapp = model.predict(dataTst)
    t = time.time() - timeS
    print('Each study need: '+str(t/mapp.shape[0])+' second')
    return mapp

# mappA = predict(modelA,model_typeA,wImgNTst)
# mappB = predict(modelB,model_typeB,wImgNTst)

# mappA = predict(modelA,model_typeA,wImgNTst*np.repeat(maskBodyTst,6,axis=-1))
mappA = predict(modelA,model_typeA,wImgDnTst*np.repeat(maskBodyTst,6,axis=-1))
mappB = predict(modelB,model_typeB,wImgNTst*np.repeat(maskBodyTst,6,axis=-1))

model_nameC = 'Denoising&R2Fitting'
if sigmaTrnA == 0:
    mappC = np.load(os.path.join(data_dir,'pImgDnM'+str(sigmaTst)+'_'+str(m)+'.npy'))[-21:]
else:
    mappC = np.load(os.path.join(data_dir,'pImgDn'+str(sigmaTst)+'_'+str(m)+'.npy'))[-21:]

print('Map shape: '+str(mappA.shape)+str(mappB.shape))

print('='*98+'\nEvaluating...')
nrmseB = metricx.nRMSE(pImgTst[...,1],mappB[...,1],mask=maskLiverTst,mean=False)

nrmseA = metricx.nRMSE(pImgTst[...,1],mappA[...,1],mask=maskLiverTst,mean=False)
nrmseC = metricx.nRMSE(pImgTst[...,1],mappC[...,1],mask=maskLiverTst,mean=False)
PnrmseA = metricx.Pvalue(nrmseA,nrmseB,alt='greater')
PnrmseC = metricx.Pvalue(nrmseC,nrmseB,alt='greater')
print('(NRMSE) \n'+model_nameA+': \n'+str(nrmseA)+'\nMean = '+str(np.mean(nrmseA))+' SD = '+str(np.std(nrmseA))+
              '\n'+model_nameB+': \n'+str(nrmseB)+'\nMean = '+str(np.mean(nrmseB))+' SD = '+str(np.std(nrmseB))+
              '\n'+model_nameC+': \n'+str(nrmseC)+'\nMean = '+str(np.mean(nrmseC))+' SD = '+str(np.std(nrmseC))+
              '\nP value (A vs B): '+str(PnrmseA),
              '\nP value (C vs B): '+str(PnrmseC))

ssimA  = metricx.SSIM(pImgTst[...,1],mappA[...,1],mask=maskLiverTst,mean=False)
ssimB  = metricx.SSIM(pImgTst[...,1],mappB[...,1],mask=maskLiverTst,mean=False)
ssimC  = metricx.SSIM(pImgTst[...,1],mappC[...,1],mask=maskLiverTst,mean=False)
PssimA  = metricx.Pvalue(ssimA,ssimB,alt='less')
PssimC  = metricx.Pvalue(ssimC,ssimB,alt='less')
print('(SSIM) \n'+model_nameA+': \n'+str(ssimA)+'\nMean = '+str(np.mean(ssimA))+' SD = '+str(np.std(ssimA))+
             '\n'+model_nameB+': \n'+str(ssimB)+'\nMean = '+str(np.mean(ssimB))+' SD = '+str(np.std(ssimB))+
             '\n'+model_nameC+': \n'+str(ssimC)+'\nMean = '+str(np.mean(ssimC))+' SD = '+str(np.std(ssimC))+
             '\nP value (A vs B): '+str(PssimA),
             '\nP value (C vs B): '+str(PssimC))

print('='*98+'\nPloting maps from different methods...')
map_m1ncm = np.load(os.path.join(data_dir,'pImgM1NCM'+str(sigmaTst)+'_'+str(m)+'.npy'))[-21:,...,0:2]
map_pcanr = np.load(os.path.join(data_dir,'pImgPCANR'+str(sigmaTst)+'_'+str(m)+'.npy'))[-21:,...,0:2]

nrmse_m1ncm = metricx.nRMSE(pImgTst[...,1],map_m1ncm[...,1],mask=maskLiverTst,mean=False)
nrmse_pcanr = metricx.nRMSE(pImgTst[...,1],map_pcanr[...,1],mask=maskLiverTst,mean=False)

ssim_m1ncm  = metricx.SSIM(pImgTst[...,1],map_m1ncm[...,1],mask=maskLiverTst,mean=False,data_range=1024)
ssim_pcanr  = metricx.SSIM(pImgTst[...,1],map_pcanr[...,1],mask=maskLiverTst,mean=False,data_range=1024)

pImgTst   = pImgTst*maskBodyTst
map_m1ncm = map_m1ncm*maskBodyTst
map_pcanr = map_pcanr*maskBodyTst
# mappA     = mappA*maskBodyTst
mappA     = mappA[...,0:2]*maskBodyTst
mappB     = mappB*maskBodyTst
mappC     = mappC*maskBodyTst

vmaxs = np.array([800,240,400,500,600,200,350,200,200,150, 150,1000,200,600,125,350,1100,900,300,1000,300])
vmaxs = vmaxs*m
i     = 19
up    = 200.0*m
plt.figure(figsize=(40,30))
plt.subplot(4,3,1),plt.axis('off')
plt.imshow(pImgTst[i,:,:,1],  cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title('Reference'),plt.colorbar(fraction=0.022)
plt.subplot(4,3,2),plt.axis('off')
plt.imshow(map_m1ncm[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title('M$_1$NCM'),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssim_m1ncm[i]),loc='left')
plt.subplot(4,3,3),plt.axis('off')
plt.imshow(map_pcanr[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title('PCANR'),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssim_pcanr[i]),loc='left')

plt.subplot(4,3,4),plt.axis('off')
plt.imshow(wImgNTst[i,:,:,0],cmap='gray',interpolation='none',vmin=0,vmax=400),plt.title('TE$_0$'),plt.colorbar(fraction=0.022)
plt.subplot(4,3,5),plt.axis('off')
plt.imshow(np.abs(map_m1ncm[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title('M1NCM (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmse_m1ncm[i]),loc='left')
plt.subplot(4,3,6),plt.axis('off')
plt.imshow(np.abs(map_pcanr[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title('PCANR (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmse_pcanr[i]),loc='left')

plt.subplot(4,3,7),plt.axis('off')
plt.imshow(mappA[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title(model_nameA),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssimA[i]),loc='left')
plt.subplot(4,3,8),plt.axis('off')
plt.imshow(mappC[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title(model_nameC),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssimC[i]),loc='left')
plt.subplot(4,3,9),plt.axis('off')
plt.imshow(mappB[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title(model_nameB),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssimB[i]),loc='left')

plt.subplot(4,3,10),plt.axis('off')
plt.imshow(np.abs(mappA[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title(model_nameA+' (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmseA[i]),loc='left')
plt.subplot(4,3,11),plt.axis('off')
plt.imshow(np.abs(mappC[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title(model_nameC+' (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmseC[i]),loc='left')
plt.subplot(4,3,12),plt.axis('off')
plt.imshow(np.abs(mappB[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title(model_nameB+' (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmseB[i]),loc='left')

plt.savefig(os.path.join('figure','pImg'+str(i)))

# Regression Analysis
print('ROI Analysing...')
mean_refer,_ = helper.mean_std_roi(pImgTst[...,1],maskParenTst[...,1])
mean_m1ncm,_ = helper.mean_std_roi(map_m1ncm[...,1],maskParenTst[...,1])
mean_pcanr,_ = helper.mean_std_roi(map_pcanr[...,1],maskParenTst[...,1])
mean_predA,_ = helper.mean_std_roi(mappA[...,1],maskParenTst[...,1])
mean_predB,_ = helper.mean_std_roi(mappB[...,1],maskParenTst[...,1])
mean_predC,_ = helper.mean_std_roi(mappC[...,1],maskParenTst[...,1])
print(np.round(mean_refer,2))

xLimit = 1000*m
plt.figure(figsize=(7.5,7.5))
plt.scatter(mean_refer, mean_m1ncm,color='g',marker='o',label='M$_1$NCM')
fn_m1ncm = np.poly1d(np.polyfit(mean_refer,mean_m1ncm,1))
plt.plot([20,xLimit-20],fn_m1ncm([20,xLimit-20]),'--g')

plt.scatter(mean_refer, mean_pcanr,color='b',marker='o',label='PCANR')
fn_pcanr = np.poly1d(np.polyfit(mean_refer,mean_pcanr,1))
plt.plot([20,xLimit-20],fn_pcanr([20,xLimit-20]),'--b')

plt.scatter(mean_refer, mean_predA,color='orange',marker='o',label=model_nameA)
fn_pred_comp = np.poly1d(np.polyfit(mean_refer,mean_predA,1))
plt.plot([20,xLimit-20],fn_pred_comp([20,xLimit-20]),'--',color='orange')

plt.scatter(mean_refer, mean_predB,color='r',marker='o',label=model_nameB)
fn_pred_prop = np.poly1d(np.polyfit(mean_refer,mean_predB,1))
plt.plot([20,xLimit-20],fn_pred_prop([20,xLimit-20]),'--r')

plt.plot([0,xLimit],[0,xLimit],'-k')
plt.ylim([0,xLimit]),plt.ylabel('Estimated $R_2^*$ ($s^{-1}$)')
plt.xlim([0,xLimit]),plt.xlabel('Refference $R_2^*$ ($s^{-1}$)')
plt.legend(loc='upper left')
plt.title('ROI-Analysis'+' ('+str(sigmaTst)+')')
plt.savefig(os.path.join('figure','Regression.png'))

# Bland-Altman Analysis
print('Bland-Altman plot Analysing...')
yLimit = 150.0*m
plt.figure(figsize=(20,10))
plt.subplot(231)
helper.bland_altman_plot(x=mean_m1ncm,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title('M$_1$NCM vs Refer',loc='left')
plt.subplot(232)
helper.bland_altman_plot(x=mean_pcanr,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title('PCANR vs Refer',loc='left')
plt.subplot(234)
helper.bland_altman_plot(x=mean_predA,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title(model_nameA+' vs Refer',loc='left')
plt.subplot(235)
helper.bland_altman_plot(x=mean_predC,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title(model_nameC+' vs Refer',loc='left')
plt.subplot(236)
helper.bland_altman_plot(x=mean_predB,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title(model_nameB+' vs Refer',loc='left')
plt.savefig(os.path.join('figure','BAplot.png'))

# In vivo data testing
print('='*98+'\nIn vivo data evaluating...')
data_dir_iv  = os.path.join('data','liver','InVivo')
wImg_iv      = np.load(os.path.join(data_dir_iv,'wImg121.npy'))
maskBody_iv  = np.load(os.path.join(data_dir_iv,'maskBody.npy'))
maskParen_iv = np.load(os.path.join(data_dir_iv,'maskParenchyma.npy'))
map_pcanr_iv = np.load(os.path.join(data_dir_iv,'pImgPCANRnew.npy'))
map_m1ncm_iv = np.load(os.path.join(data_dir_iv,'pImgM1NCMnew.npy'))


mappA_iv = predict(modelA,model_typeA,wImg_iv*np.repeat(maskBody_iv,6,axis=-1))
mappB_iv = predict(modelB,model_typeB,wImg_iv*np.repeat(maskBody_iv,6,axis=-1))
# mappC_iv = np.load(os.path.join(data_dir_iv,'pImgDNF.npy'))

m_m1ncm,_ = helper.mean_std_roi(map_m1ncm_iv[...,1],maskParen_iv[...,1])
m_pcanr,_ = helper.mean_std_roi(map_pcanr_iv[...,1],maskParen_iv[...,1])
m_A,_     = helper.mean_std_roi(mappA_iv[...,1],maskParen_iv[...,1])
m_B,_     = helper.mean_std_roi(mappB_iv[...,1],maskParen_iv[...,1])

id_iv   = [7,20,100,111]
vmax_p  = [300,400,900,1000]
vmax_w  = [500,500,500,500]
row,col = 5, 4
plt.figure(figsize=(30,20))
plt.subplot(row,col,1),plt.axis('off'),plt.title('T2*w image',loc='left')
plt.imshow(wImg_iv[id_iv[0],:,:,0]*maskBody_iv[id_iv[0],...,1],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w[0]),plt.title('Normal'),plt.colorbar(fraction=0.022)
plt.subplot(row,col,2),plt.axis('off')
plt.imshow(wImg_iv[id_iv[1],:,:,0]*maskBody_iv[id_iv[1],...,1],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w[1]),plt.title('Mild'),plt.colorbar(fraction=0.022)
plt.subplot(row,col,3),plt.axis('off')
plt.imshow(wImg_iv[id_iv[2],:,:,0]*maskBody_iv[id_iv[2],...,1],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w[2]),plt.title('Moderate'),plt.colorbar(fraction=0.022)
plt.subplot(row,col,4),plt.axis('off')
plt.imshow(wImg_iv[id_iv[3],:,:,0]*maskBody_iv[id_iv[3],...,1],cmap='gray',interpolation='none',vmin=0,vmax=vmax_w[3]),plt.title('Severe'),plt.colorbar(fraction=0.022)

plt.subplot(row,col,5),plt.axis('off'),plt.title('M$_1$NCM',loc='left'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[0]])),loc='right')
plt.imshow(map_m1ncm_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,6),plt.axis('off'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[1]])),loc='right')
plt.imshow(map_m1ncm_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,7),plt.axis('off'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[2]])),loc='right')
plt.imshow(map_m1ncm_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,8),plt.axis('off'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[3]])),loc='right')
plt.imshow(map_m1ncm_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.subplot(row,col,9),plt.axis('off'),plt.title('PCANR',loc='left'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[0]])),loc='right')
plt.imshow(map_pcanr_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,10),plt.axis('off'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[1]])),loc='right')
plt.imshow(map_pcanr_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,11),plt.axis('off'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[2]])),loc='right')
plt.imshow(map_pcanr_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,12),plt.axis('off'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[3]])),loc='right')
plt.imshow(map_pcanr_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.subplot(row,col,13),plt.axis('off'),plt.title(model_nameA,loc='left'),plt.title('R2* = '+str(np.round(m_A[id_iv[0]])),loc='right')
plt.imshow(mappA_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,14),plt.axis('off'),plt.title('R2* = '+str(np.round(m_A[id_iv[1]])),loc='right')
plt.imshow(mappA_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,15),plt.axis('off'),plt.title('R2* = '+str(np.round(m_A[id_iv[2]])),loc='right')
plt.imshow(mappA_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,16),plt.axis('off'),plt.title('R2* = '+str(np.round(m_A[id_iv[3]])),loc='right')
plt.imshow(mappA_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.subplot(row,col,17),plt.axis('off'),plt.title(model_nameB,loc='left'),plt.title('R2* = '+str(np.round(m_B[id_iv[0]])),loc='right')
plt.imshow(mappB_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,18),plt.axis('off'),plt.title('R2* = '+str(np.round(m_B[id_iv[1]])),loc='right')
plt.imshow(mappB_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,19),plt.axis('off'),plt.title('R2* = '+str(np.round(m_B[id_iv[2]])),loc='right')
plt.imshow(mappB_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,20),plt.axis('off'),plt.title('R2* = '+str(np.round(m_B[id_iv[3]])),loc='right')
plt.imshow(mappB_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.savefig(os.path.join('figure','pImgInVivo'))

import methods
refer = np.zeros(wImg_iv.shape[0])
TEs  = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
for i in range(wImg_iv.shape[0]): refer[i] = methods.AverageThenFitting(wImg_iv[i],maskParen_iv[i,...,0],tes=TEs,model='Offset')[1]

# refer = m_m1ncm
# refer = m_pcanr
refer = m_B

# Bland-Altman Analysis
print('Bland-Altman plot Analysing...')
yLimit = 200.0*m
plt.figure(figsize=(25,5))
plt.subplot(141)
helper.bland_altman_plot(x=m_m1ncm[100:],y=refer[100:],xLimit=xLimit,yLimit=yLimit),plt.title('M$^1$NCM vs Reference')
plt.subplot(142)
helper.bland_altman_plot(x=m_pcanr[100:],y=refer[100:],xLimit=xLimit,yLimit=yLimit),plt.title('PCANR vs Reference')
plt.subplot(143)
helper.bland_altman_plot(x=m_A[100:],    y=refer[100:],xLimit=xLimit,yLimit=yLimit),plt.title(model_nameA+' vs Reference')
plt.subplot(144)
helper.bland_altman_plot(x=m_B[100:],    y=refer[100:],xLimit=xLimit,yLimit=yLimit), plt.title(model_nameB+' vs Reference')
plt.savefig(os.path.join('figure','BAplotInVivo.png'))

plt.figure(figsize=(6,6))
indx = np.argsort(refer[100:])
plt.plot(np.arange(refer[100:].shape[0]),refer[100:][indx],'o',markersize=7,color='red',label='refer')
plt.plot(np.arange(refer[100:].shape[0]),m_m1ncm[100:][indx],'o',markersize=7,color='green',label='M$^1$NCM')
plt.plot(np.arange(refer[100:].shape[0]),m_pcanr[100:][indx],'o',markersize=7,color='blue',label='PCANR')
plt.plot(np.arange(refer[100:].shape[0]),m_A[100:][indx],'o',markersize=7,color='magenta',label=model_nameA)
plt.legend()

plt.savefig(os.path.join('figure','BAplotInVivo2.png'))

