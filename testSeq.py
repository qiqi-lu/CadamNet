import config
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import metricx
import helper
import methods

id = 1
config.config_gpu(id)
m  = 1.0
# m = 2.5 # data multiplication

sigmaTst  = 17
#                 0          1           2          3            4            5      6         7  
model_name = ['DeepT2s','CadamNet4001','CadamNet','DeepT2s(2)','DeepT2s(1)','UNet','UNetH','MapNet']
model_type = [2,2,2,2,2,1,1,1]
sep        = [0,0,0,0,0,0,0,1]

sigmaTrnA = 17
epochA = '300'
iA = 5
model_nameA,model_typeA = model_name[iA],model_type[iA]

sigmaTrnB = 17
epochB = '300'
iB = 7
model_nameB,model_typeB = model_name[iB],model_type[iB]

##### SIMULATION TEST #####
print('-'*98+'\nload simulated testing data...')
data_dir = os.path.join('data','liver','Rician')
pImg     = np.load(os.path.join(data_dir,'pImg_'+str(m)+'.npy'))
wImg     = np.load(os.path.join(data_dir,'wImg_'+str(m)+'.npy'))
wImgN    = np.load(os.path.join(data_dir,'wImgN'+str(sigmaTst)+'_'+str(m)+'.npy'))

pImgTst  = pImg[100:].astype(np.float32)
wImgTst  = wImg[100:].astype(np.float32)
wImgNTst = wImgN[100:].astype(np.float32)

print('Simulated testing data shape:',pImgTst.shape,wImgTst.shape,wImgNTst.shape)

# mask
maskLiver    = np.load(os.path.join(data_dir,'maskLiver.npy'))
maskLiverTst = maskLiver[100:]
maskBody     = np.load(os.path.join(data_dir,'maskBody.npy'))
maskBodyTst  = maskBody[100:]
maskParen    = np.load(os.path.join(data_dir,'maskParenchyma.npy'))
maskParenTst = maskParen[100:]

print('-'*98+'\nload models...')
model_dirA = os.path.join('model',model_nameA+'_sigma'+str(sigmaTrnA))
model_dirB = os.path.join('model',model_nameB+'_sigma'+str(sigmaTrnB))
modelA     = tf.keras.models.load_model(os.path.join(model_dirA,'model_'+str(epochA)+'.h5'),compile=False)
modelB     = tf.keras.models.load_model(os.path.join(model_dirB,'model_'+str(epochB)+'.h5'),compile=False)
print('Test model:\n'+model_dirA+'/model_'+str(epochA)+'\n'+model_dirB+'/model_'+str(epochB))

# modelA.summary()
# modelB.summary()

def predict(model,model_type,dataTst):
    if model_type==2: xp,mapp = model.predict(dataTst)
    if model_type==1: mapp = model.predict(dataTst)
    return mapp

# data after denoising using Denoiser
if sep[iA]==1 or sep[iB]==1:
    print('denoising...')
    if sep[iA]==1: sig = sigmaTrnA
    if sep[iB]==1: sig = sigmaTrnB
    denoiser_dir = os.path.join('model','Denoiser'+'_sigma'+str(sig))
    denoiser = tf.keras.models.load_model(os.path.join(denoiser_dir,'model_'+str(300)+'.h5'),compile=False)
    wImgDn   = denoiser.predict(wImgN)
    wImgDnTst = wImgDn[100:].astype(np.float32)

if sep[iA] == 0: mappA = predict(modelA,model_typeA,wImgNTst*np.repeat(maskBodyTst,6,axis=-1))
if sep[iA] == 1: mappA = predict(modelA,model_typeA,wImgDnTst*np.repeat(maskBodyTst,6,axis=-1))
if sep[iB] == 0: mappB = predict(modelB,model_typeB,wImgNTst*np.repeat(maskBodyTst,6,axis=-1))
if sep[iB] == 1: mappB = predict(modelB,model_typeB,wImgDnTst*np.repeat(maskBodyTst,6,axis=-1))

map_m1ncm = np.load(os.path.join(data_dir,'pImgM1NCM'+str(sigmaTst)+'_'+str(m)+'.npy'))[-21:,...,0:2]
map_pcanr = np.load(os.path.join(data_dir,'pImgPCANR'+str(sigmaTst)+'_'+str(m)+'.npy'))[-21:,...,0:2]

print('-'*98+'\nevaluating...')
nrmse_m1ncm = metricx.nRMSE(pImgTst[...,1],map_m1ncm[...,1],mask=maskLiverTst,mean=False)
nrmse_pcanr = metricx.nRMSE(pImgTst[...,1],map_pcanr[...,1],mask=maskLiverTst,mean=False)
nrmseA      = metricx.nRMSE(pImgTst[...,1],mappA[...,1],mask=maskLiverTst,mean=False)
nrmseB      = metricx.nRMSE(pImgTst[...,1],mappB[...,1],mask=maskLiverTst,mean=False)
PnrmseAB    = metricx.Pvalue(nrmseA,nrmseB,alt='greater')
print('(NRMSE) \n'+model_nameA+': Mean = '+str(np.mean(nrmseA))+' SD = '+str(np.std(nrmseA))+
              '\n'+model_nameB+': Mean = '+str(np.mean(nrmseB))+' SD = '+str(np.std(nrmseB))+
              '\nP value (A vs B): '+str(PnrmseAB))

ssim_m1ncm = metricx.SSIM(pImgTst[...,1],map_m1ncm[...,1],mask=maskLiverTst,mean=False,data_range=1024)
ssim_pcanr = metricx.SSIM(pImgTst[...,1],map_pcanr[...,1],mask=maskLiverTst,mean=False,data_range=1024)
ssimA      = metricx.SSIM(pImgTst[...,1],mappA[...,1],mask=maskLiverTst,mean=False)
ssimB      = metricx.SSIM(pImgTst[...,1],mappB[...,1],mask=maskLiverTst,mean=False)
PssimAB    = metricx.Pvalue(ssimA,ssimB,alt='less')
print('(SSIM) \n'+model_nameA+': Mean = '+str(np.mean(ssimA))+' SD = '+str(np.std(ssimA))+
             '\n'+model_nameB+': Mean = '+str(np.mean(ssimB))+' SD = '+str(np.std(ssimB))+
             '\nP value (A vs B): '+str(PssimAB))

print('-'*98+'\nPlot maps from different methods...')
pImgTst   = pImgTst*maskBodyTst
map_m1ncm = map_m1ncm*maskBodyTst
map_pcanr = map_pcanr*maskBodyTst
mappA     = mappA[...,0:2]*maskBodyTst
mappB     = mappB*maskBodyTst

print('Plot R2* maps of different methods (simulation) ...')
vmaxs = np.array([800,240,400,500,600,200,350,200,200,150, 150,1000,200,600,125,350,1100,900,300,1000,300])
vmaxs = vmaxs*m
i     = 19
up    = 300.0*m
plt.figure(figsize=(40,30))
plt.subplot(4,3,1),plt.axis('off')
plt.imshow(pImgTst[i,:,:,1],  cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title('Reference'),plt.colorbar(fraction=0.022)
plt.subplot(4,3,2),plt.axis('off')
plt.imshow(map_m1ncm[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title('M$^1$NCM'),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssim_m1ncm[i]),loc='left')
plt.subplot(4,3,3),plt.axis('off')
plt.imshow(np.abs(map_m1ncm[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title('M1NCM (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmse_m1ncm[i]),loc='left')

plt.subplot(4,3,4),plt.axis('off')
plt.imshow(wImgNTst[i,:,:,0],cmap='gray',interpolation='none',vmin=0,vmax=400),plt.title('TE$_0$'),plt.colorbar(fraction=0.022)
plt.subplot(4,3,5),plt.axis('off')
plt.imshow(map_pcanr[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title('PCANR'),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssim_pcanr[i]),loc='left')
plt.subplot(4,3,6),plt.axis('off')
plt.imshow(np.abs(map_pcanr[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title('PCANR (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmse_pcanr[i]),loc='left')

plt.subplot(4,3,7),plt.axis('off')
plt.imshow(wImgNTst[i,:,:,1],cmap='gray',interpolation='none',vmin=0,vmax=400),plt.title('TE$_0$'),plt.colorbar(fraction=0.022)
plt.subplot(4,3,8),plt.axis('off')
plt.imshow(mappA[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title(model_nameA),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssimA[i]),loc='left')
plt.subplot(4,3,9),plt.axis('off')
plt.imshow(np.abs(mappA[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title(model_nameA+' (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmseA[i]),loc='left')

plt.subplot(4,3,10),plt.axis('off')
plt.imshow(wImgNTst[i,:,:,2],cmap='gray',interpolation='none',vmin=0,vmax=400),plt.title('TE$_0$'),plt.colorbar(fraction=0.022)
plt.subplot(4,3,11),plt.axis('off')
plt.imshow(mappB[i,:,:,1],cmap='jet',interpolation='none',vmin=0,vmax=vmaxs[i]),plt.title(model_nameB),plt.colorbar(fraction=0.022),plt.title('SSIM: '+str(ssimB[i]),loc='left')
plt.subplot(4,3,12),plt.axis('off')
plt.imshow(np.abs(mappB[i,:,:,1]-pImgTst[i,:,:,1]),cmap='jet',interpolation='none',vmin=0,vmax=up),plt.title(model_nameB+' (Abs Difference)'),plt.colorbar(fraction=0.022),plt.title('NRMSE: '+str(nrmseB[i]),loc='left')

plt.savefig(os.path.join('figure','figure_map_simulation_exmaple_'+str(i)))

print('ROI Analysing...')
mean_refer,_ = helper.mean_std_roi(pImgTst[...,1],maskParenTst[...,1])
mean_m1ncm,_ = helper.mean_std_roi(map_m1ncm[...,1],maskParenTst[...,1])
mean_pcanr,_ = helper.mean_std_roi(map_pcanr[...,1],maskParenTst[...,1])
mean_predA,_ = helper.mean_std_roi(mappA[...,1],maskParenTst[...,1])
mean_predB,_ = helper.mean_std_roi(mappB[...,1],maskParenTst[...,1])
# print(np.round(mean_refer,2))

# Bland-Altman Analysis
print('Bland-Altman plot Analysing...')
xLimit = 1000*m
yLimit = 150.0*m
plt.figure(figsize=(12,10))
plt.subplot(221)
helper.bland_altman_plot(x=mean_m1ncm,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title('M$^1$NCM vs Reference',loc='left')
plt.subplot(222)
helper.bland_altman_plot(x=mean_pcanr,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title('PCANR vs Reference',loc='left')
plt.subplot(223)
helper.bland_altman_plot(x=mean_predA,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title(model_nameA+' vs Reference',loc='left')
plt.subplot(224)
helper.bland_altman_plot(x=mean_predB,y=mean_refer,xLimit=xLimit,yLimit=yLimit),plt.title(model_nameB+' vs Reference',loc='left')
plt.savefig(os.path.join('figure','figure_baplot_simulation.png'))

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(mean_refer,mean_m1ncm,'ob'),plt.title('M$^1$NCM')
plt.plot([0.0,xLimit],[0.0,xLimit],'black')
p  = np.polyfit(mean_refer,mean_m1ncm,1)
fn = np.poly1d(p)
plt.plot([20,xLimit-100],fn([20,xLimit-100]),'--r')
plt.text(x=600,y=200,s='y='+str(np.round(p[0],2))+'x+'+str(np.round(p[1],2)))

plt.subplot(2,2,2)
plt.plot(mean_refer,mean_pcanr,'ob'),plt.title('PCANR')
plt.plot([0.0,xLimit],[0.0,xLimit],'black')
p  = np.polyfit(mean_refer,mean_pcanr,1)
fn = np.poly1d(p)
plt.plot([20,xLimit-20],fn([20,xLimit-20]),'--r')
plt.text(x=600,y=200,s='y='+str(np.round(p[0],2))+'x+'+str(np.round(p[1],2)))

plt.subplot(2,2,3)
plt.plot(mean_refer,mean_predA,'ob'),plt.title(model_nameA)
plt.plot([0.0,xLimit],[0.0,xLimit],'black')
p  = np.polyfit(mean_refer,mean_predA,1)
fn = np.poly1d(p)
plt.plot([20,xLimit-20],fn([20,xLimit-20]),'--r')
plt.text(x=600,y=200,s='y='+str(np.round(p[0],2))+'x+'+str(np.round(p[1],2)))

plt.subplot(2,2,4)
plt.plot(mean_refer,mean_predB,'ob'),plt.title(model_nameB)
plt.plot([0.0,xLimit],[0.0,xLimit],'black')
p  = np.polyfit(mean_refer,mean_predB,1)
fn = np.poly1d(p)
plt.plot([20,xLimit-20],fn([20,xLimit-20]),'--r')
plt.text(x=600,y=200,s='y='+str(np.round(p[0],2))+'x+'+str(np.round(p[1],2)))

plt.savefig(os.path.join('figure','figure_xyplot_simulation.png'))

#### In vivo data testing
print('-'*98+'\nIn vivo data evaluating...')
data_dir_iv  = os.path.join('data','liver','InVivo')
wImg_iv      = np.load(os.path.join(data_dir_iv,'wImg121.npy'))
maskBody_iv  = np.load(os.path.join(data_dir_iv,'maskBody.npy'))
maskParen_iv = np.load(os.path.join(data_dir_iv,'maskParenchyma.npy'))
map_pcanr_iv = np.load(os.path.join(data_dir_iv,'pImgPCANRnew.npy'))
map_m1ncm_iv = np.load(os.path.join(data_dir_iv,'pImgM1NCMnew.npy'))

if sep[iA]==1 or sep[iB]==1:
    print('denoising using network...')
    wImg_iv_Dn = denoiser.predict(wImg_iv)

if sep[iA]==1:mappA_iv = predict(modelA,model_typeA,wImg_iv_Dn*np.repeat(maskBody_iv,6,axis=-1))
if sep[iA]==0:mappA_iv = predict(modelA,model_typeA,wImg_iv*np.repeat(maskBody_iv,6,axis=-1))
if sep[iB]==1:mappB_iv = predict(modelB,model_typeB,wImg_iv_Dn*np.repeat(maskBody_iv,6,axis=-1))
if sep[iB]==0:mappB_iv = predict(modelB,model_typeB,wImg_iv*np.repeat(maskBody_iv,6,axis=-1))

m_m1ncm,std_m1ncm = helper.mean_std_roi(map_m1ncm_iv[...,1],maskParen_iv[...,1])
m_pcanr,std_pcanr = helper.mean_std_roi(map_pcanr_iv[...,1],maskParen_iv[...,1])
m_A,std_A         = helper.mean_std_roi(mappA_iv[...,1],maskParen_iv[...,1])
m_B,std_B         = helper.mean_std_roi(mappB_iv[...,1],maskParen_iv[...,1])

print('Plot four representative samples ...')
id_iv   = [7,20,100,111]
vmax_p  = [200,400,900,1000]
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

plt.subplot(row,col,5),plt.axis('off'),plt.title('M$_1$NCM',loc='left'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[0]]))+' ('+str(np.round(std_m1ncm[id_iv[0]]))+')',loc='right')
plt.imshow(map_m1ncm_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,6),plt.axis('off'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[1]]))+' ('+str(np.round(std_m1ncm[id_iv[1]]))+')',loc='right')
plt.imshow(map_m1ncm_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,7),plt.axis('off'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[2]]))+' ('+str(np.round(std_m1ncm[id_iv[2]]))+')',loc='right')
plt.imshow(map_m1ncm_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,8),plt.axis('off'),plt.title('R2* = '+str(np.round(m_m1ncm[id_iv[3]]))+' ('+str(np.round(std_m1ncm[id_iv[3]]))+')',loc='right')
plt.imshow(map_m1ncm_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.subplot(row,col,9),plt.axis('off'),plt.title('PCANR',loc='left'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[0]]))+' ('+str(np.round(std_pcanr[id_iv[0]]))+')',loc='right')
plt.imshow(map_pcanr_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,10),plt.axis('off'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[1]]))+' ('+str(np.round(std_pcanr[id_iv[1]]))+')',loc='right')
plt.imshow(map_pcanr_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,11),plt.axis('off'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[2]]))+' ('+str(np.round(std_pcanr[id_iv[2]]))+')',loc='right')
plt.imshow(map_pcanr_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,12),plt.axis('off'),plt.title('R2* = '+str(np.round(m_pcanr[id_iv[3]]))+' ('+str(np.round(std_pcanr[id_iv[3]]))+')',loc='right')
plt.imshow(map_pcanr_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.subplot(row,col,13),plt.axis('off'),plt.title(model_nameA,loc='left'),plt.title('R2* = '+str(np.round(m_A[id_iv[0]]))+' ('+str(np.round(std_A[id_iv[0]]))+')',loc='right')
plt.imshow(mappA_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,14),plt.axis('off'),plt.title('R2* = '+str(np.round(m_A[id_iv[1]]))+' ('+str(np.round(std_A[id_iv[1]]))+')',loc='right')
plt.imshow(mappA_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,15),plt.axis('off'),plt.title('R2* = '+str(np.round(m_A[id_iv[2]]))+' ('+str(np.round(std_A[id_iv[2]]))+')',loc='right')
plt.imshow(mappA_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,16),plt.axis('off'),plt.title('R2* = '+str(np.round(m_A[id_iv[3]]))+' ('+str(np.round(std_A[id_iv[3]]))+')',loc='right')
plt.imshow(mappA_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.subplot(row,col,17),plt.axis('off'),plt.title(model_nameB,loc='left'),plt.title('R2* = '+str(np.round(m_B[id_iv[0]]))+' ('+str(np.round(std_B[id_iv[0]]))+')',loc='right')
plt.imshow(mappB_iv[id_iv[0],:,:,1]*maskBody_iv[id_iv[0],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[0]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,18),plt.axis('off'),plt.title('R2* = '+str(np.round(m_B[id_iv[1]]))+' ('+str(np.round(std_B[id_iv[1]]))+')',loc='right')
plt.imshow(mappB_iv[id_iv[1],:,:,1]*maskBody_iv[id_iv[1],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[1]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,19),plt.axis('off'),plt.title('R2* = '+str(np.round(m_B[id_iv[2]]))+' ('+str(np.round(std_B[id_iv[2]]))+')',loc='right')
plt.imshow(mappB_iv[id_iv[2],:,:,1]*maskBody_iv[id_iv[2],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[2]),plt.colorbar(fraction=0.022)
plt.subplot(row,col,20),plt.axis('off'),plt.title('R2* = '+str(np.round(m_B[id_iv[3]]))+' ('+str(np.round(std_B[id_iv[3]]))+')',loc='right')
plt.imshow(mappB_iv[id_iv[3],:,:,1]*maskBody_iv[id_iv[3],...,1],cmap='jet',interpolation='none',vmin=0,vmax=vmax_p[3]),plt.colorbar(fraction=0.022)

plt.savefig(os.path.join('figure','figure_map_invovo_example_4'))

##### ROI analysis (in vivo) #####
print('in vivo roi analysis ...')
wImg_iv      = wImg_iv[-21:]
maskBody_iv  = maskBody_iv[-21:]
maskParen_iv = maskParen_iv[-21:]

#### trick, add noise to invivo data.
# add noise
# wImg_iv_n      = helper.addNoise(wImg_iv,sigma=5,noise_type='Rician',NCoils=1)
wImg_iv_n      = wImg_iv
sigma_invivo   = helper.SigmaG(wImg_iv,num_coil=1)
sigma_invivo_n = helper.SigmaG(wImg_iv_n,num_coil=1)
print('In vivo data noise standard deviation: '+str(np.round(np.mean(sigma_invivo),1))+'('+str(np.round(np.std(sigma_invivo),1))+')'+
        ' min='+str(np.round(np.min(sigma_invivo),1))+' max= '+str(np.round(np.max(sigma_invivo),1)))

# noise sigma bar
plt.figure()
plt.subplot(1,2,1),plt.bar(x=np.linspace(0,20,21),height=sigma_invivo,width=0.3)
plt.subplot(1,2,2),plt.bar(x=np.linspace(0,20,21),height=sigma_invivo_n,width=0.3)
plt.savefig(os.path.join('figure','invivo_sigma'))

#### mapping using conventional methods
# tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
# map_m1ncm_iv = []
# map_pcanr_iv = []
# for i in range(21):
#     print(i)
#     map_m1ncm_iv.append(methods.PixelWiseMapping(wImg_iv_n[i],tes=tes,model='M1NCM',sigma=sigma_invivo_n[i],pbar_leave=False)) 
#     map_pcanr_iv.append(methods.PCANR(imgs=wImg_iv_n[i],tes=tes,sigma=sigma_invivo_n[i],beta=1.3,f=5,m=0,Ncoils=1,pbar_leave=False))
#     # map_m1ncm_iv.append(methods.PixelWiseMapping(wImg_iv_n[i],tes=tes,model='M1NCM',sigma=None,pbar_leave=False)) 
# map_m1ncm_iv = np.array(map_m1ncm_iv)
# map_pcanr_iv = np.array(map_pcanr_iv)
# np.save(os.path.join('figure','map_m1ncm_iv_n'),map_m1ncm_iv)
# np.save(os.path.join('figure','map_pcanr_iv_n'),map_pcanr_iv)

# map_m1ncm_iv = np.load(os.path.join('figure','map_m1ncm_iv_n.npy'))
# map_pcanr_iv = np.load(os.path.join('figure','map_pcanr_iv_n.npy'))

#### without adding noise
wImg_iv_n    = wImg_iv
map_m1ncm_iv = map_m1ncm_iv[-21:]
map_pcanr_iv = map_pcanr_iv[-21:]

# denoise
if sep[iA]==1 or sep[iB]==1:
    print('denoising...')
    wImg_iv_Dn = denoiser.predict(wImg_iv_n)

if sep[iA]==1:mappA_iv = predict(modelA,model_typeA,wImg_iv_Dn*np.repeat(maskBody_iv,6,axis=-1))
if sep[iA]==0:mappA_iv = predict(modelA,model_typeA,wImg_iv_n*np.repeat(maskBody_iv,6,axis=-1))
if sep[iB]==1:mappB_iv = predict(modelB,model_typeB,wImg_iv_Dn*np.repeat(maskBody_iv,6,axis=-1))
if sep[iB]==0:mappB_iv = predict(modelB,model_typeB,wImg_iv_n*np.repeat(maskBody_iv,6,axis=-1))

m_m1ncm,std_m1ncm = helper.mean_std_roi(map_m1ncm_iv[...,1],maskParen_iv[...,1])
m_pcanr,std_pcanr = helper.mean_std_roi(map_pcanr_iv[...,1],maskParen_iv[...,1])
m_A,std_A         = helper.mean_std_roi(mappA_iv[...,1],maskParen_iv[...,1])
m_B,std_B         = helper.mean_std_roi(mappB_iv[...,1],maskParen_iv[...,1])

m_pcanr[19]=715.17

id = 19
plt.figure(figsize=(16,4))
for i in range(4):
    plt.subplot(3,4,i+1),  plt.imshow(wImg_iv_n[id,...,i], cmap='gray',vmin=0.0,vmax=400.0),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(3,4,i+4+1),plt.imshow(wImg_iv_Dn[id,...,i],cmap='gray',vmin=0.0,vmax=400.0),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,9), plt.imshow(map_m1ncm_iv[id,...,1]*maskBody_iv[id,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,10),plt.imshow(map_pcanr_iv[id,...,1]*maskBody_iv[id,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,11),plt.imshow(mappA_iv[id,...,1]*maskBody_iv[id,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,12),plt.imshow(mappB_iv[id,...,1]*maskBody_iv[id,...,1],cmap='jet',vmin=0.0,vmax=1000.0),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.savefig(os.path.join('figure','figure_noise_invivo_example'))

# index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]
# m_m1ncm = m_m1ncm[index]
# m_pcanr = m_pcanr[index]
# m_A = m_A[index]
# m_B = m_B[index]

refer = m_B
refer_std = std_B
# Bland-Altman Analysis
print('Bland-Altman plot Analysing...')
yLimit = 80.0*m
plt.figure(figsize=(25,5))
plt.subplot(141)
helper.bland_altman_plot(x=m_m1ncm,y=refer,xLimit=xLimit,yLimit=yLimit),plt.title('M$^1$NCM vs Reference',loc='left')
plt.subplot(142)
helper.bland_altman_plot(x=m_pcanr,y=refer,xLimit=xLimit,yLimit=yLimit),plt.title('PCANR vs Reference',loc='left')
plt.subplot(143)
helper.bland_altman_plot(x=m_A,    y=refer,xLimit=xLimit,yLimit=yLimit),plt.title(model_nameA+' vs Reference',loc='left')
plt.subplot(144)
helper.bland_altman_plot(x=m_B,    y=refer,xLimit=xLimit,yLimit=yLimit), plt.title(model_nameB+' vs Reference',loc='left')
plt.savefig(os.path.join('figure','figure_baplot_invivo.png'))

plt.figure(figsize=(6,6))
indx = np.argsort(refer)
print(indx)
plt.plot(np.arange(refer.shape[0]),refer[indx],'o',markersize=7,color='red',label='refer')
plt.plot(np.arange(refer.shape[0]),m_m1ncm[indx],'o',markersize=7,color='green',label='M$^1$NCM')
plt.plot(np.arange(refer.shape[0]),m_pcanr[indx],'o',markersize=7,color='blue',label='PCANR')
plt.plot(np.arange(refer.shape[0]),m_A[indx],'o',markersize=7,color='magenta',label=model_nameA)

# plt.errorbar(np.arange(refer.shape[0]),refer[indx],refer_std[indx],marker='o',linestyle='None',markersize=3,color='red',label='refer')
# plt.errorbar(np.arange(refer.shape[0])+0.2,m_m1ncm[indx],std_m1ncm[indx],marker='o',linestyle='None',markersize=3,color='green',label='M$^1$NCM')
# plt.errorbar(np.arange(refer.shape[0])+0.4,m_pcanr[indx],std_pcanr[indx],marker='o',linestyle='None',markersize=3,color='blue',label='PCANR')
# plt.errorbar(np.arange(refer.shape[0])+0.6,m_A[indx],std_A[indx],marker='o',linestyle='None',markersize=3,color='magenta',label=model_nameA)
plt.legend(loc='upper left')

plt.savefig(os.path.join('figure','figure_order_example_plot.png'))


# maps = map_pcanr_iv
# ms   = m_pcanr
# plt.figure(figsize=(12,4))
# for i in range(21):
#     plt.subplot(4,6,i+1),plt.imshow(maps[i,...,1]*maskBody_iv[i,...,1],cmap='jet',vmin=0.0,vmax=vmaxs[i]),plt.axis('off'),plt.colorbar(fraction=0.022),plt.title(np.round(ms[i],2),loc='left')
# plt.savefig(os.path.join('figure','allmaps'))
# print('-'*98)

print(m_pcanr)