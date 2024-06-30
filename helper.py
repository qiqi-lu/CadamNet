# import skimage
import tensorflow as tf
import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import struct
import cv2


def findLastCheckpoint(save_dir):
    """
    Find the last saved model number.
    ### ARGUMENTS
    - save_dir, dir of the saved model.

    ### RETURN
    - initial_epoch, the number of last saved model.
    """
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

def makePairedData(para_maps,tes,sigma,noise_type='Rician',NCoils=1):
    """
    From parameter maps to magnitude images, and add noise.
    ## RETURN
    magnitude images, noisy magnitude images.
    """
    type=['Rician', 'Gaussian','ChiSquare']
    assert noise_type in type, 'Unsupported noise type.'

    if len(para_maps.shape)==3:
        para_maps=para_maps[np.newaxis]
    
    nte=tes.shape[-1]
    in_shape = para_maps.shape
    s0_vec = np.reshape(para_maps[...,0],[-1,1]) 
    r2_vec = np.reshape(para_maps[...,1],[-1,1])
    s=s0_vec*np.exp(-1*tes/1000.0*r2_vec)
    imgs   = np.reshape(s,[in_shape[-4],in_shape[-3],in_shape[-2],nte]) # may need a batch dim
    if sigma>0:
        imgs_n = addNoise(imgs=imgs,sigma=sigma,noise_type=noise_type,NCoils=NCoils)
    else:
        imgs_n = imgs
    return imgs, imgs_n

def addNoise(imgs,sigma,noise_type='Rician',NCoils=1):
    """
    Add nosie to the inputs.
    ### ARGUMENTS
    - imgs, the image to be added noise.
    - sigma, noise sigma.
    - noise_type, type of noise.

    ### RETURN
    - img_n: images with noise.
    """
    type = ['Rician', 'Gaussian','ChiSquare']
    assert sigma>0, 'Noise sigma must higher than 0.'
    assert noise_type in type, 'Unsupported noise type.'
    assert NCoils > 0, 'Coils number should larger than 0.'

    print('Add '+noise_type+' noise ('+str(sigma)+')...')
    if noise_type == 'Gaussian':
        imgs_n = imgs+np.random.normal(loc=0,scale=sigma,size=imgs.shape)
    if noise_type == 'Rician':
        r = imgs+np.random.normal(loc=0,scale=sigma,size=imgs.shape)
        i = np.random.normal(loc=0,scale=sigma,size=imgs.shape)
        imgs_n=np.sqrt(r**2+i**2)
    if noise_type == 'ChiSquare':
        imgs = imgs/np.sqrt(NCoils)
        imgs_n = np.zeros(imgs.shape)
        for i in range(NCoils):
            r = imgs+np.random.normal(loc=0,scale=sigma,size=imgs.shape)
            i = np.random.normal(loc=0,scale=sigma,size=imgs.shape)
            imgs_n=imgs_n+r**2+i**2
        imgs_n = np.sqrt(imgs_n)
    return imgs_n

def addNoiseMix(imgs,sigma_low,sigma_high,noise_type='Rician'):
    """
    Add nosie to the inputs.
    ### ARGUMENTS
    - imgs, the image to be added noise.
    - sigma, noise sigma.
    - noise_type, type of noise.

    ### RETURN
    - img_n: images with noise.
    """
    type = ['Rician', 'Gaussian']
    assert noise_type in type, 'Unsupported noise type.'
    print('Add '+noise_type+' noise (mixed)...')
    n = imgs.shape[0]
    imgs_n = []
    pbar = tqdm.tqdm(total=n,desc='AddNoise: ')
    for i in range(n):
        pbar.update(1)
        sigma = np.random.choice([1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0])
        img=imgs[i]
        if noise_type == 'Gaussian':
            img_n = img+np.random.normal(loc=0,scale=sigma,size=img.shape)
        if noise_type == 'Rician':
            r = img+np.random.normal(loc=0,scale=sigma,size=img.shape)
            i = np.random.normal(loc=0,scale=sigma,size=img.shape)
            img_n=np.sqrt(r**2+i**2)
        imgs_n.append(img_n)
    pbar.close()
    imgs_n = np.array(imgs_n)
    return imgs_n

def makeBlockImage(img_size=(5,5),block_size=(5,5),type='Random',value=None):
    """
    Make a block like image.
    ### ARGUMENTS
    - img_size, num of blocks.
    - block_size, size of block.
    - type, type of value get.
        - `Random`: value=[low,high], default: [0,1000], random (uniform) value for each block.
        - `UserDefined`: value=[...], user defiened value for each block.
    
    ### RETURN
    - image_blcok, created block image.
    - values, values correspond to each block, with a shape of [img_size].
    """
    types = ['Random', 'UserDefined']
    assert type in types, 'Unsupported type.'

    if type=='UserDefined':
        assert value!=None & len(value)==img_size[0]*img_size[1], 'Inconsistent between img_size and value shape.'
        values=value

    if type=='Random':
        if value==None:
            value=[0,1000]
            values=np.random.uniform(value[0],value[1],size=img_size[0]*img_size[1]) # values for each block
        if value!=None:
            assert value.shape[-1]==2,'Value should be [low,high].'
            values=np.random.uniform(value[0],value[1],size=img_size[0]*img_size[1]) # values for each block

    values = np.reshape(values,img_size)
    img    = np.ones([img_size[0],img_size[1],block_size[0],block_size[1]])
    for i in range(0,img_size[0]):
        block_row=img[i,0,:,:]*values[i,0]
        for j in range(1,img_size[1]):
            block_row=np.hstack((block_row,img[i,j,:,:]*values[i,j]))
        if i ==0:
            img_block=block_row
        else:
            img_block=np.vstack((img_block,block_row))
    return img_block, values

def LogLinearN(imgs,TEs,n=0):
    """
    Log-Linear Method.

    Perfrom a pixel-wise linear fit of the decay curve after a log transofrmation (using the first `n` data point).
    ### ARGUMENTS
    - imgs, shape = [batch,w,h,c] or [w,h,c]
    - TEs : Echo Time (ms)
    - n: number of data poitn to caluclate.

    ### RETURN
    - maps : parameter maps [batch,h,w,2].
    """
    assert len(imgs.shape)==4 or 3, 'Data with shape of [batch,w,h,c] or [w,h,c] is needed.'
    assert imgs.shape[-1]==TEs.shape[-1], 'The TEs and the data shape is confliting.'

    if len(imgs.shape)==3: imgs=imgs[np.newaxis]
    if n==0: n=TEs.shape[-1]

    imgs_v=np.reshape(imgs,[-1,imgs.shape[-1]])+1e-7 # images point vector with all data point
    imgs_v=np.abs(imgs_v)
    x=TEs[0:n]/1000.0
    y=np.log(imgs_v[:,0:n]) # logsignal

    x_mean = np.mean(x)
    y_mean = np.reshape(np.mean(y,axis=1),[-1,1])
    w = np.reshape(np.sum((x-x_mean)*(y-y_mean),axis=1)/np.sum((x-x_mean)**2),[-1,1])
    b = y_mean-w*x_mean

    r2_v = -w[:,0]
    s0_v = np.exp(b)[:,0]

    map_v=np.zeros(shape=(imgs_v.shape[0],2))
    map_v[:,0]=np.abs(s0_v)
    map_v[:,1]=np.abs(r2_v)
    maps=np.reshape(map_v,[imgs.shape[0],imgs.shape[1],imgs.shape[2],2])
    return maps

def checkNaN(data):
    """
    Check whether a NaN in data.
    """
    import pandas as pd
    if len(data.shape)!=2: data=np.reshape(data,[-1,1])
    df=pd.DataFrame(data)
    print(df.isnull().any(axis=0))

def makePatch(data,patch_size=32,stride=8,rescale=False,aug_times=8):
    """
    Patching data.
    ### ARGUMENTS
    - data, shape = [batch,h,w,c] or [h,w,c].
    - patch_size, patch size.
    - stride, patch moving step.
    - rescale, whether ot rescale images.
    - aug_times, the time of augmentation.

    ### RETURN
    - patch, patches with shape of [n,patch_size,patch_size,c]
    """
    print('Patching...')
    import cv2

    if len(data.shape)==3: data=data[np.newaxis]
    if rescale == True: scales = [1, 0.9, 0.8, 0.7]
    if rescale == False: scales = [1]

    n,h,w,c=data.shape
    patches = []
    for id in range(n):
        for s in scales:
            # rescale image size
            h_scaled, w_scaled = int(h*s),int(w*s)
            img_rescaled = cv2.resize(data[id], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches
            for i in range(0, h_scaled-patch_size+1, stride):
                for j in range(0, w_scaled-patch_size+1, stride):
                    x = img_rescaled[i:i+patch_size, j:j+patch_size,:]
                    # data aug
                    for k in range(0, aug_times):
                        x_aug = data_aug(x, mode=k)
                        patches.append(x_aug)
    patch = np.array(patches)
    print('Patches shape: ',patch.shape)
    return patch

def data_aug(img, mode=0):
    """
    Data augmentation operator.
    ### ARGUMEMTS
    - img, shape=[h,w,c]
    - mode, mode of operation.
    
    ### RETURN
    Image after flip or rotation with the same size of img.
    """
    assert len(img.shape)!=4, 'Unsupported data shape when augmenting.'
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img,axes=(0,1))
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        return img

def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch<=100:
        lr = initial_lr
    elif epoch<=200:
        lr = initial_lr/2
    elif epoch<=300:
        lr = initial_lr/10 
    else:
        lr = initial_lr/20 
    return lr

def lr_schedule2(epoch):
    initial_lr = 0.0005
    if epoch<=100:
        lr = initial_lr
    elif epoch<=200:
        lr = initial_lr/2
    elif epoch<=300:
        lr = initial_lr/4 
    else:
        lr = initial_lr/8 
    return lr

def lr_schedule3(epoch):
    initial_lr = 0.0001
    if epoch<=50:
        lr = initial_lr
    elif epoch<=75:
        lr = initial_lr/2
    elif epoch<=100:
        lr = initial_lr/4 
    else:
        lr = initial_lr/8 
    return lr

def lr_schedule4(epoch):
    initial_lr = 0.001
    if epoch<=100:
        lr = initial_lr
    elif epoch<=150:
        lr = initial_lr/2
    elif epoch<=200:
        lr = initial_lr/4 
    else:
        lr = initial_lr/8 
    return lr

def lr_schedule5(epoch):
    initial_lr = 0.001
    if epoch<=200:
        lr = initial_lr
    elif epoch<=250:
        lr = initial_lr/2
    elif epoch<=300:
        lr = initial_lr/4 
    else:
        lr = initial_lr/8 
    return lr

def mean_std_roi(x,mask):
    """
    Calculate the mena and the std in ROI of the data (x).
    ### ARGUMENTS
    - x, shape = [batch, h, w]
    - mask, shape = [batch,h,w]

    ### RETURN
    - mean, shape = [batch], the mean value in ROI of each study.
    - std, shape = [batch], the std in ROI of each study.
    """
    mask = (mask+1)%2 # region of maskout (0 -> 1)
    x_masked = np.ma.array(x,mask=mask)
    mean = x_masked.mean(axis=(1,2))
    std  = x_masked.std(axis=(1,2))

    return mean, std

def bland_altman_plot(x,y,xLimit=1000.0,yLimit=200):
    """
    Bland Altman plot (x-y).
    ### ARGUMENTS
    - x, shape = [batch,h,w].
    - y, shape = [batch,h,w].
    - mask, shape = [batch,h,w].

    ### RETURN
    Plot a Bland Altman plot into figure.
    """
    mean = np.mean([x, y], axis=0)
    # print(mean)
    diff = x - y      # Difference between data1 and data2
    md   = np.mean(diff)        # Mean of the difference
    sd   = np.std(diff, axis=0) # Standard deviation of the difference

    plt.plot(mean, diff,'o',color='blue') # data point
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='black', linestyle='--')
    plt.axhline(md - 1.96*sd, color='black', linestyle='--')

    plt.title('('+str(np.round(md+1.96*sd,2))+','+str(np.round(md,2))+','+str(np.round(md-1.96*sd,2))+')',loc='right')
    plt.ylim([-yLimit,yLimit]),plt.ylabel('$R_2^*$ Difference ($s^{-1}$)')
    plt.xlim([0,xLimit]),plt.xlabel('Mean $R_2^*$ ($s^{-1}$)')

def SigmaG(data,num_coil=1,mask=None,mean=True):
    """
    Calculate the mean of the bkg in [each study, each TE]
    ### RETURN
    - sigma_g: sigma map of noise in each TE in each study. [num_study, num_TE]
        - if `mean=True`, return the sigma of each study.
    """
    if len(data.shape)==3: data=data[np.newaxis]    
    n,h,w,c = data.shape
    if mask is None:
        f=5 # roi size
        roi1 = np.zeros((n,c))
        roi2 = np.zeros((n,c))
        for i in range(n): # each study
            for j in range(c): # each TE time
                roi1[i,j] = np.mean(data[i,1:1+f,1:1+f,j]**2)
                roi2[i,j] = np.mean(data[i,1:1+f,w-f-1:w-1+f,j]**2)
        mean_bkg = np.mean([roi1,roi2],axis=0)
    if mask is not None:
        mask = (mask+1)%2
        kernel = np.ones((5,5))
        for i in range(n): mask[i] = cv2.erode(mask[i],kernel=kernel)
        mask = np.repeat(mask[...,np.newaxis],c,axis=-1)
        mean_bkg = np.sum(np.sum((data*mask)**2,axis=1),axis=1)/np.sum(np.sum(mask,axis=1),axis=1)
    sigma_g  = np.sqrt(mean_bkg/(2*num_coil))
    if mean==True: sigma_g = np.mean(sigma_g,axis=1)
    return sigma_g

def LogsPlot(log_dir,dataSaved='all',plot='validation'):
    import pandas
    logs = pandas.read_csv(os.path.join(log_dir,'log.csv'))
    if dataSaved == 'only_train':
        assert plot == 'train', 'Only training loss was saved.'
        if plot  =='train': n = range(1,len(logs.columns))
    if dataSaved =='all':
        if plot  == 'train': n = range(1,int((len(logs.columns)+1)/2))
        if plot  == 'validation': n = range(int((len(logs.columns)+1)/2), (len(logs.columns)))
    for i in n:
        col = np.array(logs[logs.columns.values[i]])[0:300]
        plt.plot(np.log10(col),label=logs.columns[i]+' '+log_dir)
    plt.ylim([7,8])
    plt.legend()

def measureR2(maps):
    assert len(maps.shape)==3, 'Need [batch, h, w] data.'
    num_study = maps.shape[0]
    R2 = []
    level = []
    x = [30,30,25,30,30,35,30,45,30,30,25,40,35,30,30,35,30,35,50,32,30,30,40,
         30,30,30,45,30,30,30,35,32,35,30,25,30,32,30,30,40,30,30,30,30,32,45,
         30,30,30,30,30,30,35,30,30,35,30,35,30,30,30,45,50,30,30,30,30,30,30,
         25,30,40,30,30,30,35,35,35,30,30,30,30,30,40,40,40,35,30,35,35,35,30,
         40,30,35,35,40,40,40,35,35,40,30,30,30,30,30,30,30,35,35,30,30,35,35,
         40,35,30,35,30,40]
    y = [40,40,40,45,45,45,45,45,40,45,45,45,45,45,45,45,40,45,45,45,30,45,40,
         45,45,45,45,45,45,45,45,45,45,20,45,45,45,45,45,45,40,45,45,45,45,45,
         40,45,45,45,45,40,45,45,40,45,45,35,45,35,40,45,20,45,45,45,45,45,45,
         50,40,45,45,40,45,40,40,40,45,45,40,45,45,45,35,40,40,30,40,40,40,40,
         40,45,40,45,20,35,45,40,45,45,35,45,45,45,45,45,45,45,45,45,20,40,40,
         40,30,35,40,40,35]
    w=5
    h=5
    
    for i in range(0,num_study):
        #### show ROI position
        # img = plt.figure()
        # ax_img = img.add_subplot(1,1,1)
        # rect = plt.Rectangle((x[i],y[i]), w, h, fill=False,edgecolor='red',linewidth=1.0)
        # ax_img.add_patch(rect)
        # plt.imshow(maps[i,:,:],vmax=1000),plt.title(i)
        # plt.savefig(os.path.join('figure','mapWithRect'))
        
        mean = np.mean(maps[i,y[i]:y[i]+h,x[i]:x[i]+w])
        R2.append(mean)
        level.append(toLevel(mean))

    return R2, level

def toLevel(R2s):
    """
    Check HIC level.
    """
    T2s = 1000.0/R2s
    if   T2s>6.3: return 0
    elif T2s>2.8: return 1
    elif T2s>1.4: return 2
    else: return 3

def readRAW(file,shape=(64,128),scale=255):
    """
    Read RAW data.
    ### ARGUMENTS
    - file: file path.
    - shape: data shape.
    - scale: rescale data.

    ### RETURN
    - data: data after reshaping.
    """
    size = os.path.getsize(file)
    with open(file,mode='rb') as f:
        data = f.read(size)
        data = struct.unpack('B'*size, data)
        data = np.reshape(np.array(data)/scale, shape)
    return data

def sigmaCorrection(wImg,sigma=0):
    print('Noise Correction.')
    if sigma == 0: sigmas = SigmaG(wImg,num_coil=1)
    if sigma != 0: sigmas = np.repeat(sigma,repeats=wImg.shape[0])
    tmp  = np.zeros(wImg.shape)
    for i in range(wImg.shape[0]): tmp[i] = wImg[i]**2-2*(sigmas[i]**2)
    tmp_p = np.sqrt(np.abs(tmp))
    tmp_n = -np.sqrt(np.abs(tmp))
    wImgC = np.where(tmp<0,tmp_n,tmp_p)
    return wImgC

def balance(wPatches,pPatches,sgPatches):
    '''
    Balance the data according to R2*.
    '''
    assert len(wPatches.shape) == 4, 'Need 4 dims data.'
    assert len(pPatches.shape) == 4, 'Need 4 dims data.'
    assert len(sgPatches.shape) == 4, 'Need 4 dims data.'
    weights  = np.mean(np.mean(pPatches[...,1],axis=-1),axis=-1)
    n       = wPatches.shape[0]
    num_aug = [-5,-9,-5,-2,1,1,3,8,9,9,9,9,9,9,9,9]
    wPatches_aug,pPatches_aug,sgPatches_aug = [],[],[]
    for i in range(n):
        m = num_aug[int(np.floor(weights[i]/100.0))]
        if m < 0: # throw some patches
            factor = np.abs(m)
            deci   = np.random.choice(factor,1)
            if deci == 0: 
                wPatches_aug.append(wPatches[i])
                pPatches_aug.append(pPatches[i])
                sgPatches_aug.append(sgPatches[i])
        if m > 0: # aug patches
            for j in range(m):
                wPatches_aug.append(data_aug(wPatches[i],mode=j))
                pPatches_aug.append(data_aug(pPatches[i],mode=j))
                sgPatches_aug.append(data_aug(sgPatches[i],mode=j))
    wPatches_aug  = np.array(wPatches_aug)
    pPatches_aug  = np.array(pPatches_aug)
    sgPatches_aug = np.array(sgPatches_aug)
    return wPatches_aug,pPatches_aug,sgPatches_aug


if __name__ == '__main__':
    # import metricx
    # tes =np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # # tes =np.array([0.80, 1.05, 1.30, 1.55, 1.80, 2.05, 2.30, 2.55, 2.80, 3.05, 3.30, 3.55]) # (Wood2005)3
    # sigma=30

    # # remake=True
    # remake=False
    # if remake:
    #     # Parameter maps
    #     size=(20,20)
    #     n=10

    #     x=np.zeros((n,size[0]*10,size[1]*10,2))
    #     for i in range(n):
    #         x[...,0],_=makeBlockImage(img_size=size,block_size=(10,10),type='Random',value=[300,400])
    #         x[...,1],_=makeBlockImage(img_size=size,block_size=(10,10),type='Random',value=[0,1000])
    #     print(x.shape)

    #     # make simulated images
    #     # imgs,imgs_n,maps=makePairedData(para_maps=x,tes=tes,sigma=sigma,noise_type='Rician')
    #     imgs,imgs_n=makePairedData(para_maps=x,tes=tes,sigma=sigma,noise_type='Gaussian')
    #     maps=x
    #     print(imgs.shape)
    #     print(imgs_n.shape)
    #     print(maps.shape)
        
    #     path = os.path.join('data','test')
    #     if not os.path.exists(path=path):
    #         os.makedirs(path)

    #     np.save(os.path.join(path,'map_'+str(sigma)),maps)
    #     np.save(os.path.join(path,'imgs_'+str(sigma)),imgs)
    #     np.save(os.path.join(path,'imgsN_'+str(sigma)),imgs_n)
    
    # if remake==False:
        
    #     maps=np.load(os.path.join('data','test','map_'+str(sigma)+'.npy'))
    #     imgs=np.load(os.path.join('data','test','imgs_'+str(sigma)+'.npy'))
    #     imgs_n=np.load(os.path.join('data','test','imgsN_'+str(sigma)+'.npy'))

    # maps_n = LogLinearN(imgs_n,tes,n=2)

    # plt.figure(figsize=(30,20))
    # vmax=400
    # i=1
    # # images without nosie
    # plt.subplot(3,4,1)
    # plt.imshow(imgs[i,:,:,0],cmap='gray',vmax=vmax,vmin=0)
    # plt.subplot(3,4,2)
    # plt.imshow(imgs[i,:,:,1],cmap='gray',vmax=vmax,vmin=0)
    # plt.subplot(3,4,3)
    # plt.imshow(imgs[i,:,:,2],cmap='gray',vmax=vmax,vmin=0)
    # plt.subplot(3,4,4)
    # plt.imshow(imgs[i,:,:,3],cmap='gray',vmax=vmax,vmin=0)
    # #  images with noise
    # plt.subplot(3,4,5)
    # plt.imshow(imgs_n[i,:,:,0],cmap='gray',vmax=vmax,vmin=0)
    # plt.subplot(3,4,6)
    # plt.imshow(imgs_n[i,:,:,1],cmap='gray',vmax=vmax,vmin=0)
    # plt.subplot(3,4,7)
    # plt.imshow(imgs_n[i,:,:,2],cmap='gray',vmax=vmax,vmin=0)
    # plt.subplot(3,4,8)
    # plt.imshow(imgs_n[i,:,:,3],cmap='gray',vmax=vmax,vmin=0)
    # # parameter maps (reference)
    # nRMSEs = metricx.nRMSE(maps[...,0],maps_n[...,0])
    # nRMSEr = metricx.nRMSE(maps[...,1],maps_n[...,1])
    # print(nRMSEr)
    # print(np.mean(nRMSEr))
    # SSIMs = metricx.SSIM(maps[...,0],maps_n[...,0],data_range=1024)
    # SSIMr = metricx.SSIM(maps[...,1],maps_n[...,1],data_range=1024)
    # print(SSIMr)
    # print(np.mean(SSIMr))

    # plt.subplot(3,4,9)
    # plt.imshow(maps[i,:,:,0],cmap='jet',vmax=450,vmin=300),plt.colorbar(fraction=0.022)
    # plt.subplot(3,4,10)
    # plt.imshow(maps[i,:,:,1],cmap='jet',vmax=1100,vmin=0),plt.colorbar(fraction=0.022)
    # plt.subplot(3,4,11)
    # plt.imshow(maps_n[i,:,:,0],cmap='jet',vmax=450,vmin=300),plt.colorbar(fraction=0.022),plt.title('RE='+str(nRMSEs[i]),loc='left'),plt.title('SSIM='+str(SSIMs[i]),loc='right')
    # plt.subplot(3,4,12)
    # plt.imshow(maps_n[i,:,:,1],cmap='jet',vmax=1100,vmin=0),plt.colorbar(fraction=0.022),plt.title('RE='+str(nRMSEr[i]),loc='left'),plt.title('SSIM='+str(SSIMr[i]),loc='right')

    # plt.savefig(os.path.join('figure','tmp.png'))

    # nRMSEs=[]
    # nRMSEr=[]
    # SSIMs=[]
    # SSIMr=[]
    # for i in range(12):
    #     p = LogLinearN(imgs_n,tes,n=i)
    #     nRMSEs.append(metricx.nRMSE(maps[...,0],p[...,0],mean=True))
    #     nRMSEr.append(metricx.nRMSE(maps[...,1],p[...,1],mean=True))
    #     SSIMs.append(metricx.SSIM(maps[...,0],p[...,0],data_range=1024,mean=True))
    #     SSIMr.append(metricx.SSIM(maps[...,1],p[...,1],data_range=1024,mean=True))
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1),plt.title('NRMSE')
    # plt.plot(np.linspace(1,12,12),nRMSEs,'r',label='S0')
    # plt.plot(np.linspace(1,12,12),nRMSEr,'b',label='R2')
    # plt.legend()
    # plt.subplot(1,2,2),plt.title('SSIM')
    # plt.plot(np.linspace(1,12,12),SSIMs,'r',label='S0')
    # plt.plot(np.linspace(1,12,12),SSIMr,'b',label='R2')
    # plt.legend()
    # plt.savefig(os.path.join('figure','plot.png'))

    ##### DATA NOISE ANALYSIS #####
    # s = 17
    # wImgN = np.load(os.path.join('data','liver','Rician','wImgN'+str(s)+'.npy'))
    # sgm = SigmaG(wImgN,mean=True)
    # sgmM = SigmaG(wImgN,mean=False)
    # plt.figure(figsize=(3,20))
    # plt.imshow(sgmM),plt.colorbar(fraction=0.022)
    # plt.savefig(os.path.join('figure','sg.png'))
    # print(sgm)

    ##### MODEL LOSS ANALYSIS #####
    sigma = 17

    log_dir = os.path.join('model','CadamNet_sigma'+str(sigma))
    # log_dir = os.path.join('model','UNet_sigma'+str(sigma))
    # log_dir = os.path.join('model','CadamNet25_sigma'+str(sigma))
    # log_dir = os.path.join('model','CadamNet3_sigma'+str(sigma))
    # log_dir = os.path.join('model','DeepT2s_sigma'+str(sigma))

    # log_dir2 = os.path.join('model','UNetH_sigma'+str(sigma))
    # log_dir2 = os.path.join('model','CadamNet3_sigma'+str(sigma))
    # log_dir2 = os.path.join('model','UNet_sigma'+str(sigma))
    # log_dir2 = os.path.join('model','UNetH25_sigma'+str(sigma))
    # log_dir2 = os.path.join('model','Denoiser_sigma'+str(sigma))
    log_dir2 = os.path.join('model','MapNet_sigma'+str(sigma))
    plt.figure()
    plt.title(log_dir+' VS '+log_dir2)
    plot = 'validation'
    # plot = 'train'
    LogsPlot(log_dir=log_dir,dataSaved='all',plot=plot)
    # LogsPlot(log_dir=log_dir,dataSaved='only_train',plot=plot)
    LogsPlot(log_dir=log_dir2,dataSaved='all',plot=plot)
    # LogsPlot(log_dir=log_dir2,dataSaved='only_train',plot=plot)
    plt.savefig(os.path.join('figure','logs'))

    ##### STUDY R2 VALUES #####
    # m=1.0
    # pImg = np.load(os.path.join('data','liver','Rician','pImg_'+str(m)+'.npy'))
    # mask = np.load(os.path.join('data','liver','Rician','maskParenchyma.npy'))
    # mean,std = mean_std_roi(pImg[...,1],mask[...,1])
    # n=116
    # # mean=mean[100:]
    # plt.figure(figsize=(15,8))
    # # plt.bar(range(mean.shape[0]),height=mean)
    # plt.bar(range(mean.shape[0]),height=np.sort(mean))
    # # plt.imshow(pImg[n,...,1],cmap='jet',vmax=2000),plt.colorbar(fraction=0.022),plt.axis('off')
    # plt.savefig(os.path.join('figure','pImgR2.png'))
    # print(np.round(mean,decimals=1))
    # print(mean[n])

    ##### IN VIVO DATA PROCESSING #####
    # file_dir = os.path.join('data','liver','InVivo')
    # data_dir = os.path.join('data_test_clinical','mask')

    # HIC_level = ['normal','mild','moderate','severe']

    # masks_body  = []
    # masks_liver = []
    # masks_parenchyma = []

    # for i in range(4):
    #     mask_body = readRAW(os.path.join(data_dir,HIC_level[i]+'_body.raw'))
    #     mask_liver = readRAW(os.path.join(data_dir,HIC_level[i]+'_liver.raw')) 
    #     mask_parenchyma = readRAW(os.path.join(data_dir,HIC_level[i]+'_liver_parenchyma.raw'))

    #     masks_body.append(mask_body)
    #     masks_liver.append(mask_liver)
    #     masks_parenchyma.append(mask_parenchyma)

    # masks_body = np.array(masks_body)
    # masks_liver = np.array(masks_liver)
    # masks_parenchyma = np.array(masks_parenchyma)
    # masks_body=np.load(os.path.join(file_dir,'pImgPCANR4.npy'))[...,1]
    # masks_liver=np.load(os.path.join(file_dir,'pImgPCANR121.npy'))[...,1]
    # masks_liver=np.load(os.path.join(file_dir,'pImgM1NCM121.npy'))[...,1]
    # print(masks_body.shape,masks_liver.shape)

    # plt.figure(figsize=(30,20))
    # plt.subplot(4,4,1),plt.axis('off'),plt.title('Body',loc='left')
    # plt.imshow(masks_body[0]),plt.title('Normal')
    # plt.subplot(4,4,2),plt.axis('off')
    # plt.imshow(masks_body[1]),plt.title('Mild')
    # plt.subplot(4,4,3),plt.axis('off')
    # plt.imshow(masks_body[2]),plt.title('Moderate')
    # plt.subplot(4,4,4),plt.axis('off')
    # plt.imshow(masks_body[3]),plt.title('Severe')

    # plt.subplot(4,4,5),plt.axis('off'),plt.title('Liver',loc='left')
    # plt.imshow(masks_liver[0]),plt.title('Normal')
    # plt.subplot(4,4,6),plt.axis('off')
    # plt.imshow(masks_liver[1]),plt.title('Mild')
    # plt.subplot(4,4,7),plt.axis('off')
    # plt.imshow(masks_liver[2]),plt.title('Moderate')
    # plt.subplot(4,4,8),plt.axis('off')
    # plt.imshow(masks_liver[3]),plt.title('Severe')

    # plt.subplot(4,4,9),plt.axis('off'),plt.title('Parenchyma',loc='left')
    # plt.imshow(masks_parenchyma[0]),plt.title('Normal')
    # plt.subplot(4,4,10),plt.axis('off')
    # plt.imshow(masks_parenchyma[1]),plt.title('Mild')
    # plt.subplot(4,4,11),plt.axis('off')
    # plt.imshow(masks_parenchyma[2]),plt.title('Moderate')
    # plt.subplot(4,4,12),plt.axis('off')
    # plt.imshow(masks_parenchyma[3]),plt.title('Severe')
    # plt.savefig(os.path.join('figure','maskInVivo.png'))
