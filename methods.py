import numpy as np
import time
import tqdm
import os
from scipy.optimize import least_squares
from scipy.special import ive
from scipy import optimize,special
import matplotlib.pyplot as plt
import helper
import cv2
import metricx


def costfun_NCEXP(c,x,y,sigma=None):
    """
    Noise correction fitting cost function, first moment type for Rician noise.
    ### ARGUMENTS
    - c, estimated parameters.
    - x, echo time.
    - y, measured signal.
    """
    if sigma == None:
        s = c[0]*np.exp(-x/c[1])
        alpha = (0.5*s/c[2])**2
        tempw = (1+2*alpha)*ive(0,alpha)+2*alpha*ive(1,alpha)
        return y-np.sqrt(0.5*np.pi*c[2]*c[2])*tempw
    if sigma != None:
        # s = c[0]*np.exp(-x/c[1])
        s = c[0]*np.exp(-x*c[1]/1000.0)
        alpha = (0.5*s/sigma)**2
        tempw = (1+2*alpha)*ive(0,alpha)+2*alpha*ive(1,alpha)
        return y-np.sqrt(0.5*np.pi*sigma*sigma)*tempw

def costfun_M1NCM(c,x,y,sigma=None,NCoils=1):
    """
    Noise correction fitting cost function, first moment type for non-center chi square noise.
    ### ARGUMENTS
    - c, estimated parameters.
    - x, echo time.
    - y, measured signal.
    - sigma, background noise sigma.
    - NCoils, number of coils.
    """
    if sigma == None:
        a     = c[2]*np.sqrt(np.pi/2.0)
        frac  = special.factorial2(2*NCoils-1)/(np.power(2,NCoils-1)*special.factorial(NCoils-1))
        s     = c[0]*np.exp(-x/c[1])
        alpha = -0.5*(s/c[2])**2
        F     = special.hyp1f1(-0.5, NCoils,alpha)
        fun   = a*frac*F
        return y-fun
    if sigma != None:
        a     = sigma*np.sqrt(np.pi/2.0)
        frac  = special.factorial2(2*NCoils-1)/(np.power(2,NCoils-1)*special.factorial(NCoils-1))
        s     = c[0]*np.exp(-x/c[1])
        alpha = -0.5*(s/sigma)**2
        F     = special.hyp1f1(-0.5, NCoils,alpha)
        fun   = a*frac*F
        return y-fun

def costfun_Offset(c,x,y):
    """
    Offset model fitting function.
    """
    return y-(c[0]*np.exp(-x/c[1])+c[2])

def costfun_EXP(c,x,y):
    s = c[0]*np.exp(-x/c[1])
    # s = c[0]*np.exp(-x*c[1]/1000.0)
    return y-s

def costfun_M2NCM(c,x,y,sigma=None,NCoils=1):
    """
    Second moment noise correction model cost function.
    """
    if sigma == None: return y-(c[0]*c[0]*np.exp(-2*x/c[1])+2*NCoils*c[2]*c[2])
    # if sigma != None: return y-(c[0]*c[0]*np.exp(-2*x/c[1])+2*NCoils*sigma*sigma)
    if sigma != None: return y-np.sqrt(c[0]*c[0]*np.exp(-2*x*c[1]/1000.0)+2*NCoils*sigma*sigma)

def EXP(TEs,signal_measured,**kwargs):
    """
    Calculate T2 using first-moment noise correction model.
    ### ARGUMENT:
    - TEs (vector): TE of the T2s weighted images
    - signal_measured (vector): value of signal measured at one pixel
    ### RETURN:
    - S0 (float): fitted signal at TE = 0
    - T2 (float): fitted T2 value
    - sigma: estimated noise sigma.
    """  
    maxsignal = np.max([np.max(signal_measured),0.0])
    c0 = np.array([maxsignal, 0.5*np.max(TEs)]) 
    bounds = ([maxsignal, 0.1], [10.0*maxsignal+1.0, 3.0*max(TEs)])
    # c0  = np.array([maxsignal,1000.0/(0.5*np.max(TEs))])
    # bounds = ([maxsignal, 10.0], [10.0*maxsignal+1.0, 2000.0]); # @QiqiLu
    res = least_squares(costfun_EXP, c0, args=(TEs, signal_measured), bounds = bounds,method='trf')
    S0, R2 = res.x[0], 1000.0/res.x[1]
    # S0, R2 = res.x[0], res.x[1]
    return [S0, R2]

def NCEXP(TEs,signal_measured,sigma=None,**kwargs):
    """
    Calculate T2 using first-moment noise correction model.
    ### ARGUMENT:
    - TEs (vector): TE of the T2s weighted images
    - signal_measured (vector): value of signal measured at one pixel
    ### RETURN:
    - S0 (float): fitted signal at TE = 0
    - T2 (float): fitted T2 value
    - sigma: estimated noise sigma.
    """  
    maxsignal = np.max(signal_measured)
    if sigma == None:
        c0 = np.array([maxsignal, 0.5*np.max(TEs), 10.0]) 
        bounds = ([maxsignal, 0.1, 0.0], [10.0*maxsignal+1.0, 3.0*max(TEs), 100.0])
        res = least_squares(costfun_NCEXP, c0, args=(TEs, signal_measured), bounds = bounds,method='trf')
        S0, R2, sg = res.x[0], 1000.0/res.x[1], res.x[2]
        return [S0, R2, sg]
    if sigma != None:
        # c0 = np.array([maxsignal, 0.5*np.max(TEs)]) 
        c0  = np.array([maxsignal,1000.0/(0.5*np.max(TEs))])
        bounds = ([maxsignal, 10.0], [10.0*maxsignal+1.0, 1500.0]); # @QiqiLu

        # bounds = ([maxsignal, 0.1], [10.0*maxsignal+1.0, 3.0*max(TEs)])
        res = least_squares(costfun_NCEXP, c0, args=(TEs, signal_measured,sigma), bounds = bounds,method='trf')
        # S0, R2= res.x[0], 1000.0/res.x[1]
        S0, R2= res.x[0], res.x[1]
        return [S0, R2]

def M1NCM(TEs,signal_measured,sigma=None,NCoils=1,**kwargs):
    """
    Calculate T2 using first-moment noise correction model.
    ### ARGUMENT:
    - TEs (vector): TE of the T2s weighted images
    - signal_measured (vector): value of signal measured at one pixel
    ### RETURN:
    - S0 (float): fitted signal at TE = 0
    - T2 (float): fitted T2 value
    - sigma: estimated noise sigma.
    """  
    maxsignal = np.max(signal_measured)
    
    if sigma == None:
        c0 = np.array([maxsignal, 0.5*np.max(TEs), 10.0]) 
        bounds = ([maxsignal, 0.1, 0.0], [10.0*maxsignal+1.0, 3.0*max(TEs), 100.0])
        res = least_squares(costfun_M1NCM, c0, args=(TEs, signal_measured,sigma,NCoils), bounds = bounds,method='trf')
        S0, R2, sg = res.x[0], 1000.0/res.x[1], res.x[2]
        return [S0, R2, sg]
    if sigma != None:
        c0 = np.array([maxsignal, 0.5*np.max(TEs)]) 
        bounds = ([maxsignal, 0.1], [10.0*maxsignal+1.0, 3.0*max(TEs)])
        res = least_squares(costfun_M1NCM, c0, args=(TEs, signal_measured,sigma,NCoils), bounds = bounds,method='trf')
        S0, R2 = res.x[0], 1000.0/res.x[1]
        return [S0, R2]

def M2NCM(TEs,signal_measured,sigma=None,NCoils=1,**kwargs):
    """
    Second moment noise correction model.
    ### Parameters:
    - TEs (vector): TE of the T2s weighted images
    - signal_measured (vector): squared value of signal measured at on pixel
    ### Returns:
    - S0 (float): fitted signal at TE = 0
    - T2 (float): fitted T2 value
    - sigma (float): fitted sigma value
    """
    maxsignal = np.max(signal_measured)
    minsignal = np.min(signal_measured)
    if sigma==None: # estimated sigma
        c0 = np.array([maxsignal, 0.5*max(TEs), maxsignal/20.0])
        bounds = ([maxsignal, 0.1, min(0.5*minsignal, 0.0)], [10.0*maxsignal+1.0, 300.0, 2.0*maxsignal+1.0]); # @QiqiLu
        res = least_squares(costfun_M2NCM, c0, args=(TEs, signal_measured**2,sigma,NCoils), bounds = bounds, method='trf')
        S0, T2, sg = res.x[0], res.x[1], res.x[2]
        return [S0, 1000.0/T2, sg]
    if sigma!=None: # fixed sigma
        # c0  = np.array([maxsignal, 0.5*np.max(TEs)])
        c0  = np.array([maxsignal,1000.0/(0.5*np.max(TEs))])
        # bounds = ([maxsignal, 0.1], [10.0*maxsignal+1.0, 3.0*max(TEs)]); # @QiqiLu
        bounds = ([maxsignal, 10.0], [10.0*maxsignal+1.0, 1500.0]); # @QiqiLu
        # res = least_squares(costfun_M2NCM, c0, args=(TEs, signal_measured**2,sigma,NCoils), bounds = bounds, method='trf')
        res = least_squares(costfun_M2NCM, c0, args=(TEs, signal_measured,sigma,NCoils), bounds = bounds, method='trf')
        # S0, R2 = res.x[0], 1000.0/res.x[1]
        S0, R2 = res.x[0], res.x[1]
        return [S0, R2]

def Offset(TEs,signal_measured,**kwargs):
    """Calculate T2 using Offset model
    ### Arguments:
    - TEs (vector): TE of the T2s weighted images
    - signal_measured (vector): value of signal measured at one pixel

    ### Returns:
    - S0 (float): fitted signal at TE = 0
    - T2 (float): fitted T2 value
    - offset (float): fitted offset value in offset model
    """
    maxsignal = max(signal_measured)
    c0 = np.array([maxsignal, 0.5*max(TEs), 10.0])
    bounds = ([maxsignal, 0.1, 0.0], [max(20.0*maxsignal, 100.0), 300.0, 300.0]) # @YanqiuFeng
    # bounds = ([0, 0.1, 0.0], [np.inf, 3.0*max(TEs), 100]) # @ QiqiLu
    res = least_squares(costfun_Offset, c0, args=(TEs, signal_measured), bounds = bounds, method='trf',
                        ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    S0, R2, offset = res.x[0], 1000.0/res.x[1], res.x[2]
    return [S0, R2, offset]

def RsError(signal_measured,signal_fitted):
    """
    Rs Error.
    ### Argumens:
    - signal_measured (vector): value of signal measured at one pixel
    - signal_fitted (vector): value of signal fitted at one pixel
    ### Returns:
    - 1-sse /sst (float): Rs error
    """      
    ave_value = np.mean(signal_measured)
    
    temp = signal_measured - ave_value
    sst = sum(temp**2)    
    
    temp = signal_measured - signal_fitted
    sse = sum(temp**2)  
    
    return 1.0-sse/sst

def Truncation(TEs,signal_measured,**kwargs):
    """
    Automated truncation algorithm.
    ### Argument
    - TEs (vector): TE of the T2s weighted images
    - signal_measured (vector): value of signal measured at one pixel

    ### Return
    - S0 (float): fitted signal at TE = 0
    - T2 (float): fitted T2 value
    - num_trunc (int): fitted number of truncation 
    """  
    Rs_error,R2_last,num_trunc = 0.0,0.0,0
    TEs_all = TEs
    signal_measured_all = signal_measured
    error_limit = 0.990 # @YanqiuFeng
    
    while ((Rs_error < error_limit) & (TEs.shape[0] > 2)):
        para = EXP(TEs=TEs, signal_measured=signal_measured)
        signal_fitted = para[0]*np.exp(-TEs*para[1])
        Rs_error = RsError(signal_measured,signal_fitted)
        if ((Rs_error < error_limit) & (TEs.shape[0] > 2)):
            R2_last,S0 = para[1],para[0]
            signal_measured = np.delete(signal_measured,-1)
            TEs = np.delete(TEs,-1)
            num_trunc += 1
    R2,S0 = para[1],para[0]

    if ((R2_last > 0.0) & (TEs.shape[0] > 2)):
        while ((abs(R2-R2_last)/R2 > 0.025) & (TEs.shape[0]>2)):
            signal_measured = np.delete(signal_measured,-1)
            TEs = np.delete(TEs,-1)
            num_trunc += 1
            para = EXP(TEs=TEs,signal_measured=signal_measured)
            R2_last,R2,S0 = R2,para[1],para[0]

    if (R2 < 1000.0/20.0): # @TaigangHe2013
        Para = EXP(TEs=TEs_all,signal_measured=signal_measured_all)
        S0, R2, num_trunc = Para[0],Para[1],0
        
    return [S0, R2, num_trunc]

def PCANR(imgs,tes,sigma,f=5,m=0,beta=2.0,Ncoils=1,pbar_leave=True,model='NCEXP'):
    """
    Pixelwise curve-fitting with adaptive neighbor regularization `PCANR` methods for R2* parameter reconstruction.
    (Only for Rician noise)

    ### AUGMENTS
    - imgs: input measured images. [h, w, c]
    - tes: echo times.
    - sigma: background noise sigma.
    - f: neighbour size (2f+1)*(2f+1).
    - m: similarity patch size.
    - beta: regularization parameter.
    - Ncoils: (unused), when apply to multi coils, the noise correction model need to be modified.
    - pbar_leave: whether to leave pbar after ending.
    ### RETURN
    - maps: parameter maps. [batch, h, w, [S0 R2]]
    """
    models = ['NCEXP','SQEXP','M1NCM']
    assert model in models, 'Unsupported model.'
    assert len(imgs.shape) == 3, 'Unsupported size, [h,w,c] is needed.'
    # tes = tes/1000.0 # ms -> s
    h   = sigma*beta
    row,col,c = imgs.shape

    a     = sigma*np.sqrt(np.pi/2.0)
    frac  = special.factorial2(2*Ncoils-1)/(np.power(2,Ncoils-1)*special.factorial(Ncoils-1))

    imgs = np.pad(imgs,pad_width=((f+m,f+m),(f+m,f+m),(0,0)),mode='symmetric')
    S0   = np.zeros(shape=(row+f+m,col+f+m))
    R2   = np.zeros(shape=(row+f+m,col+f+m))
    one  = np.ones(shape=((2*f+1)**2,1))

    time.sleep(1.0)
    pbar = tqdm.tqdm(total=row*col,desc='PCANR',leave=pbar_leave)

    for i in range(f+m,f+m+row):
        for j in range(f+m,f+m+col):
            pbar.update(1)
            win = np.zeros((2*f+1,2*f+1)) # weight matrix
            for k in range(i-f,i+f+1):
                for l in range(j-f,j+f+1):
                    win[k-i+f,l-j+f] = (np.linalg.norm(imgs[i-m:i+m+1,j-m:j+m+1,:]-imgs[k-m:k+m+1,l-m:l+m+1,:]))**2 # distance calculation
            win  = np.exp(-win/(c*(2*m+1)**2)/(h**2)) # average on each pixel
            # wins = np.sort(win,axis=None)
            # win[f,f] = wins[-1] # Original code, maybe unrequired.
            if win[f,f]==0: win[f,f]=1
            win = win/np.sum(win) # normalization
            win = np.reshape(win,((2*f+1)**2,1))

            p = imgs[i-f:i+f+1,j-f:j+f+1,:] # data point in searching window
            p = np.reshape(p,((2*f+1)**2,c))

            S0_0 = np.max(imgs[i,j,:])+0.01
            R2_0 = 0.5*np.max(tes)/1000.0
            c0   = np.array([S0_0, R2_0]) 
            bounds = ((S0_0, 10.0*S0_0), (1e-5, 3.0))

            def costfun(c,x,y):
                if model == 'NCEXP':
                    #### first-moment noise correction mdeol #####
                    s     = c[0]*np.exp(-x*c[1])
                    alpha = (0.5*s/sigma)**2
                    tempw = (1+2*alpha)*ive(0,alpha)+2*alpha*ive(1,alpha)
                    fun   = np.sqrt(0.5*np.pi*sigma**2)*tempw
                    e     = np.sum((y-one*fun)**2,axis=1)[...,np.newaxis]
                    cost  = np.sum(win*e)

                if model == 'SQEXP':
                    #### second-moment noise correction model #####
                    fun   = np.sqrt(c[0]**2*np.exp(-2.0*x*c[1])+2.0*sigma**2)
                    e     = np.sum((y-one*fun)**2,axis=1)[...,np.newaxis]
                    cost  = np.sum(win*e)

                if model == 'M1NCM':
                    #### first-moment noncentral chi noise correction model #####
                    s     = c[0]*np.exp(-x*c[1])
                    alpha = -0.5*(s/sigma)**2
                    F     = special.hyp1f1(-0.5, Ncoils,alpha)
                    fun   = a*frac*F
                    e     = np.sum((y-one*fun)**2,axis=1)[...,np.newaxis]
                    cost  = np.sum(win*e)
                return cost
            res = optimize.minimize(costfun, c0,args=(tes,p), bounds=bounds)
            S0[i,j], R2[i,j] = res.x[0], res.x[1]*1000.0
    pbar.close()
    map = np.stack([S0,R2],axis=-1)
    return map[f+m:,f+m:,:]

def regularizationMatrix(imgs,i,j,sigma=7,f=5,m=0,beta=2.0,Ncoils=1):
    """
    Calculate regularization matrix in `PCANR` method.
    ### ARGUMENT
    - imgs: data.
    - i,j: data point position wanted.
    - sigma: background noise standard deviation.
    - f: neighbour size.
    - m: similarity window size.
    - beta: regularization parameter.
    - Ncoils: (unused) number of coils.

    ### RETURN
    - win: regularization matrix.
    """
    imgs = np.pad(imgs,pad_width=((f+m,f+m),(f+m,f+m),(0,0)),mode='symmetric')
    h   = sigma*beta
    row,col,c = imgs.shape
    win = np.zeros((2*f+1,2*f+1))
    for k in range(i-f,i+f+1):
        for l in range(j-f,j+f+1):
            win[k-i+f,l-j+f] = (np.linalg.norm(imgs[i-m:i+m+1,j-m:j+m+1,:]-imgs[k-m:k+m+1,l-m:l+m+1,:]))**2
    win  = np.exp(-win/(c*(2*m+1)**2)/(h**2)) # average on each pixel
    # wins = np.sort(win,axis=None)
    # win[f,f] = wins[-1]
    if win[f,f]==0: win[f,f]=1
    win  = win/np.sum(win) # normalization

    plt.figure()
    plt.subplot(1,2,1),plt.imshow(imgs[i-f:i+f+1,j-f:j+f+1,1],cmap='jet'),plt.colorbar(fraction=0.022)
    plt.subplot(1,2,2),plt.imshow(win[:,:],cmap='jet',vmax=np.max(win)),plt.colorbar(fraction=0.022)
    plt.savefig('figure/regularizationMatrix.png')

    return win

def invalid_model(x):
    raise Exception('Invalid Model.')
def invalid_filter(x):
    raise Exception('Invalid Filter.')

def NLM(imgs):
    """
    Nonlocal mean.
    """
    if len(imgs.shape)==2: imgs = imgs[...,np.newaxis]
    imgs = imgs.astype('uint16')
    for i in range(imgs.shape[2]):
        imgs[:,:,i]=cv2.fastNlMeansDenoising(src=imgs[:,:,i], dst=None, h=[5.0], templateWindowSize=3, searchWindowSize=11, normType = cv2.NORM_L1)
    imgs = imgs.astype('float32')
    return imgs

def PixelWiseMapping(imgs,tes,model='M1NCM',sigma=None,NCoils=1,pbar_leave=True,filtering=None):
    models = {
        'NCEXP':  NCEXP,
        'M1NCM':  M1NCM,
        'M2NCM':  M2NCM,
        'SQEXP':  M2NCM,
        'Offset': Offset,
        'EXP': EXP,
        'Truncation': Truncation,
    }

    filters = {
        'NLM':NLM,
    }

    if filtering != None:
        filter = filters.get(filtering,invalid_filter)
        imgs = filter(imgs)

    fun    = models.get(model,invalid_model)

    h,w,c = imgs.shape
    n     = h*w
    maps = []
    imgs_v= np.reshape(imgs,(-1,c))

    pbar = tqdm.tqdm(total=n,desc=model+': ',leave=pbar_leave)
    for i in range(n):
        pbar.update(1)
        para = fun(TEs=tes,signal_measured=imgs_v[i],sigma=sigma,NCoils=NCoils)
        maps.append(para)
    pbar.close()
    maps = np.reshape(maps,(h,w,-1))
    return maps

def AverageThenFitting(imgs,mask,tes,model='EXP',sigma=0,NCoils=1):
    models = {
        'M1NCM':  NCEXP,
        'M2NCM':  M2NCM,
        'Offset': Offset,
        'EXP': EXP,
        'Truncation': Truncation,
    }
    fun    = models.get(model,invalid_model)
    signal_ave = np.zeros(tes.shape[0])
    for i in range(tes.shape[0]): signal_ave[i] = np.sum(imgs[...,i]*mask)/np.sum(mask)
    para = fun(TEs=tes,signal_measured=signal_ave,sigma=sigma,NCoils=NCoils)
    return para

if __name__ == '__main__':
    ##### DATA INFORMATION #####
    TEs  = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # m = 2.5
    # m = 1.0
    # sigma = 9
    # sigma = 2.46
    # sigma = 6.96
    # sigma = 7.5
    # print(sigma)

    data_dir = os.path.join('data','liver','InVivo')
    # data_dir = os.path.join('data','liver','Rician')
    # data_dir = os.path.join('data','liver','ChiSquare')

    dataN = np.load(os.path.join(data_dir,'wImg121.npy'))
    # dataN = np.load(os.path.join(data_dir,'wImgDnM.npy'))
    # dataN = np.load(os.path.join(data_dir,'wImgDn'+str(sigma)+'.npy'))
    # dataN = np.load(os.path.join(data_dir,'wImgDnM'+str(sigma)+'.npy'))
    # dataN = np.load(os.path.join(data_dir,'wImgN'+str(sigma)+'_'+str(m)+'.npy'))

    # para  = np.load(os.path.join(data_dir,'pImg'+'_'+str(m)+'.npy'))

    sigma = helper.SigmaG(dataN,num_coil=1)

    # ##### SINGLE STUDY RECONSTRUCTION #####
    # n=100
    # dataN = dataN[n]
    # para  = para[n]
    # # dataN = dataN[20:40,40:60]
    # # para  = para[20:40,40:60]

    # # map = PCANR(dataN,tes=TEs,sigma=sigmas[n],beta=1.3,f=5,m=0,Ncoils=1,model='M1NCM')
    # map = PixelWiseMapping(dataN,TEs,model='M1NCM',sigma=None,NCoils=1)
    # print(map.shape)

    # np.save('figure/map.npy',map)
    # map = np.load('figure/map.npy')

    # maskLiver    = np.load(os.path.join(data_dir,'maskLiver.npy'))
    # nrmse = metricx.nRMSE(para[:,:,1][np.newaxis],map[:,:,1][np.newaxis],maskLiver[n,...,1][np.newaxis])
    # ssim  = metricx.SSIM(para[:,:,1][np.newaxis],map[:,:,1][np.newaxis],maskLiver[n,...,1][np.newaxis])
    # print(nrmse,ssim)
    # vmax = 1000*m
    # plt.figure(figsize=(20,10))
    # plt.subplot(2,2,1)
    # plt.imshow(para[:,:,1],vmax=vmax,vmin=0,cmap='jet'),plt.colorbar(fraction=0.022)
    # plt.subplot(2,2,2)
    # plt.imshow(map[:,:,1],vmax=vmax,vmin=0,cmap='jet'),plt.colorbar(fraction=0.022)
    # plt.subplot(2,2,3)
    # plt.imshow(np.abs(map[:,:,1]-para[:,:,1]),vmax=200,vmin=0,cmap='jet'),plt.colorbar(fraction=0.022)
    # plt.savefig('figure/aa.png')

    ##### MULTI STUDY RECONSTRUCTION #####
    mod = 'PCANR'
    print(mod)
    num = dataN.shape[0]
    maps = []
    bar = tqdm.tqdm(total=num,desc='ALL')
    for i in range(num):
        bar.update(1)
        map = PCANR(dataN[i],tes=TEs,sigma=sigma[i],beta=1.3,f=5,m=0,Ncoils=1,model='SQEXP',pbar_leave=False)
        # map = PixelWiseMapping(dataN[i],TEs,model=mod,sigma=sigma[i],NCoils=1,pbar_leave=False)
        maps.append(map)
    bar.close()
    maps=np.array(maps)
    # np.save(os.path.join(data_dir,'pImgM1NCMn'+str(sigma)+'_'+str(m)+'.npy'),maps)
    np.save(os.path.join(data_dir,'pImgPCANRsqexp.npy'),maps)

    # maskLiver = np.load(os.path.join(data_dir,'maskLiver.npy'))
    # nrmse     = metricx.nRMSE(para[...,1],maps[...,1],maskLiver[...,1])
    # ssim      = metricx.SSIM(para[...,1],maps[...,1],maskLiver[...,1])
    # print(np.mean(nrmse),np.mean(ssim))
    # ############
    




