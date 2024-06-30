# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:11:46 2020

@author: Recklexx
"""
import numpy as np
# import matplotlib.pyplot as plt
# import time
from scipy.optimize import least_squares
from scipy.special import ive 
# from tqdm import tqdm
# import os
# from data import create_ideal_and_simu_data


def calc_T2(TEs,signal_measured):
    """Calculate T2 using original model
    # Argumens:
        TEs (vector): TE of the T2s weighted images
        signal_measured (vector): value of signal measured at one pixel
    # Returns:
        S0 (float): fitted signal at TE = 0
        T2 (float): fitted T2 value
    """  
    maxsignal = max(signal_measured)
    c0 = np.array([maxsignal, 0.5*max(TEs)]) 
    # bounds = ([maxsignal, 0.1], [max(10.0*maxsignal, 200.0), 3.0*max(TEs)]) # @QiqiLu
    bounds = ([maxsignal, 0.1], [max(10.0*maxsignal, 100.0), 1000.0]) # @YanqiuFeng
    res = least_squares(fun_trunc, c0, args=(TEs, signal_measured), bounds = bounds, 
                        method='trf',ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    # res = least_squares(fun_trunc, c0, args=(TEs, signal_measured), 
    #                     method='lm',ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    S0, T2 = res.x[0], res.x[1]
    return S0, T2
    
def fun_trunc(c,x,y):
    """Auto truncation fitting function
    """
    return y-(c[0]*np.exp(-x/c[1]))

def Truncation(TEs,signal_measured):
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
    Rs_error = 0.0
    T2_last = 0.0
    num_trunc = 0
    
    TEs_all = TEs
    signal_measured_all = signal_measured
    
    # error_limit = 0.995 # @QiqiLu
    error_limit = 0.990 # @YanqiuFeng
    
    while ((Rs_error < error_limit) & (TEs.shape[0] > 2)):
        s0,t2 = calc_T2(TEs=TEs, signal_measured=signal_measured)
        signal_fitted = s0*np.exp(-TEs/t2)
        Rs_error = calc_Rs_error(signal_measured,signal_fitted)
        if ((Rs_error < error_limit) & (TEs.shape[0] > 2)):
            T2_last = t2
            S0 = s0
            signal_measured = np.delete(signal_measured,-1)
            TEs = np.delete(TEs,-1)
            num_trunc += 1
    
    T2 = t2
    S0 = s0
    
    if ((T2_last > 0.0) & (TEs.shape[0] > 2)):
        while ((abs(T2-T2_last)/T2 > 0.025) & (TEs.shape[0]>2)):
            signal_measured = np.delete(signal_measured,-1)
            TEs = np.delete(TEs,-1)
            num_trunc += 1
            s0,t2 = calc_T2(TEs=TEs,signal_measured=signal_measured)
            T2_last = T2
            T2 = t2
            S0 = s0
    
    # # @TaigangHe2013
    if (T2 > 20.0):
        S0,T2 = calc_T2(TEs=TEs_all,signal_measured=signal_measured_all)
        num_trunc = 0
    # # ==============
        
    return S0, T2, num_trunc




def fun_NCEXP(c,x,y):
    """Noise correction fitting function, first moment type
    """
    s = c[0]*np.exp(-x/c[1])
    alpha = (0.5*s/c[2])**2
    tempw = (1+2*alpha)*ive(0,alpha)+2*alpha*ive(1,alpha)
    return y-np.sqrt(0.5*np.pi*c[2]*c[2])*tempw

def fun_SQEXP(c,x,y):
    """Noise correction fitting function, second moment type
    """
    return y-(c[0]*c[0]*np.exp(-2*x/c[1])+2*c[2]*c[2])


def fun_mono_exp(S0,T2,TEs):
    """calculate the ideal value at each TE
    """
    return S0*np.exp(-TEs/T2)
    



def calc_T2_trunc_2sigma(TEs,signal_measured,sigma):
    num_trunc = 0;
    while ((signal_measured[-1]<3*sigma) & (TEs.shape[0] > 2)):
        signal_measured = np.delete(signal_measured,-1)
        TEs = np.delete(TEs,-1)
        num_trunc += 1
    s0,t2 = calc_T2(TEs=TEs,signal_measured=signal_measured)
    T2 = t2
    S0 = s0
    return S0, T2, num_trunc
        
        
def calc_Rs_error(signal_measured, signal_fitted):
    """Calculate Rs Error
    # Argumens:
        signal_measured (vector): value of signal measured at one pixel
        signal_fitted (vector): value of signal fitted at one pixel
    # Returns:
        1-sse /sst (float): Rs error
    """      
    ave_value = np.mean(signal_measured)
    
    temp = signal_measured - ave_value
    sst = sum(temp**2)    
    
    temp = signal_measured - signal_fitted
    sse = sum(temp**2)  
    
    return 1.0-sse/sst



def calc_T2_NCEXP(TEs,signal_measured):
    """Calculate T2 using noise correction model
    # Argumens:
        TEs (vector): TE of the T2s weighted images
        signal_measured (vector): value of signal measured at one pixel
    # Returns:
        S0 (float): fitted signal at TE = 0
        T2 (float): fitted T2 value
    """  
    maxsignal = max(signal_measured)
    minsignal = min(signal_measured)
    
    c0 = np.array([maxsignal, 0.5*max(TEs), 10]) 
    bounds = ([maxsignal, 0.1, min(0.5*minsignal, 0.0)], [max(10.0*maxsignal, 200.0), 3.0*max(TEs), 100.0])
    # bounds = ([0.0, 0.1, min(0.5*minsignal, 0)], [np.inf, 2.0*max(TEs), 100.0])
    res = least_squares(fun_NCEXP, c0, args=(TEs, signal_measured), bounds = bounds, 
                        method='trf',ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    # res = least_squares(fun_NCEXP, c0, args=(TEs, signal_measured), 
    #                     method='lm',ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    S0, T2, sigma = res.x[0], res.x[1], res.x[2]
    return S0, T2, sigma

def calc_T2_SQEXP(TEs,signal_measured_square):
    """Calculate T2 using noise correction model, second moment type
    # Parameters:
        TEs (vector): TE of the T2s weighted images
        signal_measured (vector): squared value of signal measured at on pixel
    # Returns:
        S0 (float): fitted signal at TE = 0
        T2 (float): fitted T2 value
        sigma (float): fitted sigma value
    """
    maxsignal = np.sqrt(max(signal_measured_square))
    minsignal = min(signal_measured_square)
    
    c0 = np.array([maxsignal, 0.5*max(TEs), maxsignal/20.0])
    # bounds = ([maxsignal, 0.1, min(0.5*minsignal, 0.0)], [10.0*maxsignal, 300.0, 2.0*minsignal]); # @YanqiuFeng
    bounds = ([maxsignal, 0.1, min(0.5*minsignal, 0.0)], [10.0*maxsignal+1.0, 300.0, 2.0*maxsignal+1.0]); # @QiqiLu
    res = least_squares(fun_SQEXP, c0, args=(TEs, signal_measured_square), bounds = bounds, 
                        method='trf',ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    S0, T2, sigma = res.x[0], res.x[1], res.x[2]
    return S0, T2, sigma
    




# =============================================================================
# if __name__ == '__main__':
#     
#     # TEs  = np.loadtxt(os.path.join("datatxt","roiMultiEchoTE.txt"))
#     
# # =============================================================================
# #     # Compare results with that from matlab
# #     # read txt data into array
# #     data = np.loadtxt("roiMultiEchoData.txt")
# #     y_true_trunc = np.loadtxt("roiT2AutoTrunc.txt")
# #     y_true_offset = np.loadtxt("roiT2Offset.txt")
# #     y_true_ncexp = np.loadtxt("roiNCEXPRes.txt")
# #     
# # 
# #     res_t2_offset = np.zeros(data.shape[0])
# #     res_t2_trunc = np.zeros(data.shape[0])
# #     res_t2_ncexp = np.zeros(data.shape[0])
# #     
# #     time_start = time.time()
# #     for index_pixel in range(0,data.shape[0]):
# #         S0_offset,T2_offset,offset = calc_T2_offset(TEs=TEs, signal_measured=data[index_pixel,:])
# #         S0_trunc,T2_trunc,num_trunc = calc_T2_trunc(TEs=TEs, signal_measured=data[index_pixel,:])
# #         S0_ncexp,T2_ncexp,sigma = calc_T2_NCEXP(TEs=TEs, signal_measured=data[index_pixel,:])
# #         res_t2_offset[index_pixel] = T2_offset
# #         res_t2_trunc[index_pixel] = T2_trunc
# #         res_t2_ncexp[index_pixel] = T2_ncexp
# #         
# #     time_end = time.time()
# #     print('time:',time_end-time_start)
# #     
# #     num_pixel_show = 150
# #     plt.figure()
# #     plt.plot(y_true_trunc[:num_pixel_show],'r',label='Trunc@Matlab')
# #     plt.plot(res_t2_trunc[:num_pixel_show],'b',label='Truncation')
# #     plt.legend(loc='upper right')
# #     plt.xlabel('Pixel')
# #     plt.ylabel('T2*')
# #     
# #     plt.figure()
# #     plt.plot(y_true_offset[:num_pixel_show],'r',label='Offset@Matlab')
# #     plt.plot(res_t2_offset[:num_pixel_show],'g',label='Offset')
# #     plt.legend(loc='upper right')
# #     plt.xlabel('Pixel')
# #     plt.ylabel('T2*')
# #     
# #     plt.figure()
# #     plt.plot(y_true_ncexp[:num_pixel_show],'r',label='NCEXP@Matlab')
# #     plt.plot(res_t2_ncexp[:num_pixel_show],'k',label='NCEXP')
# #     plt.legend(loc='upper right')
# #     plt.xlabel('Pixel')
# #     plt.ylabel('T2*')
# #     
# #     plt.figure()
# #     plt.plot(res_t2_trunc[:num_pixel_show],'r',label='Truncation')
# #     plt.plot(res_t2_offset[:num_pixel_show],'g',label='Offset')
# #     plt.plot(res_t2_ncexp[:num_pixel_show],'k',label='NCEXP')
# #     plt.legend(loc='upper right')
# #     plt.xlabel('Pixel')
# #     plt.ylabel('T2*')
# # =============================================================================
#     TEs = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.40, 11.80, 13.20, 14.60, 16.00])
#     
#     flag_recreate_simu_data = 0 
#     flag_refit_simu_data = 0
# 
#     # Simulation
#     print('Simulation start...')
#     
#     # simulation data parameters
#     noise_level = np.array([15.0,30.0,60.0]) # SNR
#     # noise_level = np.array([15.0])
#     S0 = np.array([200.0])
#     # R2 = np.array([30.0,500.0,1000.0])
#     R2 = np.linspace(20.0,1000.0,20) # @YanqiuFeng
#     T2 = 1000.0/R2
#     num_T2 = T2.shape[0]
#     num_repeat = 1000
#     num_pixel = 100 # num of pixel in the ROI
#     num_channel = 1
#     
#     if flag_recreate_simu_data:
#         # create ideal and simulated noise data
#         data_ideal,data_noise,_ = create_ideal_and_simu_data(noise_level=noise_level,
#                                                            S0=S0,
#                                                            T2=T2,
#                                                            num_repeat=num_repeat,
#                                                            num_pixel=num_pixel,
#                                                            TEs=TEs,
#                                                            num_channel=num_channel)
#         # filename_simu_data_ideal = os.path.join('datasimu', 'simuDataIdeal')
#         # filename_simu_data_noise = os.path.join('datasimu', 'simuDataNoise')
#         # filename_simu_data_t2 = os.path.join('datasimu', 'simuDataT2')
#         np.save(os.path.join('datasimu', 'simuDataIdeal'),data_ideal)
#         np.save(os.path.join('datasimu', 'simuDataNoise'),data_noise)
#         np.save(os.path.join('datasimu', 'simuDataT2'),   T2)
#     else:
#         data_ideal = np.load(os.path.join('datasimu', 'simuDataIdeal.npy'))
#         data_noise = np.load(os.path.join('datasimu', 'simuDataNoise.npy'))
#         data_t2    = np.load(os.path.join('datasimu', 'simuDataT2.npy'))
#     
# # =============================================================================
# #     # Example of single pixel
# #     plt.figure()
# #     plt.plot(np.insert(TEs,0,0),np.insert(data_ideal[0,0,10,0,0,:],0,200.0),label='Identity')
# #     plt.plot(np.insert(TEs,0,0),np.insert(data_noise[0,0,10,0,0,:],0,200.0),label='Simulation')
# #     plt.legend()
# #     plt.xlabel('TE')
# #     plt.ylabel('Signal')
# # =============================================================================
#     
#     # Example of a noise-free exponential decay (solid line) with a four-channel 
#     # array coil (R2* =500 s-1, S0 = 200) and the expectation of the observed 
#     # exponential decay curves in the presence of the noncentral chi noise with 
#     # SNRs of 15, 30, and 60.
#    
#     # calculate the means of the signal of pixels over ROI
#     data_mean_pixel = np.mean(data_noise,axis=4)
#     plt.figure() # insert start point (0,200)
#     plt.plot(np.insert(TEs,0,0),np.insert(np.mean(data_ideal,axis=4)[0,0,10,0,:],0,200.0),label='Noise-free')
#     plt.plot(np.insert(TEs,0,0),np.insert(np.mean(data_mean_pixel,axis=3)[0,0,10,:],0,200.0),'--',marker='o',label='SNR=15')
#     plt.plot(np.insert(TEs,0,0),np.insert(np.mean(data_mean_pixel,axis=3)[1,0,10,:],0,200.0),'--',marker='s',label='SNR=30')
#     plt.plot(np.insert(TEs,0,0),np.insert(np.mean(data_mean_pixel,axis=3)[2,0,10,:],0,200.0),'--',marker='*',label='SNR=60')
#     plt.legend()
#     plt.xlabel('TE')
#     plt.ylabel('Signal')
#     
#     index_noise_level = 0
#     index_s0 = 0
#     
#     sigma = S0[index_s0]/noise_level[index_noise_level]
#     
#     data = data_noise[index_noise_level,index_s0,:,:,:,:]
#     
#     # calculate the means of the signal of pixels over ROI
#     data = np.mean(data,axis=2)
#     
#     # result containers
#     res_r2_offset = np.zeros([data.shape[0],data.shape[1]])
#     res_r2_trunc = np.zeros([data.shape[0],data.shape[1]])
#     res_r2_ncexp = np.zeros([data.shape[0],data.shape[1]])
#     res_r2_sqexp = np.zeros([data.shape[0],data.shape[1]])
#     
#     # res_num_trunc = np.zeros([data.shape[0],data.shape[1]])
# 
#     if flag_refit_simu_data:
#         # Fitting every repeat
#         time.sleep(1)
#         pbar = tqdm(total=num_T2*num_repeat,desc='Fitting:')
#         
#         print('Fitting...')
#         for index_t2 in range(0,data.shape[0]):
#             for index_repeat in range(0,data.shape[1]):
#                 pbar.update(1)
#                 S0_offset,T2_offset,offset = calc_T2_offset(TEs=TEs, signal_measured=data[index_t2,index_repeat,:])
#                 # S0_trunc,T2_trunc,num_trunc = calc_T2_trunc(TEs=TEs, signal_measured=data[index_t2,index_repeat,:])
#                 S0_trunc,T2_trunc,num_trunc = calc_T2_trunc_2sigma(TEs=TEs, signal_measured=data[index_t2,index_repeat,:],sigma=sigma)
#                 S0_ncexp,T2_ncexp,sigma = calc_T2_NCEXP(TEs=TEs, signal_measured=data[index_t2,index_repeat,:])
#                 S0_sqexp, T2_sqexp,_ = calc_T2_SQEXP(TEs=TEs, signal_measured_square=data[index_t2,index_repeat,:]**2)
#                 
#                 res_r2_offset[index_t2,index_repeat] = 1000.0/T2_offset
#                 res_r2_trunc[index_t2,index_repeat]  = 1000.0/T2_trunc
#                 res_r2_ncexp[index_t2,index_repeat]  = 1000.0/T2_ncexp
#                 res_r2_sqexp[index_t2,index_repeat]  = 1000.0/T2_sqexp
#                 # res_num_trunc[index_t2,index_repeat] = num_trunc
#                 
#         pbar.close()   
#         print('Fitting done.')
#         np.save(os.path.join('datasimu', 'resR2Offset'),res_r2_offset)
#         np.save(os.path.join('datasimu', 'resR2Trunc'),res_r2_trunc)
#         np.save(os.path.join('datasimu', 'resR2NCEXP'),res_r2_ncexp)
#         np.save(os.path.join('datasimu', 'resR2SQEXP'),res_r2_sqexp)
#     else:
#         res_r2_offset = np.load(os.path.join('datasimu','resR2Offset.npy'))
#         res_r2_trunc = np.load(os.path.join('datasimu','resR2Trunc.npy'))
#         res_r2_ncexp = np.load(os.path.join('datasimu','resR2NCEXP.npy'))
#         res_r2_sqexp = np.load(os.path.join('datasimu','resR2SQEXP.npy'))
#         
#     
#     # Calculate the mean (accuracy) of R2* quantification using different model    
#     mean_r2_offset = np.mean(res_r2_offset,axis=1)
#     mean_r2_trunc = np.mean(res_r2_trunc,axis=1)
#     mean_r2_ncexp = np.mean(res_r2_ncexp,axis=1)
#     mean_r2_sqexp = np.mean(res_r2_sqexp,axis=1)
#     
#     # Calculate the std (precision) of R2* quantification using different model    
#     std_r2_offset = np.std(res_r2_offset,axis=1)
#     std_r2_trunc = np.std(res_r2_trunc,axis=1)
#     std_r2_ncexp = np.std(res_r2_ncexp,axis=1)
#     std_r2_sqexp = np.std(res_r2_sqexp,axis=1)
#     
#     # plot the mean of the repeat result
#     plt.figure()
#     plt.plot(R2,R2,'-r',marker='o',label='Identity')
#     plt.plot(R2,mean_r2_offset,'-b',marker='o',label='Offset')
#     plt.plot(R2,mean_r2_trunc,'-g',marker='o',label='Truncation')
#     plt.plot(R2,mean_r2_ncexp,'-m',marker='o',label='NCEXP')
#     plt.plot(R2,mean_r2_sqexp,'-k',marker='o',label='SQEXP')
#     plt.plot([0,1000],[1000,1000],'--k')
#     plt.legend()
#     plt.title("%s%d"%('SNR=',noise_level[index_noise_level]))
#     plt.xlabel('True R2* (s-1)')
#     plt.ylabel('Mean of Estimated R2* (s-1)')
#     
#     # plot the std of the repeat result 
#     plt.figure()
#     plt.plot(R2,std_r2_offset,'-b',marker='o',label='Offset')
#     plt.plot(R2,std_r2_trunc,'-g',marker='o',label='Truncation')
#     plt.plot(R2,std_r2_ncexp,'-m',marker='o',label='NCEXP')
#     plt.plot(R2,std_r2_sqexp,'-k',marker='o',label='SQEXP')
#     plt.plot([0,1000],[100,100],'--k')
#     plt.legend()
#     plt.title("%s%d"%('SNR=',noise_level[index_noise_level]))
#     plt.xlabel('True R2* (s-1)')
#     plt.ylabel('SD of Estimated R2* (s-1)')
#      
#     # plt.figure()
#     # plt.plot(R2,np.mean(res_num_trunc,axis=1))
#     
# =============================================================================
