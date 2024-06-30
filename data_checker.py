#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:12:57 2020

@author: luqiqi
"""
def check_info(study_path):
    # check clinical data infomation
    import glob
    import os
    import numpy as np
    import pydicom
    import re
    
    # get study folder names
    study_path = os.path.join(study_path,'Study2*')
    study_names = sorted(glob.glob(study_path),key=os.path.getmtime,reverse=True)
    num_study = np.array(study_names).shape[0]
    TE,Sex,Name,Age,AcquisitionDate,SequenceName,B0,Manu,Institution,ID=[],[],[],[],[],[],[],[],[],[]
    num_female,num_male = 0,0

    # for every study
    print('Study name | B0 | Repetition time | Flip angle | TEs | Slice thickness | Resolution | Matrix | Ave | FatSat | Manu | Sequence')
    for study_name,id_study in zip(study_names, range(0,num_study)):
        image_names = glob.glob(os.path.join(study_name, '*.dcm'))
        image_names = sorted(image_names,key=os.path.getmtime,reverse=True)
        num_files = np.array(image_names).shape[0]
        # get TE
        tes = []
        for image_name,i in zip(image_names, range(0,num_files)):
            x = pydicom.dcmread(image_name)
            tes.append(x.EchoTime)
        TE.append(tes)
        # get other information
        X   = pydicom.dcmread(image_names[0])
        age = re.findall('0(\d+)Y',X.PatientAge)
        age = list(map(int,age))[0]
        if X.PatientSex=='F':
            sex=2
            num_female = num_female+1
        else:
            sex=1
            num_male = num_male+1
        Sex.append(sex)
        Name.append(X.PatientName),Age.append(age)
        AcquisitionDate.append(X.AcquisitionDate),SequenceName.append(X.SequenceName)
        Manu.append(X.Manufacturer),B0.append(X.MagneticFieldStrength)
        try:
            Institution.append(X.InstitutionName)
        except:
            Institution.append('None') # may some study have not saved institution name
        ID.append(id_study)

        # print each study information
        print(str(id_study),'|',X.MagneticFieldStrength,'|',X.RepetitionTime,'|',X.FlipAngle,'|',tes,'|',X.SliceThickness,'|',
        X.PixelSpacing,'|',X.AcquisitionMatrix,'|',X.NumberOfAverages,'|',X.ScanOptions,'|',X.Manufacturer,'|',X.SequenceName)

    # Age distribution
    age_mean,age_std,age_max,age_min = np.mean(Age),np.std(Age),np.max(Age),np.min(Age)
    print('Age distribution: '+str(int(age_mean))+'+-'+str(int(age_std))+' ('+str(age_min)+','+str(age_max)+')')
    # Sex distribution
    print('Sex distribution: '+str(num_female)+'(female)| '+str(num_male)+'(male)')    
     
    info = {'study_names':study_names,'TE':TE,'Sex':Sex,'Name':Name,'Age':Age,'AcquisitionData':AcquisitionDate,
            'SequenceName':SequenceName,'B0':B0,'Manu':Manu,'Institution':Institution,'ID':ID}
    return info

def check_noise(data_study, data_s0=None, num_coil=1):
    import os
    # get the background nosie in each weighted image of each study
    print('>> Check noise level in each study data (TE0) ...')
    import matplotlib.pyplot as plt
    import numpy as np
    
    # show the background position set
    i,h=0,5 # show the first image, the size of the background roi is 5*5 
    img = plt.figure()
    ax_img = img.add_subplot(1,1,1)
    rect1 = plt.Rectangle((122,1), h, h, fill=False,edgecolor='red',linewidth=1)
    rect2 = plt.Rectangle((1,1), h, h, fill=False,edgecolor='red',linewidth=1)
    ax_img.add_patch(rect1)
    ax_img.add_patch(rect2)
    plt.imshow(data_study[i,:,:,0]),plt.title('T2*w image Study['+str(i)+'] TE[0]'),plt.show()
    plt.savefig(os.path.join('figure','background_roi_set.png'))
    
    # calculate the mean of the bkg in [each study, each TE]
    region_bkg_mean1=np.zeros((data_study.shape[0],data_study.shape[-1]))
    region_bkg_mean2=np.zeros((data_study.shape[0],data_study.shape[-1]))
    for i in range(0,data_study.shape[0]): # each study
        for j in range(0,data_study.shape[-1]): # each TE time
            region_bkg_mean1[i,j] = np.mean(data_study[i,1:1+h,1:1+h,j]**2)
            region_bkg_mean2[i,j] = np.mean(data_study[i,1:1+h,122:122+h,j]**2)
    

    mean_bkg1 = np.mean(region_bkg_mean1,axis=-1)
    mean_bkg2 = np.mean(region_bkg_mean2,axis=-1)

    # calculate mean bkg signal in each study
    mean_bkg  = np.mean([mean_bkg1,mean_bkg2],axis=0)
    sigma_g   = np.sqrt(mean_bkg/(2*num_coil))

    # calculate the max pixel value
    s_max = np.zeros(data_study.shape[0])
    for i in range(0,data_study.shape[0]): # each study
        s_max[i] = np.max(data_study[i])


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
    mean_roi    = []
    for i in range(data_study.shape[0]):
        mean = np.mean(data_study[i,y[i]:y[i]+h,x[i]:x[i]+w,0])
        mean_roi.append(mean)
    
    SNR = mean_roi/sigma_g

    if data_s0 is not None:
        mean_roi_s0 = []
        for i in range(data_study.shape[0]):
            mean_s0 = np.mean(data_s0[i,y[i]:y[i]+h,x[i]:x[i]+w])
            mean_roi_s0.append(mean_s0)
        SNR_s0 = mean_roi_s0/sigma_g

    SNR_max = s_max/sigma_g

    print('>> All sets: sigma: '+ str(round(np.mean(sigma_g),2))+'('+str(round(np.std(sigma_g),2))+')'+
    ' SNR: '+ str(round(np.mean(SNR),2))+'('+str(round(np.std(SNR),2))+')'+
    ' SNR_s0: '+ str(round(np.mean(SNR_s0),2))+'('+str(round(np.std(SNR_s0),2))+')'+
    ' SNR_max: '+ str(round(np.mean(SNR_max),2))+'('+str(round(np.std(SNR_max),2))+')')

    print('>> Train sets: sigma: '+ str(round(np.mean(sigma_g[0:100]),2))+'('+str(round(np.std(sigma_g[0:100]),2))+')'+
    ' SNR: '+ str(round(np.mean(SNR[0:100]),2))+'('+str(round(np.std(SNR[0:100]),2))+')'+
    ' SNR_s0: '+ str(round(np.mean(SNR_s0[0:100]),2))+'('+str(round(np.std(SNR_s0[0:100]),2))+')'+
    ' SNR_max: '+ str(round(np.mean(SNR_max[0:100]),2))+'('+str(round(np.std(SNR_max[0:100]),2))+')')

    print('>> Test sets: sigma: '+ str(round(np.mean(sigma_g[100:]),2))+'('+str(round(np.std(sigma_g[100:]),2))+')'+
    ' SNR: '+ str(round(np.mean(SNR[100:]),2))+'('+str(round(np.std(SNR[100:]),2))+')'+
    ' SNR_s0: '+ str(round(np.mean(SNR_s0[100:]),2))+'('+str(round(np.std(SNR_s0[100:]),2))+')'+
    ' SNR_max: '+ str(round(np.mean(SNR_max[100:]),2))+'('+str(round(np.std(SNR_max[100:]),2))+')')

     
    plt.figure(figsize=(10,10)) 
    plt.subplot(221)
    plt.plot(np.sqrt(mean_bkg1),'b*',label='Mean (Region 1)')
    plt.plot(np.sqrt(mean_bkg2),'g*',label='Mean (Region 2)')
    plt.plot(np.sqrt(mean_bkg), 'k*',label='Mean')
    m = np.mean(np.sqrt(mean_bkg))
    plt.axhline(m,color='gray', linestyle='--'),plt.text(x=0,y=m,s=str(round(m,2)),color='r')
    plt.title('Mean of Background Signal'),plt.xlabel('Study')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(mean_roi,'k*',label='Mean (ROI)')
    # plt.plot(s_max,'k*',label='Mean (ROI)')
    plt.title('Mean of ROI Signal'),plt.xlabel('Study')
    plt.legend()
   
    plt.subplot(223)
    plt.plot(sigma_g,'k*')
    ms = np.mean(sigma_g)
    plt.axhline(ms,color='gray', linestyle='--'),plt.text(x=0,y=ms,s=str(round(ms,2)),color='r')
    plt.title('$\sigma_g$ ($N_{Coils}$='+str(num_coil)+')'),plt.xlabel('Study')

    plt.subplot(224)
    plt.plot(SNR,'*k',label='$S_{TE_0}$/$\sigma$')
    msnr = np.mean(SNR)
    plt.axhline(msnr,color='k', linestyle='--'),plt.text(x=0,y=msnr,s=str(round(msnr,2)),color='r')
    if data_s0 is not None:
        plt.plot(SNR_s0,'*g',label='$S_0$/$\sigma$')
        plt.axhline(np.mean(SNR_s0),color='g', linestyle='--')
        plt.text(x=0,y=np.mean(SNR_s0),s=str(round(np.mean(SNR_s0),2)),color='r')
    plt.title('SNR'),plt.xlabel('Study'),plt.legend()

    plt.savefig(os.path.join('figure','SNR.png'))
    

def check_map(maps):
    import numpy as np
    num_study = maps.shape[0]
    R2s_mean = []
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
    
    # import matplotlib.pyplot as plt
    for i in range(0,num_study):
        # # show ROI position
        # img = plt.figure()
        # ax_img = img.add_subplot(1,1,1)
        # rect = plt.Rectangle((x[i],y[i]), w, h, fill=False,edgecolor='red',linewidth=1.0)
        # ax_img.add_patch(rect)
        # plt.imshow(maps[i,:,:],vmax=1000)
        # plt.title(i),plt.show()
        
        mean = np.mean(maps[i,y[i]:y[i]+h,x[i]:x[i]+w])
        R2s_mean.append(mean)
        level.append(check_level(mean))

    return R2s_mean, level

def check_level(R2s):
    # check sereum ferritin level
    T2s = 1000.0/R2s
    if T2s>6.3:
        return 0
    elif T2s>2.8:
        return 1
    elif T2s>1.4:
        return 2
    else:
        return 3
        
        
if __name__ == '__main__':
    # study_path = 'data_liver_mr_same_te'
    # info = check_info(study_path)
    
    # from data_generator import get_clinical_data
    # data_noise = get_clinical_data().astype('float32')
    # num_coil = 8
    # check_noise(data_study=data_noise,num_coil=num_coil)

    from data_generator import get_data_train
    data_dicom = get_data_train(sigma_g=17)
    data_noise = data_dicom['noise data']
    data_s0    = data_dicom['s0 map']
    num_coil = 1
    check_noise(data_study=data_noise,data_s0=data_s0,num_coil=num_coil)
    
    # from data_generator import get_map
    # _,map_r2s = get_map(True)
    # R2s_mean,level = check_map(map_r2s)
    
    # from data_generator import get_data_train_patches
    # data = get_data_train_patches(sigma_g=13)
    # data_noise = data['noise free data']
    # print(data_noise.shape)
    
