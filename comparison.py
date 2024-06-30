# functions to compare results
from numpy.core.fromnumeric import mean


def bland_altman_plot(data_true,data_pred,masks):
    import matplotlib.pyplot as plt 
    # import os
    import numpy as np
    mask_np = (masks+1)%2

    data_true_masked = np.ma.array(data_true,mask=mask_np)
    data_pred_masked = np.ma.array(data_pred,mask=mask_np)

    data_true_mean = data_true_masked.mean(axis=-1).mean(-1)
    data_pred_mean = data_pred_masked.mean(axis=-1).mean(-1)+1

    mean      = np.mean([data_true_mean, data_pred_mean], axis=0)
    diff      = data_pred_mean - data_true_mean # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    plt.plot(mean, diff,'ob')
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='black', linestyle='--')
    plt.axhline(md - 1.96*sd, color='black', linestyle='--')
    offset = 1001
    plt.annotate(str(np.round(md,1)),(offset,md+1),color='red')
    plt.annotate(str(np.round(md + 1.96*sd,1)),(offset,md + 1.96*sd+1),color='red')
    plt.annotate(str(np.round(md - 1.96*sd,1)),(offset,md - 1.96*sd+1),color='red')

    plt.ylim([-200,200]),plt.ylabel('$R_2^*$ Difference ($ms^{-1}$)')
    plt.xlim([0,1000]),plt.xlabel('Mean $R_2^*$ ($ms^{-1}$)')

    # from metrics import get_p_value
    # if md>0:
    #     p = get_p_value(data_pred_mean,data_true_mean,alt='greater')
    # else:
    #     p = get_p_value(data_pred_mean,data_true_mean,alt='less')

    # plt.annotate(str(np.round(p,4)),(1000,md+50),color='red')


def roi_analysis(data_true,data_pred,masks):
    # import matplotlib.pyplot as plt
    import numpy as np
    mask_np = (masks+1)%2 # region of maskout

    data_true_masked = np.ma.array(data_true,mask=mask_np)
    data_pred_masked = np.ma.array(data_pred,mask=mask_np)

    data_true_mean = data_true_masked.mean(axis=-1).mean(-1)
    data_pred_mean = data_pred_masked.mean(axis=-1).mean(-1)

    return data_true_mean,data_pred_mean

def mean_std_roi(data,masks):
    import numpy as np
    mask_np = (masks+1)%2 # region of maskout
    data_masked = np.ma.array(data,mask=mask_np)
    mean = data_masked.mean(axis=-1).mean(-1)
    std = data_masked.std(-1).std(-1)
    return mean, std



def regression_plot(result_comp,result_prop,test_noise_level):
    # load test data
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    data_test = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'.npy'),allow_pickle=True).item()

    map_r2s_test    = data_test['r2s map']
    map_r2s_comp  = result_comp['mapp'][:,:,:,1]
    map_r2s_prop  = result_prop['mapp'][:,:,:,1]
    map_r2s_pcanr = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'_map_r2s_pcanr.npy')) 
    map_r2s_m1ncm = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'_map_r2s_m1ncm.npy')) 

    # load test data masks
    from data_generator import get_mask_test, get_mask_liver
    # masks_body_test = get_mask_test() # body mask
    mask_liver_whole,mask_liver_parenchyma = get_mask_liver() # whole liver and parenchyma mask
    # mask_liver_whole_test = mask_liver_whole[-21:]
    mask_liver_parenchyma_test = mask_liver_parenchyma[-21:]
    masks = mask_liver_parenchyma_test

    from comparison import roi_analysis
    # mean_gt, mean_m1ncm = roi_analysis(map_r2s_test,map_r2s_m1ncm,masks)
    # _, mean_pcanr       = roi_analysis(map_r2s_test,map_r2s_pcanr,masks)
    # _, mean_pred_comp   = roi_analysis(map_r2s_test,map_r2s_comp, masks)
    # _, mean_pred_prop   = roi_analysis(map_r2s_test,map_r2s_prop, masks)
    mean_gt, std_gt = mean_std_roi(map_r2s_test,masks)
    mean_m1ncm, std_m1ncm = mean_std_roi(map_r2s_m1ncm,masks)
    mean_pcanr, std_pcanr = mean_std_roi(map_r2s_pcanr,masks)
    mean_pred_comp, std_pred_comp = mean_std_roi(map_r2s_comp,masks)
    mean_pred_prop, std_pred_prop = mean_std_roi(map_r2s_prop,masks)
    print(mean_m1ncm)
    
    # print(mean_pred_comp)
    # print(mean_pred_prop)
    plt.figure(figsize=(7.5,7.5))
    plt.scatter(mean_gt, mean_m1ncm,color='g',marker='o',label='M$^1$NCM')
    fn_m1ncm = np.poly1d(np.polyfit(mean_gt,mean_m1ncm,1))
    plt.plot([20,980],fn_m1ncm([20,980]),'--g')

    plt.scatter(mean_gt, mean_pcanr,color='b',marker='o',label='PCANR')
    fn_pcanr = np.poly1d(np.polyfit(mean_gt,mean_pcanr,1))
    plt.plot([20,980],fn_pcanr([20,980]),'--b')

    plt.scatter(mean_gt, mean_pred_comp,color='orange',marker='o',label='MappingNet')
    fn_pred_comp = np.poly1d(np.polyfit(mean_gt,mean_pred_comp,1))
    plt.plot([20,980],fn_pred_comp([20,980]),'--',color='orange')

    plt.scatter(mean_gt, mean_pred_prop,color='r',marker='o',label='CadamNet')
    fn_pred_prop = np.poly1d(np.polyfit(mean_gt,mean_pred_prop,1))
    plt.plot([20,980],fn_pred_prop([20,980]),'--r')

    plt.plot([0,1000],[0,1000],'-k')
    # plt.plot([0,1000],[1000,1000],'-k')
    plt.ylim([0,1000]),plt.ylabel('Estimated $R_2^*$ ($ms^{-1}$)')
    plt.xlim([0,1000]),plt.xlabel('Ground Truth $R_2^*$ ($ms^{-1}$)')
    plt.legend(loc='upper left')
    plt.title('ROI-Analysis'+' ('+str(test_noise_level)+')')
    plt.savefig(os.path.join('figure','roi_analysis_mean.png'))

    plt.figure(figsize=(7.5,7.5))
    ind = np.argsort(mean_gt)
    plt.plot(mean_gt[ind], std_gt[ind],'-k',marker='o',label='GT')
    plt.plot(mean_gt[ind], std_m1ncm[ind],'-g',marker='o',label='M$_1$NCM')
    plt.plot(mean_gt[ind], std_pcanr[ind],'-b',marker='o',label='PCANR')
    plt.plot(mean_gt[ind], std_pred_comp[ind],'-',color='orange',marker='o',label='MappingNet')
    plt.plot(mean_gt[ind], std_pred_prop[ind],'-r',marker='o',label='CadamNet')

    plt.ylim([0,300]),plt.ylabel('Estimated $R_2^*$ ($ms^{-1}$)')
    plt.xlim([0,1000]),plt.xlabel('Ground Truth $R_2^*$ ($ms^{-1}$)')
    plt.legend(loc='upper left')
    plt.title('ROI-Analysis'+' ('+str(test_noise_level)+')')
    plt.savefig(os.path.join('figure','roi_analysis_std.png'))


        # comparison with the real value
    from comparison import bland_altman_plot
    # compare predicted map with true map
    plt.figure(figsize=(12.5,10))

    plt.subplot(221)
    bland_altman_plot(map_r2s_test,map_r2s_m1ncm,mask_liver_parenchyma_test),plt.title('M1NCM vs Ground Truth')
    plt.subplot(222)
    bland_altman_plot(map_r2s_test,map_r2s_pcanr,mask_liver_parenchyma_test),plt.title('PCANR vs Ground Truth')
    plt.subplot(223)
    bland_altman_plot(map_r2s_test,map_r2s_comp,mask_liver_parenchyma_test),plt.title('MappingNet vs Ground Truth')
    plt.subplot(224)
    bland_altman_plot(map_r2s_test,map_r2s_prop, mask_liver_parenchyma_test), plt.title('CadamNet vs Ground Truth')
    # plt.tight_layout()
    plt.savefig(os.path.join('figure','baplot_r2comparison.png'))

    plt.figure(figsize=(17,10.5))
    plt.subplot(2,3,1)
    bland_altman_plot(map_r2s_pcanr,map_r2s_m1ncm,mask_liver_parenchyma_test),plt.title('M1NCM vs PCANR')
    plt.subplot(2,3,2)
    bland_altman_plot(map_r2s_comp,map_r2s_pcanr,mask_liver_parenchyma_test),plt.title('PCANR vs MappingNet')
    plt.subplot(2,3,3)
    bland_altman_plot(map_r2s_comp,map_r2s_m1ncm,mask_liver_parenchyma_test),plt.title('M1NCM vs MappingNet')
    plt.subplot(2,3,4)
    bland_altman_plot(map_r2s_prop,map_r2s_m1ncm, mask_liver_parenchyma_test), plt.title('M1NCM vs CadamNet')
    plt.subplot(2,3,5)
    bland_altman_plot(map_r2s_prop,map_r2s_pcanr,mask_liver_parenchyma_test),plt.title('PCANR vs CadamNet')
    plt.subplot(2,3,6)
    bland_altman_plot(map_r2s_prop,map_r2s_comp,mask_liver_parenchyma_test),plt.title('MappingNet vs CadamNet')
    plt.savefig(os.path.join('figure','baplot_r2comparison2.png'))

def R_square_map(im_p,im_t,degree=1):
    import numpy as np
    w,h,c = im_p.shape
    y_pred = np.reshape(im_p,[-1,12])
    y_true = np.reshape(im_t,[-1,12])

    im_rs = []
    for i in range(w*h):
        ssreg = np.sum((y_true[i]-y_pred[i])**2)
        sstot = np.sum((y_true[i]-y_true.mean())**2)
        rs = 1- ssreg/sstot
        im_rs.append(rs)
    im_rs = np.array(im_rs)
    im_rs = np.reshape(im_rs,[w,h])
    import os
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10)),plt.axis('off')
    plt.imshow(im_rs,cmap='jet',vmin=0.94,vmax=1),plt.colorbar(fraction=0.022)
    plt.savefig(os.path.join('figure','Rsquare_map.png'))
    print('>> R_square(max)='+str(np.max(im_rs))+' (min)='+str(np.min(im_rs)))
    return im_rs


def model_test_epoch(model_name,model_type,model_sigma,test_noise_level=13):
    import tensorflow as tf
    import os
    from metrics import OtoO_mapping
    import matplotlib.pyplot as plt
    import numpy as np
    # load model
    model_dir = os.path.join('model',model_name+'_sigma'+str(model_sigma))
    res = []
    res_each = []

    epoch = []
    for model_epoch in range(10,460,10):
        if model_epoch<100:
            model_file = os.path.join(model_dir,'model_0'+str(model_epoch)+'.h5')
        else:
            model_file = os.path.join(model_dir,'model_'+str(model_epoch)+'.h5')
        # check existence
        if os.path.exists(model_file):
            print(':: Load '+model_file)
            model = tf.keras.models.load_model(model_file,compile=False)
            # test model mapping performance
            train_noise_level = model_sigma
            result_mapping = OtoO_mapping(model,model_type=model_type,train_noise_level=train_noise_level,test_noise_level=test_noise_level)
            res.append(result_mapping['re_r2s'][-1])
            res_each.append(result_mapping['re_r2s'])
            epoch.append(model_epoch)
        else:
            print(':: '+ model_file+' not exist...')
    
    REs = np.array(res)
    Epoch = np.array(epoch)
    REs_each = np.array(res_each)

    plt.figure()
    # for i in range(21):
    #     plt.plot(Epoch,RMSEs_each[:,i],'*-')

    plt.plot(Epoch,REs,'*-r'),plt.xlabel('Epoch'),plt.ylabel('RE'),plt.title(model_name+'_sigma'+str(model_sigma))
    plt.savefig(os.path.join('figure','test_epoch.png'))

    
    print(REs)
    print(Epoch)
    # print(model)

def vivo_test(model_prop,model_comp,model_type_comp=2,model_type_prop=2,filepath='data_test_clinical'):
    # use the proposed model adn the compared model to predict the vivo data results
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import cv2

    from preprocessing import read_dcm

    # T2* weighted images
    data_normal=np.zeros([1,64,128,12])
    data_mild=np.zeros([1,64,128,12])
    data_moderate=np.zeros([1,64,128,12])
    # data_severe=np.zeros([1,64,128,12])
    data_severe=np.zeros([1,64,128,12])

    data_normal[0]=np.load(os.path.join(filepath,'data_normal.npy')) 
    data_mild[0]=np.load(os.path.join(filepath,'data_mild.npy')) 
    data_moderate[0]=np.load(os.path.join(filepath,'data_moderate.npy')) 
    # data_severe[0]=np.load(os.path.join(filepath,'data_severe.npy'))
    data_severe[0]=np.load(os.path.join('data_clinical','clinical_data.npy'))[111,:,:,:]
    # data_severe[0],_=read_dcm(os.path.join('data_test_clinical','Study20081127_133901_218000_severe_new','*.dcm'))


    # estimate the sigma of noise
    from metrics import get_sigma_g
    sigma_normal = get_sigma_g(data_normal)
    sigma_mild = get_sigma_g(data_mild)
    sigma_moderate = get_sigma_g(data_moderate)
    sigma_severe = get_sigma_g(data_severe)
    print('>> sigma(normal): '+str(sigma_normal)+' sigma(mild): '+str(sigma_mild)+' sigma(moderate):'+str(sigma_moderate)+' sigma(severe):'+str(sigma_severe))

    # body masks
    mask_normal = np.load(os.path.join(filepath,'mask_normal.npy')) 
    mask_mild = np.load(os.path.join(filepath,'mask_mild.npy')) 
    mask_moderate = np.load(os.path.join(filepath,'mask_moderate.npy')) 
    mask_severe = np.load(os.path.join(filepath,'mask_severe.npy'))

    # liver parenchyma masks
    from preprocessing import read_raw
    mask_parenchyma_normal = read_raw(os.path.join(filepath,'mask_parenchyma_normal.raw'),shape=(64,128))/255
    mask_parenchyma_mild = read_raw(os.path.join(filepath,'mask_parenchyma_mild.raw'),shape=(64,128))/255
    mask_parenchyma_moderate = read_raw(os.path.join(filepath,'mask_parenchyma_moderate2.raw'),shape=(64,128))/255
    mask_parenchyma_severe = read_raw(os.path.join(filepath,'mask_parenchyma_severe.raw'),shape=(72,128))/255


    if model_type_prop==2:
        _,map_normal_prop = model_prop.predict(data_normal)
        _,map_mild_prop = model_prop.predict(data_mild)
        _,map_moderate_prop = model_prop.predict(data_moderate)
        _,map_severe_prop = model_prop.predict(data_severe)
    else:
        map_normal_prop = model_prop.predict(data_normal)
        map_mild_prop = model_prop.predict(data_mild)
        map_moderate_prop = model_prop.predict(data_moderate)
        map_severe_prop = model_prop.predict(data_severe)

    if model_type_comp==2:
        _,map_normal_comp = model_comp.predict(data_normal)
        _,map_mild_comp = model_comp.predict(data_mild)
        _,map_moderate_comp = model_comp.predict(data_moderate)
        _,map_severe_comp = model_comp.predict(data_severe)
    else:
        map_normal_comp = model_comp.predict(data_normal)
        map_mild_comp = model_comp.predict(data_mild)
        map_moderate_comp = model_comp.predict(data_moderate)
        map_severe_comp = model_comp.predict(data_severe)

    maps_m1ncm = np.load(os.path.join(filepath,'map_r2s_m1ncm.npy')) 
    maps_pcanr1 = np.load(os.path.join(filepath,'map_r2s_pcanr1.npy')) 
    maps_pcanr2 = np.load(os.path.join(filepath,'map_r2s_pcanr2.npy')) 
    maps_pcanr3 = np.load(os.path.join(filepath,'map_r2s_pcanr3.npy')) 
    maps_pcanr4 = np.load(os.path.join(filepath,'map_r2s_pcanr4.npy'))

    maps_m1ncm[2,0:64,:] = np.load(os.path.join('data_test_clinical','R2s_moderate.npy'))[0]
    # maps_m1ncm[3] = np.load(os.path.join('data_test_clinical','R2s_severe.npy'))[0]


    mean_r2s_m1ncm1 = np.sum(maps_m1ncm[0,0:64,:]*mask_parenchyma_normal)/np.sum(mask_parenchyma_normal)
    mean_r2s_m1ncm2 = np.sum(maps_m1ncm[1,0:64,:]*mask_parenchyma_mild)/np.sum(mask_parenchyma_mild)
    mean_r2s_m1ncm3 = np.sum(maps_m1ncm[2,0:64,:]*mask_parenchyma_moderate)/np.sum(mask_parenchyma_moderate)
    # mean_r2s_m1ncm4 = np.sum(cv2.resize(maps_m1ncm[3],(128,64))*np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))/np.sum(np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))
    mean_r2s_m1ncm4 = np.sum(maps_m1ncm[3]*mask_parenchyma_severe)/np.sum(mask_parenchyma_severe)
    # mean_r2s_m1ncm4 = np.sum(maps_m1ncm[3]*mask_parenchyma_severe)/np.sum(mask_parenchyma_severe)

    mean_r2s_pcanr1 = np.sum(maps_pcanr1*mask_parenchyma_normal)/np.sum(mask_parenchyma_normal)
    mean_r2s_pcanr2 = np.sum(maps_pcanr2*mask_parenchyma_mild)/np.sum(mask_parenchyma_mild)
    mean_r2s_pcanr3 = np.sum(maps_pcanr3*mask_parenchyma_moderate)/np.sum(mask_parenchyma_moderate)
    mean_r2s_pcanr4 = np.sum(cv2.resize(maps_pcanr4,(128,64))*np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))/np.sum(np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))
    # mean_r2s_pcanr4 = np.sum(maps_pcanr4*mask_parenchyma_severe)/np.sum(mask_parenchyma_severe)
    # mean_r2s_pcanr4 = np.sum(maps_pcanr4*mask_parenchyma_severe)/np.sum(mask_parenchyma_severe)

    mean_r2s_comp1 = np.sum(map_normal_comp[0,:,:,1]*mask_parenchyma_normal)/np.sum(mask_parenchyma_normal)
    mean_r2s_comp2 = np.sum(map_mild_comp[0,:,:,1]*mask_parenchyma_mild)/np.sum(mask_parenchyma_mild)
    mean_r2s_comp3 = np.sum(map_moderate_comp[0,:,:,1]*mask_parenchyma_moderate)/np.sum(mask_parenchyma_moderate)
    mean_r2s_comp4 = np.sum(map_severe_comp[0,:,:,1]*np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))/np.sum(np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))
    # mean_r2s_comp4 = np.sum(map_severe_comp[0,:,:,1]*mask_parenchyma_severe)/np.sum(mask_parenchyma_severe)

    mean_r2s_prop1 = np.sum(map_normal_prop[0,:,:,1]*mask_parenchyma_normal)/np.sum(mask_parenchyma_normal)
    mean_r2s_prop2 = np.sum(map_mild_prop[0,:,:,1]*mask_parenchyma_mild)/np.sum(mask_parenchyma_mild)
    mean_r2s_prop3 = np.sum(map_moderate_prop[0,:,:,1]*mask_parenchyma_moderate)/np.sum(mask_parenchyma_moderate)
    mean_r2s_prop4 = np.sum(map_severe_prop[0,:,:,1]*np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))/np.sum(np.ceil(cv2.resize(mask_parenchyma_severe,(128,64))))
    # mean_r2s_prop4 = np.sum(map_severe_prop[0,:,:,1]*mask_parenchyma_severe)/np.sum(mask_parenchyma_severe)



    print('mean R2*(M1NCM): '+str([mean_r2s_m1ncm1,mean_r2s_m1ncm2,mean_r2s_m1ncm3,mean_r2s_m1ncm4]))
    print('mean R2*(PCANR): '+str([mean_r2s_pcanr1,mean_r2s_pcanr2,mean_r2s_pcanr3,mean_r2s_pcanr4]))
    print('mean R2*(comp): '+str([mean_r2s_comp1,mean_r2s_comp2,mean_r2s_comp3,mean_r2s_comp4]))
    print('mean R2*(prop): '+str([mean_r2s_prop1,mean_r2s_prop2,mean_r2s_prop3,mean_r2s_prop4]))

    colormap = 'jet'
    # colormap = 'gray'
    plt.figure(figsize=(50,30))

    plt.subplot(5,4,1),plt.axis('off'),plt.title('m1ncm',loc='left')
    plt.imshow(maps_m1ncm[0,0:64,:]*mask_normal,cmap=colormap,interpolation='none',vmin=0,vmax=300),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,2),plt.axis('off')
    plt.imshow(maps_m1ncm[1,0:64,:]*mask_mild,cmap=colormap,interpolation='none',vmin=0,vmax=500),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,3),plt.axis('off')
    plt.imshow(maps_m1ncm[2,0:64,:]*mask_moderate,cmap=colormap,interpolation='none',vmin=0,vmax=900),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,4),plt.axis('off')
    plt.imshow(cv2.resize(maps_m1ncm[3],(128,64))*mask_severe,cmap=colormap,interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)

    plt.subplot(5,4,5),plt.axis('off'),plt.title('pcanr',loc='left')
    plt.imshow(maps_pcanr1*mask_normal,cmap=colormap,interpolation='none',vmin=0,vmax=300),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,6),plt.axis('off')
    plt.imshow(maps_pcanr2*mask_mild,cmap=colormap,interpolation='none',vmin=0,vmax=500),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,7),plt.axis('off')
    plt.imshow(maps_pcanr3*mask_moderate,cmap=colormap,interpolation='none',vmin=0,vmax=900),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,8),plt.axis('off')
    plt.imshow(cv2.resize(maps_pcanr4,(128,64))*mask_severe,cmap=colormap,interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)


    plt.subplot(5,4,9),plt.axis('off'),plt.title('compared',loc='left')
    plt.imshow(map_normal_comp[0,:,:,1]*mask_normal,cmap=colormap,interpolation='none',vmin=0,vmax=300),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,10),plt.axis('off')
    plt.imshow(map_mild_comp[0,:,:,1]*mask_mild,cmap=colormap,interpolation='none',vmin=0,vmax=500),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,11),plt.axis('off')
    plt.imshow(map_moderate_comp[0,:,:,1]*mask_moderate,cmap=colormap,interpolation='none',vmin=0,vmax=900),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,12),plt.axis('off')
    plt.imshow(map_severe_comp[0,:,:,1]*mask_severe,cmap=colormap,interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)

    plt.subplot(5,4,13),plt.axis('off'),plt.title('proposed',loc='left')
    plt.imshow(map_normal_prop[0,:,:,1]*mask_normal,cmap=colormap,interpolation='none',vmin=0,vmax=300),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,14),plt.axis('off')
    plt.imshow(map_mild_prop[0,:,:,1]*mask_mild,cmap=colormap,interpolation='none',vmin=0,vmax=500),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,15),plt.axis('off')
    plt.imshow(map_moderate_prop[0,:,:,1]*mask_moderate,cmap=colormap,interpolation='none',vmin=0,vmax=900),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,16),plt.axis('off')
    plt.imshow(map_severe_prop[0,:,:,1]*mask_severe,cmap=colormap,interpolation='none',vmin=0,vmax=1000),plt.colorbar(fraction=0.022)

    plt.subplot(5,4,17),plt.axis('off'),plt.title('mask',loc='left')
    plt.imshow(mask_parenchyma_normal),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,18),plt.axis('off')
    plt.imshow(mask_parenchyma_mild),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,19),plt.axis('off')
    plt.imshow(mask_parenchyma_moderate),plt.colorbar(fraction=0.022)
    plt.subplot(5,4,20),plt.axis('off')
    plt.imshow(np.ceil(cv2.resize(mask_parenchyma_severe,(128,64)))),plt.colorbar(fraction=0.022)

    plt.savefig(os.path.join('figure','vivo_map'))


if __name__ == "__main__":
    # import skimage.metrics
    # import config
    # set gpu config
    # index_gpu = 6
    # print('>> set gpu '+str(index_gpu))

    # sigma = 7
    # model_sigma = 111
    # config.config_gpu(index_gpu)
    # model_test_epoch('UNet',model_type=1,model_sigma=sigma,test_noise_level=sigma)
    # model_test_epoch('DeepT2s(1)',model_type=2,model_sigma=model_sigma,test_noise_level=sigma)
    # model_test_epoch('DeepT2s',model_type=2,model_sigma=sigma,test_noise_level=sigma)
    # model_test_epoch('DeepT2s(2)',model_type=2,model_sigma=111,test_noise_level=13)


    # from vis import show_loss,show_loss_val
    # show_loss('UNet',model_sigma=5,model_type=1)
    # show_loss('DeepT2s(2)',model_sigma=model_sigma,model_type=2)
    # show_loss('DeepT2s(1)',model_sigma=model_sigma,model_type=2)
    # show_loss_val('DeepT2s(1)_100epoch',model_sigma=model_sigma,model_type=2)
    # show_loss_val('DeepT2s(2)',model_sigma=model_sigma,model_type=2)
    # show_loss_val_2(model1='DeepT2s(1)', model2='UNet', sigma=model_sigma)
    # show_loss_val_2(model1='DeepT2s(2)', model2='UNet', sigma=model_sigma)
    # show_loss_val_2(model1='DeepT2s(2)', model2='DeepT2s(1)_100epoch', sigma=model_sigma)




    # from data_generator import split_data_trian_patches
    # split_data_trian_patches(sigma_g=17)

    import os
    import numpy as np 
    import pandas
    import matplotlib.pyplot as plt

    model1='DeepT2s(2)'
    model2='DeepT2s(1)_100epoch'
    # model3='UNet_200epoch'
    model3='UNet_full_200epoch'

    sigma = 111

    # load results data
    model1_dir = os.path.join('model',model1+'_sigma'+str(sigma))
    model2_dir = os.path.join('model',model2+'_sigma'+str(sigma))
    model3_dir = os.path.join('model',model3+'_sigma'+str(sigma))
    losses1 = pandas.read_csv(os.path.join(model1_dir,'log.csv'))
    losses2 = pandas.read_csv(os.path.join(model2_dir,'log.csv'))
    losses3 = pandas.read_csv(os.path.join(model3_dir,'log.csv'))

    # validation loss
    # loss1 = losses1[losses1.columns.values[2]][0:101]
    # loss2 = losses2[losses2.columns.values[1]][0:101]
    loss1 = losses1[losses1.columns.values[5]][0:200]
    loss2 = losses2[losses2.columns.values[5]][0:200]
    loss3 = losses3[losses3.columns.values[2]][0:200]

    # training loss
    loss4 = losses1[losses1.columns.values[2]][0:200]
    loss5 = losses2[losses2.columns.values[2]][0:200]
    loss6 = losses3[losses3.columns.values[1]][0:200]

    print(loss2.shape)

    plt.figure(figsize=(7.5,7.5))
    plt.title('validation loss ($\sigma_g$='+str(sigma)+')')
    # plt.plot(np.sqrt(loss1*2/(128*128*12)),'k',label=model1)
    # plt.plot(np.sqrt(loss2*2/(128*128*12)),'b',label=model2)
    # plt.plot(loss1/np.max(loss2),'k',label=model1)
    # plt.plot(loss2/np.max(loss2),'b',label=model2)
    plt.plot(np.log(loss1),'r',label='CadamNet(2)(valid)')
    plt.plot(np.log(loss2),'k',label='CadamNet(1)(valid)')
    plt.plot(np.log(loss3),'b',label='MappingNet(valid)')

    # plt.plot(np.log(loss4),'r--',label='CadamNet(2)(train)')
    # plt.plot(np.log(loss5),'k--',label='CadamNet(1)(train)')
    # plt.plot(np.log(loss6),'b--',label='MappingNet(train)')

    plt.legend()
    plt.savefig(os.path.join('figure','validatation_loss'))




    