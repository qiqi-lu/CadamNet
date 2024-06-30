def show_loss(model_name,model_sigma,model_type=2):
    import os
    import numpy as np 
    import pandas
    import matplotlib.pyplot as plt
    model_dir = os.path.join('model',model_name+'_sigma'+str(model_sigma))
    losses = pandas.read_csv(os.path.join(model_dir,'log.csv'))
    if model_type==2:
        loss_sum = losses[losses.columns.values[1]]
        loss_denoise = losses[losses.columns.values[2]]
        loss_mapping = losses[losses.columns.values[3]]
        plt.figure()
        plt.title(model_name+' model losses '+str(model_sigma))
        plt.plot(np.log(loss_sum),'k',label='Loss')
        plt.plot(np.log(loss_denoise),'b',label='Denoise loss')
        plt.plot(np.log(loss_mapping),'m',label='Recreate loss')
        plt.legend()
        plt.savefig(os.path.join('figure','losses_'+model_name))
    else:
        loss = losses[losses.columns.values[1]]
        plt.figure(),plt.title(model_name+' model losses')
        plt.plot(np.log(loss),'k',label='Loss')
        plt.legend()
        plt.savefig(os.path.join('figure','losses_'+model_name))

def show_loss_val(model_name,model_sigma,model_type=2):
    import os
    import numpy as np 
    import pandas
    import matplotlib.pyplot as plt
    model_dir = os.path.join('model',model_name+'_sigma'+str(model_sigma))
    losses = pandas.read_csv(os.path.join(model_dir,'log.csv'))
    if model_type==2:
        loss_sum = losses[losses.columns.values[4]]
        loss_denoise = losses[losses.columns.values[5]]
        loss_mapping = losses[losses.columns.values[6]]
        plt.figure()
        plt.title(model_name+' model losses '+str(model_sigma))
        # plt.plot(np.log(loss_sum),'k',label='Loss')
        # plt.plot(np.log(loss_denoise),'b',label='Denoise loss')
        # plt.plot(np.log(loss_mapping),'m',label='Recreate loss')
        plt.plot(np.sqrt(loss_sum*2/(128*128*12)),'k',label='Loss')
        plt.plot(np.sqrt(loss_denoise*2/(128*128*12)),'b',label='Denoise loss')
        # plt.plot(np.sqrt(loss_mapping*2/(128*128*12)),'m',label='Recreate loss')
        plt.legend()
        plt.savefig(os.path.join('figure','losses_val_'+model_name))
    else:
        loss = losses[losses.columns.values[4]]
        plt.figure(),plt.title(model_name+' model losses')
        plt.plot(np.log(loss),'k',label='Loss')
        plt.legend()
        plt.savefig(os.path.join('figure','losses_val_'+model_name))
    

def show_vivo_mapping_result(model,model_type=2):
    from data_generator import get_vivo_data
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    data_vivo = get_vivo_data()
    # from data_checker import check_noise
    # data_vivo_noise = check_noise(data_vivo)
    data_vivo = data_vivo.astype('float32')

    from data_generator import get_mask_test_clinical
    mask_body,mask_liver,_ = get_mask_test_clinical()

    # test model denoising on vivo data
    if model_type==2:
        xp,mapp = model.predict(data_vivo)
    else:
        mapp = model.predict(data_vivo)

    # load pcanr results
    map_r2s_pcanr = np.load(os.path.join('data_test_clinical','clinical_data_test_map_r2s_pcanr.npy'))
    # load m1ncm results
    map_r2s_m1ncm = np.load(os.path.join('data_test_clinical','clinical_data_test_map_r2s_m1ncm.npy')) 

    # show vivo data test results
    plt.figure(figsize=(30,20))
    cm = ['gray','jet']
    vmaxs = [300,500,700,2500]
    vmaxss = [700,650,500,400]
    row,col=4,4
    i=0
    plt.subplot(row,col,i+1),plt.axis('off'),plt.title('T2*w image',loc='left')
    plt.imshow(data_vivo[0,:,:,0]*mask_body[0],cmap=cm[0],interpolation='none',vmin=0,vmax=vmaxss[0]),plt.title('Normal'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+2),plt.axis('off')
    plt.imshow(data_vivo[1,:,:,0]*mask_body[1],cmap=cm[0],interpolation='none',vmin=0,vmax=vmaxss[1]),plt.title('Mild'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+3),plt.axis('off')
    plt.imshow(data_vivo[2,:,:,0]*mask_body[2],cmap=cm[0],interpolation='none',vmin=0,vmax=vmaxss[2]),plt.title('Moderate'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+4),plt.axis('off')
    plt.imshow(data_vivo[3,:,:,0]*mask_body[3],cmap=cm[0],interpolation='none',vmin=0,vmax=vmaxss[3]),plt.title('Severe'),plt.colorbar(fraction=0.022)

    i=4
    plt.subplot(row,col,i+1),plt.axis('off'),plt.title('M1NCM',loc='left')
    plt.imshow(map_r2s_m1ncm[0,:,:]*mask_body[0],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[0]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+2),plt.axis('off')
    plt.imshow(map_r2s_m1ncm[1,:,:]*mask_body[1],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[1]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+3),plt.axis('off')
    plt.imshow(map_r2s_m1ncm[2,:,:]*mask_body[2],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[2]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+4),plt.axis('off')
    plt.imshow(map_r2s_m1ncm[3,:,:]*mask_body[3],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[3]),plt.colorbar(fraction=0.022)

    i=8
    plt.subplot(row,col,i+1),plt.axis('off'),plt.title('PCANR',loc='left')
    plt.imshow(map_r2s_pcanr[0,:,:]*mask_body[0],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[0]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+2),plt.axis('off')
    plt.imshow(map_r2s_pcanr[1,:,:]*mask_body[1],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[1]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+3),plt.axis('off')
    plt.imshow(map_r2s_pcanr[2,:,:]*mask_body[2],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[2]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+4),plt.axis('off')
    plt.imshow(map_r2s_pcanr[3,:,:]*mask_body[3],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[3]),plt.colorbar(fraction=0.022)

    i=12
    plt.subplot(row,col,i+1),plt.axis('off'),plt.title('DeepT2s',loc='left')
    plt.imshow(mapp[0,:,:,1]*mask_body[0],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[0]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+2),plt.axis('off')
    plt.imshow(mapp[1,:,:,1]*mask_body[1],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[1]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+3),plt.axis('off')
    plt.imshow(mapp[2,:,:,1]*mask_body[2],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[2]),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,i+4),plt.axis('off')
    plt.imshow(mapp[3,:,:,1]*mask_body[3],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[3]),plt.colorbar(fraction=0.022)

    plt.savefig(os.path.join('figure','result_vivo_maping'))

def show_test_mapping_results(result,test_noise_level,study_id=None):
    # load test data
    import os
    import numpy as np
    data_test_name = 'simulated_data_test_'+str(test_noise_level)+'.npy'
    data_test = np.load(os.path.join('data_test_simulated',data_test_name),allow_pickle=True).item()
    data_noise_test = data_test['noise data']
    map_r2s_test    = data_test['r2s map']
    num_study_test  = data_noise_test.shape[0]

    # load test data map
    import matplotlib.pyplot as plt
    maps_pred  = result['mapp']
    map_r2s_pcanr = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'_map_r2s_pcanr.npy')) 
    map_r2s_m1ncm = np.load(os.path.join('data_test_simulated','simulated_data_test_'+str(test_noise_level)+'_map_r2s_m1ncm.npy')) 

    # load test data masks
    from data_generator import get_mask_test, get_mask_liver
    masks_body_test = get_mask_test() # body mask
    map_r2s_test    = map_r2s_test*masks_body_test

    mask_liver_whole,mask_liver_parenchyma = get_mask_liver() # whole liver and parenchyma mask
    mask_liver_whole_test = mask_liver_whole[-21:]
    mask_liver_parenchyma_test = mask_liver_parenchyma[-21:]

    map_r2s_pred  = maps_pred[:,:,:,1]*masks_body_test
    map_r2s_pcanr = map_r2s_pcanr*masks_body_test
    map_r2s_m1ncm = map_r2s_m1ncm*masks_body_test

    # vmaxs = [800,400,400,600,600,400,400,300,300,300,
    #         300,1000,400,600,400,600,1100,900,400,1000,
    #         500]
    vmaxs = [800,240,400,500,600,200,350,200,200,150,
            150,1000,200,600,125,350,1100,900,300,3000,
            300]
    vmins = 0

    # show mappping result from testing set
    ite = 0
    cm = ['gray','jet']
    if study_id == None:
        l = range(num_study_test)
    else:
        l = [study_id]

    for i in l:
        plt.figure(figsize=(20,10))
        plt.subplot(241),plt.axis('off')
        plt.imshow(data_noise_test[i,:,:,ite],cmap=cm[0],interpolation='none',vmin=0,vmax=300),plt.title('Noisy T2*w image (TE'+str(ite)+')'),plt.colorbar(fraction=0.022)
        
        plt.subplot(242),plt.axis('off')
        plt.imshow(map_r2s_m1ncm[i],interpolation='none',cmap='jet',vmin=vmins,vmax=vmaxs[i]),plt.title('M1NCM'),plt.colorbar(fraction=0.022)
        plt.subplot(243),plt.axis('off')
        plt.imshow(map_r2s_pcanr[i],interpolation='none',cmap='jet',vmin=vmins,vmax=vmaxs[i]),plt.title('PCANR'),plt.colorbar(fraction=0.022)
        plt.subplot(244),plt.axis('off')
        plt.imshow(map_r2s_pred[i,:,:],cmap=cm[1],interpolation='none',vmin=vmins,vmax=vmaxs[i]),plt.title('CadamNet'),plt.colorbar(fraction=0.022)
        
        plt.subplot(245),plt.axis('off')
        plt.imshow(map_r2s_test[i,:,:],cmap=cm[1],interpolation='none',vmin=vmins,vmax=vmaxs[i]),plt.title('Ground Truth R2*'),plt.colorbar(fraction=0.022)
        
        up=200
        plt.subplot(246),plt.axis('off')
        plt.imshow(np.abs(map_r2s_test[i]-map_r2s_m1ncm[i]),cmap=cm[1],interpolation='none',vmin=0,vmax=up),plt.title('M1NCM Difference'),plt.colorbar(fraction=0.022)
        plt.subplot(247),plt.axis('off')
        plt.imshow(np.abs(map_r2s_test[i]-map_r2s_pcanr[i]),cmap=cm[1],interpolation='none',vmin=0,vmax=up),plt.title('PCANR Difference'),plt.colorbar(fraction=0.022)
        plt.subplot(248),plt.axis('off')
        plt.imshow(np.abs(map_r2s_pred[i]-map_r2s_test[i]), cmap=cm[1],interpolation='none',vmin=0,vmax=up),plt.title('DeepT2s Difference'),plt.colorbar(fraction=0.022)
        # plt.tight_layout()
        plt.savefig(os.path.join('figure','result_mapping_study'+str(i)))

        np.save(os.path.join('data_test_simulated','map_r2s_'+str(i)),map_r2s_test[i])
        np.save(os.path.join('data_test_simulated','mask_'+str(i)),masks_body_test[i])
        np.save(os.path.join('data_test_simulated','mask_'+str(i)),mask_liver_whole_test[i])



def show_test_denoising_result(result,test_noise_level,model_name='DeepT2s',study_id=None):
    # load test data
    import os
    import numpy as np
    data_test_name = 'simulated_data_test_'+str(test_noise_level)+'.npy'
    data_test = np.load(os.path.join('data_test_simulated',data_test_name),allow_pickle=True).item()
    data_noise_test = data_test['noise data']
    data_noise_free_test = data_test['noise free data']
    num_study_test  = data_noise_test.shape[0]

    # load test data map
    import matplotlib.pyplot as plt
    xp  = result['xp']
    up  = 150
    upr = 50
    ite = [0,6,11]
    r,c = 3,6
    cm = 'gray'

    if study_id == None:
        l = range(num_study_test)
    else:
        l = [study_id]


    for i in l:
        plt.figure(figsize=(30,10))
        plt.subplot(r,c,1),plt.axis('off'),plt.title('Noise-free',loc='left')
        plt.imshow(data_noise_free_test[i,:,:,ite[0]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.title('TE '+str(ite[0]+1)),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,3),plt.axis('off')
        plt.imshow(data_noise_free_test[i,:,:,ite[1]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.title('TE '+str(ite[1]+1)),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,5),plt.axis('off')
        plt.imshow(data_noise_free_test[i,:,:,ite[2]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.title('TE '+str(ite[2]+1)),plt.colorbar(fraction=0.022)

        plt.subplot(r,c,7),plt.axis('off'),plt.title('Noisy',loc='left')
        plt.imshow(data_noise_test[i,:,:,ite[0]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,8),plt.axis('off')
        plt.imshow(np.abs(data_noise_test[i,:,:,ite[0]]-data_noise_free_test[i,:,:,ite[0]]),cmap=cm,interpolation='none',vmin=0,vmax=upr),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,9),plt.axis('off')
        plt.imshow(data_noise_test[i,:,:,ite[1]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,10),plt.axis('off')
        plt.imshow(np.abs(data_noise_test[i,:,:,ite[1]]-data_noise_free_test[i,:,:,ite[1]]),cmap=cm,interpolation='none',vmin=0,vmax=upr),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,11),plt.axis('off')
        plt.imshow(data_noise_test[i,:,:,ite[2]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,12),plt.axis('off')
        plt.imshow(np.abs(data_noise_test[i,:,:,ite[2]]-data_noise_free_test[i,:,:,ite[2]]),cmap=cm,interpolation='none',vmin=0,vmax=upr),plt.colorbar(fraction=0.022)

        plt.subplot(r,c,13),plt.axis('off'),plt.title(model_name,loc='left')
        plt.imshow(xp[i,:,:,ite[0]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,14),plt.axis('off')
        plt.imshow(np.abs(xp[i,:,:,ite[0]]-data_noise_test[i,:,:,ite[0]]),cmap=cm,interpolation='none',vmin=0,vmax=upr),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,15),plt.axis('off')
        plt.imshow(xp[i,:,:,ite[1]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,16),plt.axis('off')
        plt.imshow(np.abs(xp[i,:,:,ite[1]]-data_noise_test[i,:,:,ite[1]]),cmap=cm,interpolation='none',vmin=0,vmax=upr),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,17),plt.axis('off')
        plt.imshow(xp[i,:,:,ite[2]],cmap=cm,interpolation='none',vmin=0,vmax=up),plt.colorbar(fraction=0.022)
        plt.subplot(r,c,18),plt.axis('off')
        plt.imshow(np.abs(xp[i,:,:,ite[2]]-data_noise_test[i,:,:,ite[2]]),cmap=cm,interpolation='none',vmin=0,vmax=upr),plt.colorbar(fraction=0.022)
        plt.savefig(os.path.join('figure','result_denoise_study'+str(i)))


    j=0
    for i in range(12):
        plt.figure()
        plt.subplot(2,2,1),plt.axis('off')
        plt.imshow(xp[j,:,:,i],cmap='jet',interpolation='none',vmin=0,vmax=150),plt.title('Study '+str(j)+' TE '+str(i)+'(pred)')
        plt.subplot(2,2,2),plt.axis('off')
        plt.imshow(data_noise_free_test[j,:,:,i],cmap='jet',interpolation='none',vmin=0,vmax=150),plt.title('Study '+str(j)+' TE '+str(i)+'(nf)')
        plt.subplot(2,2,3),plt.axis('off')
        plt.imshow(np.abs(xp[j,:,:,i]-data_noise_free_test[j,:,:,i]),cmap='jet',interpolation='none',vmin=0,vmax=15),plt.title('Study '+str(j)+' TE '+str(i)+'(residual)')
        plt.savefig(os.path.join('figure','result_denoise_study_multi'+str(j)+str(i)))

def show_vivo_denoising_result(model,model_type=2):
    from data_generator import get_vivo_data
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    data_vivo = get_vivo_data()
    # from data_checker import check_noise
    # data_vivo_noise = check_noise(data_vivo)
    data_vivo = data_vivo.astype('float32')

    # test model denoising on vivo data
    if model_type==2:
        xp,mapp = model.predict(data_vivo)
    else:  
        xp = model.predict(data_vivo)

    cm = ['jet','gray','gray']
    from data_generator import get_mask_test_clinical
    mask_body,mask_liver,_ = get_mask_test_clinical()

    # mask_body = mask_liver
    mask_body = np.ones(mask_body.shape)

    i = 11
    r,c = 3,4
    vmaxs  = [400,100,50,50]
    vmaxss = [40,40,40,40]
    plt.figure(figsize=(20,10))

    plt.subplot(r,c,1),plt.axis('off'),plt.title('Noisy image',loc='left')
    plt.imshow(data_vivo[0,:,:,i]*mask_body[0],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[0]),plt.title('Normal'),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,2),plt.axis('off')
    plt.imshow(data_vivo[1,:,:,i]*mask_body[1],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[1]),plt.title('Mild'),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,3),plt.axis('off')
    plt.imshow(data_vivo[2,:,:,i]*mask_body[2],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[2]),plt.title('Moderate'),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,4),plt.axis('off')
    plt.imshow(data_vivo[3,:,:,i]*mask_body[3],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[3]),plt.title('Severe'),plt.colorbar(fraction=0.022)

    plt.subplot(r,c,5),plt.axis('off'),plt.title('DeepT2s',loc='left')
    plt.imshow(xp[0,:,:,i]*mask_body[0],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[0]),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,6),plt.axis('off')
    plt.imshow(xp[1,:,:,i]*mask_body[1],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[1]),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,7),plt.axis('off')
    plt.imshow(xp[2,:,:,i]*mask_body[2],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[2]),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,8),plt.axis('off')
    plt.imshow(xp[3,:,:,i]*mask_body[3],cmap=cm[1],interpolation='none',vmin=0,vmax=vmaxs[3]),plt.colorbar(fraction=0.022)

    plt.subplot(r,c,9),plt.axis('off'),plt.title('Residual of DeepT2s',loc='left')
    plt.imshow(np.abs(data_vivo[0,:,:,i]-xp[0,:,:,i])*mask_body[0],cmap=cm[2],interpolation='none',vmin=0,vmax=vmaxss[0]),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,10),plt.axis('off')
    plt.imshow(np.abs(data_vivo[1,:,:,i]-xp[1,:,:,i])*mask_body[1],cmap=cm[2],interpolation='none',vmin=0,vmax=vmaxss[1]),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,11),plt.axis('off')
    plt.imshow(np.abs(data_vivo[2,:,:,i]-xp[2,:,:,i])*mask_body[2],cmap=cm[2],interpolation='none',vmin=0,vmax=vmaxss[2]),plt.colorbar(fraction=0.022)
    plt.subplot(r,c,12),plt.axis('off')
    plt.imshow(np.abs(data_vivo[3,:,:,i]-xp[3,:,:,i])*mask_body[3],cmap=cm[2],interpolation='none',vmin=0,vmax=vmaxss[3]),plt.colorbar(fraction=0.022)

    plt.savefig(os.path.join('figure','result_vivo_denoise'))
