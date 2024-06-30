import tensorflow as tf
import helper

def SeparableCNN(depth=8,depth_multi=10,filters=40, image_channels=12,dilation=1):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))
    x = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),kernel_initializer='Orthogonal',padding='same',
                                        dilation_rate=(dilation,dilation),depth_multiplier=depth_multi)(inpt)
    x = tf.keras.layers.Activation('relu')(x)
    for i in range(depth-2):
        x = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),kernel_initializer='Orthogonal',padding='same',
                                        dilation_rate=(dilation,dilation),depth_multiplier=depth_multi)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
        x = tf.keras.layers.Activation('relu')(x)  
    # last layer, Conv
    x = tf.keras.layers.SeparableConv2D(filters=12,kernel_size=(3,3),strides=(1,1),kernel_initializer='Orthogonal',padding='same',
                                        dilation_rate=(dilation,dilation),depth_multiplier=depth_multi)(x)
    # ResNet architecture
    x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model

def UNetH(image_channels=12,output_channels=2):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))
    
    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(inpt) # 64*128
    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv1)  # 64*128
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1) # 32*64
    
    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2) # 16*32
    
    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv3)
    
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3) # 8*16
    
    conv4 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4) # 4*8
    
    convbase = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool4)    # 4*8
    convbase = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(convbase) # 4*8
    
    conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) # 8*16*1024
    
    conv5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc5) # 8*16*512
    conv5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv5) # 8*16*512
    
    conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3]) # 16*32*512
    
    conv6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc6) # 16*32*256
    conv6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv6) # 16*32*256
    
    conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2]) # 32*64*256
    
    conv7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc7) # 32*64*128
    conv7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv7) # 32*64*128
    
    conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(32, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1]) # 64*128*128
    
    conv8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conc8) # 64*128*64
    conv8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv8) # 64*128*64
    
    convone = tf.keras.layers.Conv2D(output_channels,(1,1),activation='relu')(conv8) # 64*128*3
    
    model = tf.keras.Model(inputs=inpt,outputs=convone)
    
    return model

def UNet(image_channels=12,output_channel=2,dilation_rate=(1,1)):
    """
    Original U-Net architecture.
    """
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels)) # 12
    
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(inpt)   # 64
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv1)  # 64
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1) # 64
    
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool1) # 128
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv2) # 128
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2) # 128
    
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool2) # 256
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv3) # 256
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3) # 256
    
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool3) # 512
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv4) # 512
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4) # 512
    
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)    #1024
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(convbase) #1024
    
    conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(512,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) #1024
    
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conc5) # 512
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv5) # 512
    
    conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3]) # 512
    
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc6) # 256
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv6) # 256
    
    conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2]) # 256
    
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc7) # 128
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv7) # 128
    
    conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1]) # 128
    
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc8) # 64
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',dilation_rate=dilation_rate)(conv8) # 64
    
    convone = tf.keras.layers.Conv2D(output_channel,(1,1),activation='relu')(conv8) # 2

    # s0 = tf.keras.layers.Activation(helper.rangeR)(convone[...,0])
    # r2 = tf.keras.layers.Activation(helper.rangeR)(convone[...,1])
    # sg = tf.keras.layers.Activation(helper.rangeR)(convone[...,2])

    # s0 = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x,1000.0))(convone[...,0])
    # r2 = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x,2000.0))(convone[...,1])
    # sg = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x,20.0))(convone[...,2])

    # p  = tf.split(convone,3,3)
    # s0 = tf.keras.layers.Lambda(lambda x: 0.0+(1000.0-0.0)*(1.0+tf.math.tanh(x))/2.0)(p[0])
    # r2 = tf.keras.layers.Lambda(lambda x: 30.0+(2500.0-30.0)*(1.0+tf.math.tanh(x))/2.0)(p[1])
    # sg = tf.keras.layers.Lambda(lambda x: 2.0+(100.0-2.0)*(1.0+tf.math.tanh(x))/2.0)(p[2])
    # convone = tf.keras.layers.Concatenate()([s0,r2,sg])
    
    model = tf.keras.Model(inputs=inpt,outputs=convone)
    
    return model

def MANTIS(image_channels=12,residual=False):
    """
    U-Net used in MANTIS architecture.
    """
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels)) # 128*64*12

    conv1 = tf.keras.layers.Conv2D(64,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(inpt) # 64*32*64
    conv1 = tf.keras.layers.BatchNormalization(axis=-1)(conv1)

    conv2 = tf.keras.layers.Conv2D(128,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv1) #32*16*128
    conv2 = tf.keras.layers.BatchNormalization(axis=-1)(conv2)

    conv3 = tf.keras.layers.Conv2D(256,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv2) #16*8*256
    conv3 = tf.keras.layers.BatchNormalization(axis=-1)(conv3)

    conv4 = tf.keras.layers.Conv2D(512,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv3) #8*4*512
    conv4 = tf.keras.layers.BatchNormalization(axis=-1)(conv4)

    conv5 = tf.keras.layers.Conv2D(512,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv4) #4*2*512
    conv5 = tf.keras.layers.BatchNormalization(axis=-1)(conv5)

    conv6 = tf.keras.layers.Conv2D(512,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv5) #2*1*512
    conv6 = tf.keras.layers.BatchNormalization(axis=-1)(conv6)

    conv7 = tf.keras.layers.Conv2D(512,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv6) #1*1*512
    conv7 = tf.keras.layers.BatchNormalization(axis=-1)(conv7)

    conv8 = tf.keras.layers.Conv2DTranspose(512,kernel_size=(4,4),stride=(2,2),activation='relu',padding='same')(conv7)#2*2*512
    conv8 = tf.keras.layers.Concatenate()([])

    model = tf.keras.Model(inputs=inpt,outputs=conv8)
    
    return model

def DnCNNn(depth,filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    layer_count += 1
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            #x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = tf.keras.layers.Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = tf.keras.layers.Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
    model = tf.keras.Model(inputs=inpt, outputs=x)
    
    return model

def DnCNN(image_channels=12,output_channels=3):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)  

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x) 

    x = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(3,3), strides=(1,1),padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x) 

    # x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model

def DnCNN3(image_channels=12):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))

    model_S0 = DnCNN(image_channels=12,output_channels=1)
    model_R2 = DnCNN(image_channels=12,output_channels=1)
    model_Sg = DnCNN(image_channels=12,output_channels=1)

    S0 = model_S0(inpt)
    R2 = model_R2(inpt)
    Sg = model_Sg(inpt)

    outpt = tf.keras.layers.Concatenate()([S0,R2,Sg])
    model = tf.keras.Model(inputs=inpt, outputs=outpt)
    return model

def UNetH3(image_channels=12):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))

    model_S0 = UNetH(image_channels=12,output_channels=1)
    model_R2 = UNetH(image_channels=12,output_channels=1)
    model_Sg = UNetH(image_channels=12,output_channels=1)

    S0 = model_S0(inpt)
    R2 = model_R2(inpt)
    Sg = model_Sg(inpt)

    outpt = tf.keras.layers.Concatenate()([S0,R2,Sg])
    model = tf.keras.Model(inputs=inpt, outputs=outpt)
    return model

def DnCNN2(image_channels=12):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))

    model_S0 = DnCNN(image_channels=12,output_channels=1)
    model_R2 = DnCNN(image_channels=12,output_channels=1)

    S0 = model_S0(inpt)
    R2 = model_R2(inpt)

    outpt = tf.keras.layers.Concatenate()([S0,R2])
    model = tf.keras.Model(inputs=inpt, outputs=outpt)
    return model  

def UNetDouble(image_channels=12,out1=2,out2=1):
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels)) # 12
    
    ##### Encoder
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(inpt)   # 64
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv1)  # 64
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1) # 64
    
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool1) # 128
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv2) # 128
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2) # 128
    
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool2) # 256
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv3) # 256
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3) # 256
    
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool3) # 512
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv4) # 512
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4) # 512
    
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)    #1024
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(convbase) #1024

    ##### Mapping Decoder
    conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(512,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) #1024
    
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conc5) # 512
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv5) # 512
    
    conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3]) # 512
    
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc6) # 256
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv6) # 256
    
    conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2]) # 256
    
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc7) # 128
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv7) # 128
    
    conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1]) # 128
    
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc8) # 64
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv8) # 64
    
    convone = tf.keras.layers.Conv2D(out1,(1,1),activation='relu',padding='same')(conv8) # 2

    ##### Denoising Decoder
    conc52 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(512,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) #1024
    
    conv52 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conc52) # 512
    conv52 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv52) # 512
    
    conc62 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv52)),conv3]) # 512
    
    conv62 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc62) # 256
    conv62 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv62) # 256
    
    conc72 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv62)),conv2]) # 256
    
    conv72 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc72) # 128
    conv72 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv72) # 128
    
    conc82 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv72)),conv1]) # 128
    
    conv82 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc82) # 64
    conv82 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv82) # 64
    
    convone2 = tf.keras.layers.Conv2D(out2,(1,1),activation='relu',padding='same')(conv82) # 2
    
    outpt = tf.keras.layers.Concatenate()([convone,convone2])
    model = tf.keras.Model(inputs=inpt,outputs=outpt)
    
    return model