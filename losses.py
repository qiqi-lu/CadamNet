import tensorflow as tf

def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true))/2

def loss_model_exp_image(y_true,y_pred):
    # y_pred: [batch_size, 64, 128,2]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # y_pred =y_pred
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    y_true = tf.reshape(y_true,[-1,12])
    recon  = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])/1000.0*tes)
    # recon = tf.reshape(recon,[-1,32,32,12])
    loss = tf.keras.backend.sum(tf.keras.backend.square(recon - y_true))/2
    return loss

def exp_square(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # y_pred = tf.clip_by_value(y_pred,clip_value_min=0.0001,clip_value_max=5000.0)
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    y_true = tf.math.square(tf.reshape(y_true,[-1,12]))
    recon  = tf.math.square(tf.reshape(y_pred[:,0],[-1,1]))*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0) + 2.0*tf.math.square(tf.reshape(y_pred[:,2],[-1,1]))
    loss   = tf.math.reduce_mean(tf.math.square(recon - y_true))
    return loss

def exp_sqrt(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred = tf.clip_by_value(y_pred,clip_value_min=0.01,clip_value_max=5000.0)
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    y_true = tf.reshape(y_true,[-1,12])
    recon  = tf.math.square(tf.reshape(y_pred[:,0],[-1,1]))*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0) + 2.0*tf.math.square(tf.reshape(y_pred[:,2],[-1,1]))
    recon  = tf.math.sqrt(recon)
    loss   = tf.math.reduce_mean(tf.math.square(recon - y_true))
    return loss

def exp(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # y_pred = tf.clip_by_value(y_pred,clip_value_min=0.0001,clip_value_max=2000.0)
    y_pred = tf.reshape(y_pred,[-1,tf.shape(y_pred)[-1]])
    y_true = tf.reshape(y_true,[-1,12])
    recon  = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0)
    loss   = tf.math.reduce_mean(tf.math.square(recon - y_true))
    return loss

def exps(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred = tf.clip_by_value(y_pred,clip_value_min=0.0001,clip_value_max=3000.0)
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    y_true = tf.math.square(tf.reshape(y_true,[-1,12]))
    recon  = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0)
    recon  = tf.square(recon)
    loss   = tf.math.reduce_mean(tf.math.square(recon - y_true))
    return loss

def tv(y_true,y_pred):
    y_pred = tf.clip_by_value(y_pred,clip_value_min=0.001,clip_value_max=5000.0)
    loss = tf.reduce_sum(tf.image.total_variation(y_pred))
    return loss

def tv5(y_true,y_pred):
    Ix = y_pred[:, 1:, 1:, :] - y_pred[:, :-1, 1:, :]
    Iy = y_pred[:, 1:, 1:, :] - y_pred[:, 1:, :-1, :]
    grad = tf.abs(Ix) + tf.abs(Iy)
    loss = tf.reduce_sum(tf.image.total_variation(grad))
    return loss

def tv6(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred = tf.reshape(y_pred,[-1,tf.shape(y_pred)[-1]])
    y_pred = tf.clip_by_value(y_pred,clip_value_min=0.0001,clip_value_max=3000.0)
    recon  = tf.math.square(tf.reshape(y_pred[:,0],[-1,1]))*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0) + 2.0*tf.math.square(tf.reshape(y_pred[:,2],[-1,1]))
    recon  = tf.sqrt(tf.reshape(recon,tf.shape(y_true)))
    loss   = tf.reduce_sum(tf.image.total_variation(recon))
    return loss

def tvw(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # y_pred = tf.clip_by_value(y_pred,clip_value_min=0.001,clip_value_max=3000.0)
    y_pred = tf.reshape(y_pred,[-1,tf.shape(y_pred)[-1]])
    recon  = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0)
    recon  = tf.reshape(recon,tf.shape(y_true))
    loss   = tf.reduce_sum(tf.image.total_variation(recon))
    return loss

def tv7(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred = tf.reshape(y_pred,[-1,tf.shape(y_pred)[-1]])
    y_pred = tf.clip_by_value(y_pred,clip_value_min=0.0001,clip_value_max=3000.0)
    recon  = tf.math.square(tf.reshape(y_pred[:,0],[-1,1]))*tf.exp(-2.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0) + 2.0*tf.math.square(tf.reshape(y_pred[:,2],[-1,1]))
    recon  = tf.sqrt(tf.reshape(recon,tf.shape(y_true)))
    Ix = recon[:, 1:, 1:, :] - recon[:, :-1, 1:, :]
    Iy = recon[:, 1:, 1:, :] - recon[:, 1:, :-1, :]
    grad = tf.abs(Ix) + tf.abs(Iy)
    loss = tf.reduce_sum(tf.image.total_variation(grad))
    return loss

def tv8(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    y_pred = tf.reshape(y_pred,[-1,tf.shape(y_pred)[-1]])
    y_pred = tf.clip_by_value(y_pred,clip_value_min=0.0001,clip_value_max=3000.0)
    recon  = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])*tes/1000.0)
    recon  = tf.reshape(recon,tf.shape(y_true))
    loss   = tf.reduce_sum(tf.image.total_variation(recon))
    return loss


def ncexp(y_true,y_pred):
    tes    = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])

    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    S0 = tf.reshape(y_pred[:,0],[-1,1])
    R2 = tf.reshape(y_pred[:,1],[-1,1])
    Sg = tf.reshape(y_pred[:,2],[-1,1])

    y_true = tf.reshape(y_true,[-1,12])

    s     = S0*tf.exp(-1.0*R2*tes/1000.0)
    alpha = tf.math.square(0.5*s/Sg)
    tempw = (1+2*alpha)*tf.math.bessel_i0e(alpha)+2*alpha*tf.math.bessel_i1e(alpha)
    recon = tf.math.sqrt(0.5*3.1415926*tf.math.square(Sg))*tempw

    loss  = tf.math.reduce_mean(tf.math.square(recon - y_true))
    return loss