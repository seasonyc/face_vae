# -*- coding: utf-8 -*-


from tensorflow import keras

import time

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, Add, Conv2D, Lambda, UpSampling2D, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from InstanceNormalization import InstanceNormalization
import dataset
from utils import save_model, load_model, img_renorm, plot_image

# Shape of images
image_shape = (160, 160, 3)

batch_size = 16
lr_decay_ratio = 0.5
epochs=3


def conv_block(x, channels, norm_func, kernel_size = 4):
    x = Conv2D(channels, kernel_size, padding='same', use_bias=False)(x)
    print(K.int_shape(x))
    x = norm_func()(x)
    x = LeakyReLU(alpha=0.1)(x) #alpha=0.2
    return x

def downsampling_conv_block(x, channels, norm_func, kernel_size = 4):
    x = ZeroPadding2D()(x)
    x = Conv2D(channels, kernel_size, strides=(2, 2), use_bias=False)(x)
    print(K.int_shape(x))
    x = norm_func()(x)
    x = LeakyReLU(alpha=0.1)(x) #alpha=0.2
    return x

def upsampling_conv_block(x, channels, norm_func, kernel_size = 3):
    x = UpSampling2D()(x)
    x = Conv2D(channels, kernel_size, padding='same', use_bias=False)(x)
    print(K.int_shape(x))
    x = norm_func()(x)
    x = LeakyReLU(alpha=0.1)(x) #alpha=0.2
    return x

def transpose_conv_block(x, channels, norm_func, kernel_size = 3):
    x = Conv2DTranspose(channels, kernel_size, padding='same', strides=(2, 2), use_bias=False)(x)
    print(K.int_shape(x))
    x = norm_func()(x)
    x = LeakyReLU(alpha=0.1)(x) #alpha=0.2
    return x

def res_block(x, channels, norm_func, kernel_size = 3):
    input_x = x
    x = Conv2D(channels, kernel_size, padding='same', use_bias=False)(x)
    x = norm_func()(x)
    x = LeakyReLU(alpha=0.1)(x) #alpha=0.2
    x = Conv2D(channels, kernel_size, padding='same', use_bias=False)(x)
    x = norm_func()(x)
    x = Add()([input_x, x])
    return x


def create_encoder(latent_channel_num, norm_func):
    encoder_iput = Input(shape=image_shape, name='image')
    x = conv_block(encoder_iput, 64, norm_func=norm_func)
    x = downsampling_conv_block(x, 128, norm_func=norm_func)
    x = downsampling_conv_block(x, 256, norm_func=norm_func)
    for i in range(5):
        x = res_block(x, 256, norm_func=norm_func)
    
    z_mean = Conv2D(latent_channel_num, 1, name='z_mean')(x)
    z_log_var = Conv2D(latent_channel_num, 1, name='z_log_var')(x)

    last_conv_shape = K.int_shape(z_mean)
    print(last_conv_shape)

    return Model(encoder_iput, [z_mean, z_log_var], name='encoder'), last_conv_shape

def create_decoder(first_conv_shape, norm_func):
    decoder_input = Input(shape=(first_conv_shape[1], first_conv_shape[2], first_conv_shape[3]), name='latent_z')
    x = Conv2D(256, 1)(decoder_input)

    x = upsampling_conv_block(x, 128, norm_func=norm_func)
    x = upsampling_conv_block(x, 64, norm_func=norm_func)
    x = Conv2D(3, 4, padding='same', use_bias=False)(x)
    x = norm_func()(x) # may be easier to train if using norm here
    x = Activation('tanh', name='rec_image')(x) #tanh to ensure output is between -1, 1
    print(K.int_shape(x))
    return Model(decoder_input, x, name='decoder')

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    #z_mean, z_log_var = args
    z_mean = args[0]
    z_log_var = args[1]
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var * 0.5) * epsilon



def create_vae(latent_channel_num, norm_func_e, norm_func_d, return_kl_loss_op=False):
    '''
    Returns:
        The VAE model. If return_kl_loss_op is True, then the
        operation for computing the KL divergence loss is 
        additionally returned.
    '''

    
    encoder, shape = create_encoder(latent_channel_num, norm_func=norm_func_e)
    decoder = create_decoder(shape, norm_func=norm_func_d)
    
    input = Input(shape=image_shape, name='image')
    z_mean, z_log_var = encoder(input)
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    rec_img = decoder(z)

    model = Model(input, rec_img, name='vae')
    
    if return_kl_loss_op:
        kl_loss = -0.5 * K.mean(1 + z_log_var \
                                 - K.square(z_mean) \
                                 - K.exp(z_log_var), axis=(1,2,3))
        return model, kl_loss
    else:
        return model

    

x_train, train_size = dataset.load_celeba('CelebA', batch_size, part='train')
x_val, val_size = dataset.load_celeba('CelebA', batch_size, part='val')


def train(selected_pm_layers, alpha = 1.0, latent_channel_num = 8, learning_rate = 0.0005,
          norm_func_e = InstanceNormalization, norm_func_d = InstanceNormalization, trained_model = None):
    from tensorflow.keras.models import model_from_json
     
    #facenet model structure: https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
    pm = model_from_json(open("model/facenet_model.json", "r").read())
     
    #pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
    pm.load_weights('model/facenet_weights.h5')
     
    #pm.summary()
    
    def perceptual_loss(input_img, rec_img):
        '''Perceptual loss for the DFC VAE'''
        outputs = [pm.get_layer(l).output for l in selected_pm_layers]
        
        model = Model(pm.input, outputs)
    
        h1_list = model(input_img)
        h2_list = model(rec_img)
        if not isinstance(h1_list, list):
            h1_list = [h1_list]
            h2_list = [h2_list]
                
        p_loss = 0.0
        
        for h1, h2 in zip(h1_list, h2_list):
            h1 = K.batch_flatten(h1)
            h2 = K.batch_flatten(h2)
            p_loss = p_loss + K.mean(K.square(h1 - h2), axis=-1)
        
        return p_loss
    
    
    # Create DFC VAE model and associated KL divergence loss operation
    vae_dfc, kl_loss = create_vae(latent_channel_num, return_kl_loss_op=True, norm_func_e=norm_func_e, norm_func_d=norm_func_d)

    if trained_model:
        vae_dfc.set_weights(trained_model.get_weights()) 
    
    def vae_dfc_loss(input_img, rec_img):
        '''Total loss for the DFC VAE'''
        return K.mean(alpha * perceptual_loss(input_img, rec_img) + kl_loss)
    
    
    opt = keras.optimizers.Adam(lr=learning_rate, epsilon=1e-08)
    vae_dfc.compile(optimizer=opt, loss=vae_dfc_loss)
    
    def schedule(epoch, lr):
        if epoch > 0:
            lr *= lr_decay_ratio
        return lr
        
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
        
    vae_dfc.fit(x_train, epochs=epochs, steps_per_epoch=train_size//batch_size,
                validation_data=(x_val), validation_steps=val_size//batch_size, callbacks=[lr_scheduler], verbose=1)

    return vae_dfc




def test_vae(vae):
    for part in ('train', 'val', 'test'):
        input_images = dataset.fetch_smallbatch_from_celeba('CelebA', part=part)
        rec_images = vae.get_layer('decoder').predict(vae.get_layer('encoder').predict(input_images)[0])
        plot_image(img_renorm(input_images), img_renorm(rec_images))



'''

selected_pm_layers = ['Conv2d_1a_3x3','Conv2d_3b_1x1', 'Conv2d_4b_3x3', 'add_5', 'add_15', 'add_21', 'Bottleneck']
vae_dfc = train(selected_pm_layers, latent_dim = 100, norm_func = BatchNormalization, deconv_func = upsamp_conv)
save_model(vae_dfc, 'face-vae' + str(time.time()))
test_vae(vae_dfc)

'''



# selected for calculating the perceptual loss.
selected_pm_layers = ['Conv2d_1a_3x3', 'Conv2d_2b_3x3', 'Conv2d_4a_3x3', 'Conv2d_4b_3x3', 'Bottleneck']

vae_dfc = train(selected_pm_layers, alpha = 1, latent_channel_num = 12)
save_model(vae_dfc, 'face-vae' + str(time.time()))
test_vae(vae_dfc)

vae_dfc = train(selected_pm_layers, alpha = 1, latent_channel_num = 24)
save_model(vae_dfc, 'face-vae' + str(time.time()))
test_vae(vae_dfc)

'''

vae_dfc = train(selected_pm_layers)
save_model(vae_dfc, 'face-vae' + str(time.time()))
test_vae(vae_dfc)



selected_pm_layers = ['Bottleneck']

vae_dfc = train(selected_pm_layers)
save_model(vae_dfc, 'face-vae' + str(time.time()))
test_vae(vae_dfc)

'''
