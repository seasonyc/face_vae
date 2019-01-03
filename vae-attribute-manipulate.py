# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

import dataset
from utils import save_model, load_model
from InstanceNormalization import InstanceNormalization



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def img_renorm(img):
    return (img + 1.0) / 2.0

def plot_image(input_images, rec_images):
    for x, r in zip(input_images, rec_images):
        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.subplot(1, 2, 2)
        plt.imshow(r)
        plt.show()
        


def compute_attribute_vector(vae, attrs, batch_size = 32):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        encoder = vae.get_layer('encoder')
        z_shape = K.int_shape(encoder.outputs[0])
        pos_vectors = np.zeros((len(attrs), z_shape[1], z_shape[2], z_shape[3]), np.float32)
        neg_vectors = np.zeros((len(attrs), z_shape[1], z_shape[2], z_shape[3]), np.float32)
        pos_nums = np.zeros((len(attrs), 1, 1, 1), np.int32)
        neg_nums = np.zeros((len(attrs), 1, 1, 1), np.int32)
        
        data, total_num = dataset.load_full_celeba_with_labels('CelebA', batch_size, attrs)
        
        iterator = data.make_one_shot_iterator()
        next_element = iterator.get_next()
        while True:
            try:
                images, labels = sess.run(next_element)
                z = encoder.predict(images)[0]
                
                for i in range(attrs):
                    pos_idx = np.argwhere(labels[:,i]==1)[:,0]
                    neg_idx = np.argwhere(labels[:,i]==-1)[:,0]
                    pos_vec = np.sum(z[pos_idx,:], 0)
                    neg_vec = np.sum(z[neg_idx,:], 0)
                    
                    pos_nums[i][0][0][0] += len(pos_idx)
                    neg_nums[i][0][0][0] += len(neg_idx)
                    pos_vectors[i] += pos_vec
                    neg_vectors[i] += neg_vec
            except tf.errors.OutOfRangeError:
                break
        
        pos_vectors /= pos_nums
        neg_vectors /= pos_nums
        attribute_vectors = pos_vectors - neg_vectors
        np.save("attribute_vectors.npy", attribute_vectors)

'''
suppose image 160 x 160
'''
def trans_attribute(vae, image, attr_trans_dic):
    with tf.Session() as sess:
        attribute_vectors = np.load("attribute_vectors.npy")
        
        image.set_shape((160, 160, 3))
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        
        z = vae.get_layer('encoder').predict(image)[0]
        
        #for attr, trans in attr_trans_dic.items():
        
        modified_image = vae.get_layer('decoder').predict(z)
        plot_image(img_renorm(image), img_renorm(modified_image))
        
        
        
attrs = ['Male', 'Attractive', 'Smiling', 'Receding_Hairline', 'Young']   


vae_dfc = load_model('face-vae-alpha0.5')
compute_attribute_vector(vae_dfc, attrs)


