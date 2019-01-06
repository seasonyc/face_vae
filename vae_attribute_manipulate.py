# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

import dataset
from utils import load_model, img_renorm, plot_image
from InstanceNormalization import InstanceNormalization



def compute_attribute_vector(vae, attrs, attribute_vectors_file, batch_size = 32):
    sess = K.get_session()
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
            
            for i in range(len(attrs)):
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
    
    print('vectors\n\n\n\n\n\n')
    print(pos_vectors)
    print(neg_vectors)
    pos_vectors /= pos_nums
    neg_vectors /= neg_nums
    
    
    attribute_vectors = {}
    pos_images = vae.get_layer('decoder').predict(pos_vectors)
    neg_images = vae.get_layer('decoder').predict(neg_vectors)
    for i in range(len(attrs)):
        attribute_vectors[attrs[i]] = pos_vectors[i] - neg_vectors[i]
        print(attrs[i])
        plot_image([img_renorm(pos_images[i])], [img_renorm(neg_images[i])])
                
    np.save(attribute_vectors_file, attribute_vectors)


'''
suppose image 160 x 160
'''
def trans_attribute(vae, image_file_name, attribute_vectors_file, attr_trans_dic):
    attribute_vectors = np.load(attribute_vectors_file).item()
    
    file = tf.read_file(image_file_name)
    image = tf.image.decode_and_crop_jpeg(file, [29, 9, 160, 160])
    image.set_shape((160, 160, 3))
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    image = image.eval(session=tf.Session())
    image = np.expand_dims(image, axis = 0)
    
    z = vae.get_layer('encoder').predict(image)[0]
    for attr, trans in attr_trans_dic.items():
        z[0] += (attribute_vectors[attr] * trans)
    
    modified_image = vae.get_layer('decoder').predict(z)
    plot_image(img_renorm(image), img_renorm(modified_image))
    
        
        
attrs = ['Male', 'Bald', 'Smiling', 'Chubby', 'Heavy_Makeup', 'Straight_Hair', 'Wavy_Hair']  


vae_dfc = load_model('face-vae-alpha0.5')
compute_attribute_vector(vae_dfc, attrs, 'attribute_vectors_5a.npy')

vae_dfc = load_model('face-vae8channels1alpha')
compute_attribute_vector(vae_dfc, attrs, 'attribute_vectors_1a.npy')


vae_dfc = load_model('face-vae-alpha0.5')
print('alpha 0.5  ............................\n\n\n\n\n')

trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_5a.npy', {'Male': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_5a.npy', {'Bald': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_5a.npy', {'Heavy_Makeup': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_5a.npy', {'Chubby': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_5a.npy', {'Wavy_Hair': 1})

trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_5a.npy', {'Male': -1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_5a.npy', {'Bald': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_5a.npy', {'Straight_Hair': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_5a.npy', {'Chubby': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_5a.npy', {'Wavy_Hair': 1})



vae_dfc = load_model('face-vae8channels1alpha')
print('alpha 1  ............................\n\n\n\n\n')

trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_1a.npy', {'Male': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_1a.npy', {'Bald': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_1a.npy', {'Heavy_Makeup': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_1a.npy', {'Chubby': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/201207.jpg', 'attribute_vectors_1a.npy', {'Wavy_Hair': 1})

trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_1a.npy', {'Male': -1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_1a.npy', {'Bald': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_1a.npy', {'Straight_Hair': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_1a.npy', {'Chubby': 1})
trans_attribute(vae_dfc, 'CelebA/img_align_celeba/200391.jpg', 'attribute_vectors_1a.npy', {'Wavy_Hair': 1})


