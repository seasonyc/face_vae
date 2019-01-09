# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

import dataset
from utils import load_model, img_renorm, plot_image, plot_images
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
    
    pos_vectors /= pos_nums
    neg_vectors /= neg_nums
    
    
    attribute_vectors = {}
    pos_images = vae.get_layer('decoder').predict(pos_vectors)
    neg_images = vae.get_layer('decoder').predict(neg_vectors)
    for i in range(len(attrs)):
        attribute_vectors[attrs[i]] = pos_vectors[i] - neg_vectors[i]
        # draw the attribute for debugging
        print(attrs[i])
        plot_image([img_renorm(pos_images[i])], [img_renorm(neg_images[i])])
                
    np.save(attribute_vectors_file, attribute_vectors)


def read_image(path):
    file = tf.read_file(path)
    image = tf.image.decode_and_crop_jpeg(file, [29, 9, 160, 160])
    image.set_shape((160, 160, 3))
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    image = image.eval(session=tf.Session())
    return image


def trans_attribute(vae, image_file_name, attribute_vectors_file, attr_trans_dic):
    attribute_vectors = np.load(attribute_vectors_file).item()
    
    image = read_image(image_file_name)
    image = np.expand_dims(image, axis = 0)
    
    z = vae.get_layer('encoder').predict(image)[0]
    
    for attr, trans in attr_trans_dic.items():
        z[0] += (attribute_vectors[attr] * trans)
    
    modified_image = vae.get_layer('decoder').predict(z)
    plot_image(img_renorm(image), img_renorm(modified_image))
    


        
def merge_2(vae, image_file_name1, image_file_name2):
    
    image1 = read_image(image_file_name1)
    image1 = np.expand_dims(image1, axis = 0)
    
    z1 = vae.get_layer('encoder').predict(image1)[0]
    
    image2 = read_image(image_file_name2)
    image2 = np.expand_dims(image2, axis = 0)
    
    z2 = vae.get_layer('encoder').predict(image2)[0]
    z_v = []
    for i in range(9):
        z = (z2[0] * i + z1[0] * (8-i)) / 8
        z_v.append(z)
    
    z_v = np.asarray(z_v, dtype=np.float32)
    images = vae.get_layer('decoder').predict(z_v)
    plot_images(img_renorm(images))
    
    

'''    
def trans_attributes(vae, image_file_name, attribute_vectors_file, attrs):
    attribute_vectors = np.load(attribute_vectors_file).item()
    image = read_image('CelebA/img_align_celeba/' + image_file_name)
    image = np.expand_dims(image, axis = 0)
    
    z = vae.get_layer('encoder').predict(image)[0]
    z_v = [z[0] + attribute_vectors[attr] for attr in attrs]
    z_v2 = [z[0] - attribute_vectors[attr] for attr in attrs]
    z_v.extend(z_v2)
    z_v = np.asarray(z_v, dtype=np.float32)
    modified_images = vae.get_layer('decoder').predict(z_v)
    images = np.append(modified_images, image, axis = 0)
    plot_images(img_renorm(images))
    
    
def random(vae):
    z_shape = K.int_shape(vae.get_layer('encoder').outputs[0])
    z = K.random_normal(shape=(z_shape[1], z_shape[2], z_shape[3]))
    z = z.eval(session=tf.Session())
    z = np.expand_dims(z, axis = 0)
    
    modified_image = vae.get_layer('decoder').predict(z)
    plot_image(img_renorm(modified_image), img_renorm(modified_image))
'''    
        
attrs = ['Male', 'Smiling', 'Young', 'Attractive', 'Black_Hair', 'Blond_Hair']  

vae_dfc = load_model('face-vae12ch_good_train')

# Linear interpolation between 2 portraits
merge_2(vae_dfc, 'test_attr_trans_from_CelebA/201011.jpg', 'test_attr_trans_from_CelebA/202033.jpg')
merge_2(vae_dfc, 'test_attr_trans_from_CelebA/201220.jpg', 'test_attr_trans_from_CelebA/202595.jpg')


# compute attribute vector and translate portrait
compute_attribute_vector(vae_dfc, attrs, 'attribute_vectors_12ch_good.npy')

trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/202033.jpg', 'attribute_vectors_12ch_good.npy', {'Attractive':1,'Smiling':1})

trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/201207.jpg', 'attribute_vectors_12ch_good.npy', {'Male':1,'Blond_Hair':-1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/201278.jpg', 'attribute_vectors_12ch_good.npy', {'Male':1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/201349.jpg', 'attribute_vectors_12ch_good.npy', {'Male':1})

trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/201790.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1,'Young':1,'Attractive':1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/202016.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1,'Blond_Hair':1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/200235.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1,'Attractive':1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/202052.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/202163.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1,'Attractive':1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/202443.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1,'Young':1})
trans_attribute(vae_dfc, 'test_attr_trans_from_CelebA/202516.jpg', 'attribute_vectors_12ch_good.npy', {'Male':-1,'Blond_Hair':1})




