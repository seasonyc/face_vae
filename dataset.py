# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


def load_celeba(data_dir, batch_size, prefetch_batch=1, num_threads=4, buffer_size=4096, part='train'):
    
    list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
    img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
        

    names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
    img_paths = [os.path.join(img_dir_jpg, name) for name in names]

    
    if part == 'test':
        img_paths = img_paths[182637:]
    elif part == 'val':
        img_paths = img_paths[162770:182637]
    else:
        img_paths = img_paths[:162770] 

   
    img_num = len(img_paths)
    
    # load and cache the raw image files
    def load_func(path):
        file = tf.read_file(path)
        return file
    
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    # celebA files are shuffled already, don't shuffle here, reading the files consecutively may have better performance(because files may not be placed consecutively in the disk)
    # cache the files in memory to read disk only once, otherwise use TFRecord to speed up disk reading
    dataset = dataset.map(load_func, num_parallel_calls=num_threads)
    dataset = dataset.cache()
    
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size))
    # decode jpg and make the dataset from the cache
    def decode_and_preprocess_func(file):
        image = tf.image.decode_and_crop_jpeg(file, [29, 9, 160, 160])
        #image = tf.image.resize_images(image, [128, 128], tf.image.ResizeMethod.BICUBIC)
        image.set_shape((160, 160, 3))
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        
        return image, image
    
    dataset = dataset.map(decode_and_preprocess_func, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size).prefetch(prefetch_batch)

    return dataset, img_num
    

 

def fetch_smallbatch_from_celeba(data_dir, count=10, num_threads=4, part='train'):
    
    list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
    img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
        

    names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
    img_paths = [os.path.join(img_dir_jpg, name) for name in names]

    
    if part == 'test':
       img_paths = img_paths[182637:]
    elif part == 'val':
        img_paths = img_paths[162770:182637]#
    else:
        img_paths = img_paths[:162770]

   
    img_num = len(img_paths)
    
    # load and cache the raw image files
    def load_and_decode_func(path):
        file = tf.read_file(path)
        image = tf.image.decode_and_crop_jpeg(file, [29, 9, 160, 160])
        #image = tf.image.resize_images(image, [128, 128], tf.image.ResizeMethod.BICUBIC)
        image.set_shape((160, 160, 3))
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        return image 

    
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    
    dataset = dataset.shuffle(img_num)
    
    dataset = dataset.map(load_and_decode_func, num_parallel_calls=num_threads)
    dataset = dataset.batch(count).take(1)

    element = dataset.make_one_shot_iterator().get_next()
    images = None
    with tf.Session() as sess:    
        images = sess.run(element)
    return images
    



att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
            'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
            'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
            'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
            'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
            'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
            'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
            'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
            'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
            'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
            'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

atts = ['Male', 'Attractive', 'Smiling']

def load_full_celeba_with_labels(data_dir, batch_size, atts, prefetch_batch=1, num_threads=4):
    
    list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
    img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
        

    names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
    img_paths = [os.path.join(img_dir_jpg, name) for name in names]

    att_id = [att_dict[att] + 1 for att in atts]
    labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int8)

    # because norm layers are trained by training set, this function actually loads full training data 
    img_paths = img_paths[:162770] 
    labels = labels[:162770]
 
    img_num = len(img_paths)
    
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def process_func(path, label):
        file = tf.read_file(path)
        image = tf.image.decode_and_crop_jpeg(file, [29, 9, 160, 160])
        #image = tf.image.resize_images(image, [128, 128], tf.image.ResizeMethod.BICUBIC)
        image.set_shape((160, 160, 3))
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        
        return image, label
    
    dataset = dataset.map(process_func, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size).prefetch(prefetch_batch)

    return dataset, img_num
   

# test:
    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_load_celeba():
    with tf.Session() as sess:
        data, total_num = load_celeba('CelebA', 32, part='val')
        print(total_num)
        print(data.output_types)  
        print(data.output_shapes)
        iterator = data.make_one_shot_iterator()
        next_element = iterator.get_next()
        for i in range(10):
            img = sess.run(next_element)
            #print(img.shape)
            plt.imshow(img[0][0], interpolation='spline16')
            plt.show()
            plt.imshow(img[1][0], interpolation='spline16')
            plt.show()
        
def test_fetch():
    data = fetch_smallbatch_from_celeba('CelebA', part='val')
    for i in range(10):
        plt.imshow(data[i], interpolation='spline16')
        plt.show()
            
        
def test_load_celeba_with_labels():
    with tf.Session() as sess:
        data, total_num = load_full_celeba_with_labels('CelebA', 32, atts)
        print(total_num)
        print(data.output_types)  
        print(data.output_shapes)
        iterator = data.make_one_shot_iterator()
        next_element = iterator.get_next()
        for i in range(10):
            img, labels = sess.run(next_element)
            #print(img.shape)
            plt.imshow(img[0], interpolation='spline16')
            plt.show()
            print(labels[0])

# test_load_celeba_with_labels()

'''        
def plot_image(input_images, rec_images):
    for x, r in zip(input_images, rec_images):
        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.subplot(1, 2, 2)
        plt.imshow(r)
        plt.axis('off')
        plt.show()
       
def img_renorm(img):
    return (img + 1.0) / 2.0        
        
plot_image(img_renorm(fetch_smallbatch_from_celeba('CelebA', part='val')), img_renorm(fetch_smallbatch_from_celeba('CelebA'))) 
'''        
        
        
