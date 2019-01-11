# -*- coding: utf-8 -*-
from __future__ import print_function 
import numpy as np  
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def img_renorm(img):
    return (img + 1.0) / 2.0

def plot_image(input_images, rec_images, save_image=True):
    for x, r in zip(input_images, rec_images):
        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.subplot(1, 2, 2)
        plt.imshow(r)
        if save_image:
            plt.savefig('image_pair'+ str(time.time()) + '.jpg')
        plt.show()
        
def save_model(model, name):
    json_string = model.to_json()
    file = open(name + '.json', 'w') 
    file.write(json_string) 
    file.close() 
    model.save_weights(name + '.h5')

def load_model(name):
    from tensorflow.keras.models import model_from_json
    model = model_from_json(open(name + '.json', 'r').read())
    model.load_weights(name + '.h5')
    return model
    
#generate random index
def generate_rand_index():
    index=np.arange(10000)  
    np.random.shuffle(index)  
    print(index[0:20])
    
    np.save("validation_index.npy",index[0:5000])
    np.save("test_index.npy",index[5000:10000])
    
def load_index():
    index_v = np.load("validation_index.npy")
    index_t = np.load("test_index.npy")
    print(index_v[0:20])
    print(index_t[0:20])


def plot_images(images, save_image=True):
    num = len(images)
    fig = plt.figure(figsize = (num*2.5,1*2.5))
    i = 1
    for image in images:
        plt.subplot(1, num, i)
        plt.imshow(image, aspect='auto')
        plt.axis('off')
        i += 1
    if save_image:
        plt.savefig('images'+ str(time.time()) + '.jpg')
    plt.show()

