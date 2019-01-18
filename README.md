# Spatial Z VAE for face translation
This is a VAE, which can generate face with high reconstruction quality by:
- using face-recognize network output as its perceptual Loss instead of the pixel reconstruction loss, and 
- using spatial latent vector to keep more infromation.

## Some results:
Attractive and Smiling:

![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/image_pair1547048621.390507.jpg)

Smiling:

![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/image_pair1547048628.0408874.jpg)

Gender exchange with some attribute trans or not:

![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/image_pair1547048625.1077194.jpg)![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/image_pair1547048623.6556363.jpg)
![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/image_pair1547048622.0885468.jpg)![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/image_pair1547048628.7209263.jpg)

Linear interpolation between 2 faces:
![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/images1547095219.6064727.jpg)
![](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/pictures/images1547095340.6845298.jpg)

More result examples are in https://github.com/seasonyc/Spatial-Z-VAE/tree/master/pictures


## Usage:
1. Clone this repo, the code is based on tensorflow 1.12.0 and tensorflow.keras 2.1.6-tf, InstanceNormalization.py is a workaround because keras 2.1.6-tf doesn't include keras-contrib InstanceNormalization.
2. Download CelebA dataset and put the files into CelebA folder.
3. Download facenet keras model and put into model folder.
4. Run spatial_z_vae.py to train the vae
5. Use vae_attribute_manipulate.py to compute attribute vector and translate the portrait. You can use [my trained model](https://github.com/seasonyc/Spatial-Z-VAE/blob/master/trained_face_vae/face_vae_12ch.zip) or train your own.

See [the wiki](https://github.com/seasonyc/Spatial-Z-VAE/wiki) for more technical details.





Why does VAE generate blurry image?(and why GAN generates sharp?) 

This topic is already discussed even argued hotly. 

Someone focus on the KL loss, someone tried to use a better loss instead of L2 distance. Yes, if we decrease the weight of KL loss, we can get a little better quality. And some experiments like [DFC-VAE](https://arxiv.org/abs/1610.00291v1) have gotten available achievement on the aspect of better loss. However by those works, the generative face is still blurry and common! Currently, the best result of VAE looks to be [IntroVAE](https://arxiv.org/abs/1807.06358), but its reconstructed face is not very like the original one!

I think previous works didn't catch one point -- that is they usually used a small latent dimension: 100, 256, up to 512. Such latent vector might be not enough to represent the original information or not enough for encoder/decoder learning. By contrast, we usually train an overcapacity DNN to fit a function. 

So the encoder is unable to pass enough information through the bottleneck(latent vector) to the decoder, meanwhile gradient descent forces to minimize L2 distance loss(or any other loss), VAE network can only output a mean value~~ that means a blurry and a common image/face.


Then, if we allow enough information to pass through the network, VAE can have a good reconstruction quality, that is what this project did. I adopt a shallow conv network for encoder, a 1x1 conv is appended to that to generate latent vector, here's no flatten and dense to avoid too much computation, and then a shallow upsampling/conv network is connected as the decoder. The channel num of 1x1 conv can be changed to control the information quantity, more latent channels, better reconstruction quality. 


GAN trains a network to learn a mapping from a latent distribution to image, but actually it trains a network to remember the distribution of the image information... so it can generate sharp image details but actually not from the latent vector(more or less from....) 

Either VAE or GAN can generate/translate image with high reconstruction quality if it has enough information of the original image, e.g. StarGAN, attGAN.

A special case is [CVAE-GAN](https://arxiv.org/abs/1703.10155), it can synthesize high quality images in fine-grained categories, such as faces of a specific person or objects in a category. If you read the paper carefully, you can find it actually remembers the features of the specific identity by its G mostly and its E a bit. The paper mentioned inpainting experiment, which just shows the network synthesizes the image without depending on the input! Legacy VAE has similar effect: mask some part of a face, VAE will output full face without any mask. Anyway, CVAE-GAN is a nice work:)



This project is not inspired by:
* [DFC-VAE](https://arxiv.org/abs/1610.00291v1)
* [facenet VAE](https://github.com/davidsandberg/facenet/wiki/Variational-autoencoder)
* [StarGAN](https://arxiv.org/abs/1711.09020)

although they are prior to this project.

However, the implementation of the network, training and perceptual loss is inspired by [DFC-VAE](https://arxiv.org/abs/1610.00291v1), the implementation of the perceptual loss is inspired by [facenet VAE](https://github.com/davidsandberg/facenet/wiki/Variational-autoencoder), the implementation of the network is inspired by [StarGAN](https://arxiv.org/abs/1711.09020).

The code structure of spatial_z_vae.py is inspired by Martin [Krasser's dfc-vae code](https://github.com/krasserm/bayesian-machine-learning/blob/master/variational_autoencoder_dfc.ipynb).

In this project, Spatial Z is not the purpose but the means in order to provide enough information in latent vector, anyway spatial latent vector may help to get better reconstruction quality. Spatial Z is not inspired by:
* [Spatial Variational Auto-Encoding via Matrix-Variate Normal Distributions](https://openreview.net/forum?id=ryOBB6g-M)
* [Feature Map Variational Auto-Encoders](https://openreview.net/forum?id=Hy_o3x-0b)
* The comments of above reviews also mentioned [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934)

The implementation is not inspired by them too, but I noticed them existed before I finished this project.

I am not sure if VAE is a hopeful idea than GAN, but the idea of latent vector manipulation is interesting, so I tried this project.

