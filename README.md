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
