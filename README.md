# A VAE for face translation
This VAE uses face-recognize network output as its perceptual Loss instead of the pixel reconstruction loss, the result examples are in pictures directory.


## Usage:
1. Clone this repo, the code is based on tensorflow 1.12.0 and tensorflow.keras 2.1.6-tf, InstanceNormalization.py is a workaround because keras 2.1.6-tf doesn't include keras-contrib InstanceNormalization.
2. Download CelebA dataset and put the files into CelebA folder.
3. Download facenet keras model and put into model folder.
4. Run face_vae.py to train the vae
5. Use vae_attribute_manipulate.py to compute attribute vector and translate the portrait.

