
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,MaxPooling2D, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

from utils import *
from time import sleep

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

"""
class joinImages(_Merge):
    
    def _merge_function(self,inputs):
        return K.concatenate((inputs[0], inputs[1]), axis=1)
"""

class WGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        
        self.img_input_rows=int(self.img_rows/2)
        
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.img_input_shape=(self.img_input_rows, self.img_cols, self.channels)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        
        
        # Noise input
        z_disc = Input(shape=(self.latent_dim,),name="z_disc_input")
        imageInput=Input(shape=self.img_input_shape,name="imageInput_input")
        
        # Generate image based of noise (fake sample)
        #fake_img = self.generator(z_disc)
        
        #multiple inputs  
        fake_img=self.generator([imageInput,z_disc])
        
        #fake_img_completed=joinImages()([imageInput,fake_img])
        #print("fake_img",fake_img)
        
        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)
        
        print("")
        print("fake shape !!!!!!!!!!!!!!",fake.shape)
        print("valid shape",valid.shape)
        print("")
        
        print("real and fake",real_img, fake_img)
        print("imageInput",imageInput)
        fake_img_c=concatenate([imageInput, fake_img], axis=1)
        print("fake_img_c",fake_img_c)
        print("")
        #sys.exit()
        
        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img_c])
        
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)
        
        print("")
        print("validity_interpolated shape",validity_interpolated.shape)
        print("")
        
        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc,imageInput],
                            outputs=[valid, fake, validity_interpolated],name="critic_model")
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator([imageInput,z_gen])
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model([imageInput,z_gen], valid,name="generator_model")
        self.generator_model.summary()
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        #functional model (2 concatenated inputs)
        
        inputImageShape=self.img_input_shape
        
        inputNoise=Input(shape=(self.latent_dim,))
        inputImage=Input(shape=inputImageShape)
        
        mergeNeurons=2600
        
        #noise net
        n=Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim)(inputNoise)
        n=Reshape((7, 7, 128))(n)
        n=UpSampling2D()(n)
        n=Conv2D(128, kernel_size=4, padding="same")(n)
        n=BatchNormalization(momentum=0.8)(n)
        n=Activation("relu")(n)
        n=UpSampling2D()(n)
        n=Conv2D(64, kernel_size=4, padding="same")(n)
        n=BatchNormalization(momentum=0.8)(n)
        n=Activation("relu")(n)
        #
        n=Flatten()(n)
        n=Dense(mergeNeurons,activation='relu')(n)
        #n=Reshape((28, 28, 64))(n)
        n=Model(inputs=inputNoise,outputs=n)
        
        #n.summary()
        
        #image net
        i=Conv2D(256,(3,3),padding="same",strides=(4,4),input_shape=inputImageShape,activation='relu')(inputImage)
        i=MaxPooling2D(pool_size=(2, 2))(i)
        i=Dropout(0.2)(i)
        i=Conv2D(128,(3,3),padding="same",activation='relu')(i)
        i=MaxPooling2D(pool_size=(2, 2))(i)
        i=Dropout(0.2)(i)
        i=Conv2D(26,(3,3),padding="same",activation='relu')(i)
        #i=MaxPooling2D(pool_size=(2, 2))(i)
        #i=Conv2D(256,(3,3),padding="same",activation='relu')(i)
        i=Flatten()(i)
        i=Dense(mergeNeurons,activation='relu')(i)
        i=Model(inputs=inputImage,outputs=i)
        
        print("GENERATOR MODEL!!!")
        #i.summary()
        
        #merge them
        combined = concatenate([i.output, n.output])
        
        #more layers
        #z=Conv2D(self.channels, kernel_size=4, padding="same")(combined)
        
        z=Dense((self.img_input_shape[0]*self.img_input_shape[1]),activation='relu')(combined)
        print (z)
        #z=Flatten()(combined)
        #z=Conv2D(256,(3,3),padding="same",activation='relu')(combined)
        z=Reshape(inputImageShape)(z)
        
        z=Activation("tanh")(z)
        
        model=Model(inputs=[i.input,n.input],outputs=z,name="generator_model")
        model.summary()
        #return Model(noise, img)
        return model
        
    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        
        
        print("CRITIC MODEL!!!")
        model.summary()
        print("self.img_shape",self.img_shape)
        #sys.exit()
        img = Input(shape=self.img_shape)
        print("img input",img)
        validity = model(img)
        print("validity model",validity)
        return Model(img, validity,name="validity_model")

    def train(self, epochs, batch_size, sample_interval=50):
        print("")
        print("")
        print("TRAIN START")
        print("")
        print("")
        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        
        #print("X_train shape",X_train.shape)
        
        X_train=loadData()
        #print("X_train jamon shape",X_train.shape)
        
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        
        #fake=buildFakeImages(batch_size,[self.img_rows,self.img_cols])
        imgsToComplete=cropImageInputs(batch_size,[self.img_rows,self.img_cols])
        imgsToComplete=(imgsToComplete.astype(np.float32) - 127.5) / 127.5
        
        #print("imgsToComplete",imgsToComplete)
        
        #print("valid",valid.shape)
        #print("fake",fake.shape)
        
        #build fake images
        
        
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                """
                #show train images
                for img in imgs:
                    showImage(img[:,:,0]*255,"F")
                    sleep(2)
                """
                
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                imgs_input=imgsToComplete
                imgs_input=imgs_input[idx]
                
                #print(noise.shape)
                #print("holi")
                #sys.exit()
                # Train the critic
                
                print("")
                print("imgs",imgs.shape)
                print("noise",noise.shape)
                print("imgs_input",imgs_input.shape)
                print("dummy",dummy.shape)
                print("")
                
                d_loss = self.critic_model.train_on_batch([imgs,noise, imgs_input],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
   
    wgan.train(epochs=2001, batch_size=32, sample_interval=100)