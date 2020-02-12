
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
from keras.optimizers import *

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

from utils import *
from time import sleep

batchSize=32

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((batchSize, 1, 1, 1))
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

        self.loss="sparse_categorical_crossentropy"
        self.optm=Adam(lr=1e-3)

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
        
        inputImage=Input(shape=inputImageShape)
        inputNoise=Input(shape=(self.latent_dim,))
        
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
        n=Dense(mergeNeurons,activation='sigmoid')(n)
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
        i=Conv2D(64,(3,3),padding="same",activation='relu')(i)
        
        #i=MaxPooling2D(pool_size=(2, 2))(i)
        #i=Conv2D(256,(3,3),padding="same",activation='relu')(i)
        
        i=Conv2D(32,(3,3),padding="same",activation='relu')(i)
        
        i=Flatten()(i)
        i=Dense(mergeNeurons,activation='sigmoid')(i)
        i=Model(inputs=inputImage,outputs=i)
        
        print("GENERATOR MODEL!!!")
        #i.summary()
        
        #merge them
        combined = concatenate([i.output, n.output])
        
        #more layers
        z=Dense((32),activation='relu')(combined)
        #z=Dense((self.img_input_shape[0]*self.img_input_shape[1]),activation='relu')(combined)
        z=Dense((self.img_input_shape[0]*self.img_input_shape[1]),activation='relu')(z)
        #print (z)
        #z=Flatten()(combined)
        #z=Conv2D(256,(3,3),padding="same",activation='relu')(combined)
        z=Reshape(inputImageShape)(z)
        
        #z=Activation("tanh")(z)
        z=Activation("sigmoid")(z)
        
        model=Model(inputs=[i.input,n.input],outputs=z,name="generator_model")
        model.summary()
        #return Model(noise, img)
        return model
        
    def build_critic(self):

        model = Sequential()
        
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same",name="input_critic"))
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
        model.add(Dense(1,name="dense_boolean"))
        
        
        print("CRITIC MODEL!!!")
        model.summary()
        #print("self.img_shape",self.img_shape)
        #sys.exit()
        img = Input(shape=self.img_shape)
        #print("img input",img)
        validity = model(img)
        #print("validity model",validity)
        return Model(img, validity,name="validity_model")

    def train(self, epochs, batch_size, sample_interval=50):
        
        print("TRAIN START")
        
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)
        
        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        
        #self.generator.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        #self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        
        self.generator.compile(loss=self.loss, optimizer=self.optm)
        self.critic.compile(loss=self.loss, optimizer=self.optm)
        
        X_train,X_test,Y_train,Y_test,GT_train,GT_test=loadDataReady()
        print("X_train,X_test,Y_train,Y_test",X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
        
        # Adversarial ground truths
        #valid = -np.ones((batch_size, 1))
        #fake =  np.ones((batch_size, 1))
        
        for _ in range(epochs):
            
            #batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            X_train_b = X_train[idx]
            Y_train_b = Y_train[idx]
            GT_train_b = GT_train[idx]
            
            print("epoch ",_,"/",epochs)
            
            for i in range(X_train_b.shape[0]):
                
                #noise
                #noise = np.random.normal(0, 1, (1, self.latent_dim))
                noise=np.random.uniform(0,1,(1, self.latent_dim))
            
                #top X image
                topImage=X_train_b[i].reshape((1,)+X_train_b[i].shape)
                
                #generate
                generated=self.generator.predict([topImage,noise])
          
                #combine X image
                combined=np.vstack((topImage[0],generated[0]))
                
                #showImage(imtoShow,"L")
                #sleep(2)
                
                #real image
                realImage=GT_train_b[i].reshape((1,)+GT_train_b[i].shape)
                
                #discriminate image
                combinedP=combined.reshape((1,)+combined.shape)
                
                #fake=self.critic.predict(combinedP)
                #real=self.critic.predict(realImage)
                fake=np.array(0.0)
                real=np.array(.99)
                
                Xfit=np.array([combined,realImage[0]])
                #Yfit=np.array([fake[0],real[0]])
                Yfit=np.array([fake,real])
                
                #save training
                self.critic.fit(Xfit,Yfit,epochs=1,verbose=0)
                
                YfitG=Y_train_b[i].reshape((1,)+Y_train_b[i].shape)
                
                self.generator.fit([topImage,noise],YfitG,verbose=0)
            
            showImageNormal(combined)
            
            sleep(2)
        
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        #print ("noise shape",noise.shape)
        #print("self.X_test",self.X_test.shape)
        #print("self.X_test",self.X_test)
        np.random.shuffle(self.X_test)
        imgC=self.X_test[0:noise.shape[0]]
        #print("imgC shape", imgC.shape)
        gen_imgs = self.generator.predict([imgC,noise])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("graphs/jamon_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
   
    wgan.train(epochs=100, batch_size=batchSize, sample_interval=100)