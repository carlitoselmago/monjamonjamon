import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import pandas as pd
import os

import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adadelta,RMSprop
from keras.layers.embeddings import Embedding
from keras.layers import merge
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers,MaxPooling2D,Reshape
from keras.initializers import Constant
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import Conv2D

from utils import *



#input data
imageInputSize=[400,10]#[800,20] #X,Y .original is 800x800 if lower, image will be resized
outputSize=[imageInputSize[0],1]

#X,Y=prepareData(imageInputSize[1],outputSize[1])
X,Y=prepareData(imageInputSize,outputSize[1])

#print (X,Y)

features=3
numneurons=50
loss='mse'#'mean_squared_error' # Since the metric is MSE/RMSE, no change
optimizer = 'adam'#rmsprop'     # Recommended optimizer for RNN, no change
activation = 'linear' 
batch_size = 1
nb_epoch =  5#800

inputShape=(imageInputSize[1],imageInputSize[0],3)

#model part
model = Sequential()
model.add(Conv2D(255, (8, 8), strides=(4, 4), input_shape=inputShape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(4,4), padding='valid',dim_ordering="th"))
model.add(Flatten())

#model.add(LSTM(255,input_shape=inputShape, return_sequences=False))
#model.add(Dense(255,input_shape=(1,1,3),name='pred'))

model.add(Dense(np.prod((1,imageInputSize[0],3)), activation='tanh'))
model.add(Reshape((1,imageInputSize[0],3)))

model.compile(loss=loss, optimizer=optimizer)
history=model.fit(X,Y,batch_size=batch_size,epochs=nb_epoch,validation_split=0.1)   


# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

model.save('trained/jamonjamonjamon.h5')    

#predicted=predictBatch(X_test,model)
#graph(Y_test,predicted)
