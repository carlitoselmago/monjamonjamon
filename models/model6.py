import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adadelta,RMSprop
from keras.layers.embeddings import Embedding
from keras.layers import merge
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout,MaxPooling2D,Reshape,Conv2DTranspose
from keras.initializers import Constant
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D

modelname="model6"

def createModel(inputShape,loss,optimizer):
    
    model=Sequential()
    #model.add(Conv2D(255, (2, 2), input_shape=inputShape))
    model.add(Conv2D(256,(3,3),padding="same",strides=(4,4),input_shape=inputShape,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128,(3,3),padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,(3,3),padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(256,(3,3),padding="same",activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dense(360,activation='relu'))
    
    model.add(Dense((inputShape[1]),activation='relu'))
    
    #model.add(Reshape((inputShape[1],1)))
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    
    return model