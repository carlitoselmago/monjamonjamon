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


def createModel(inputShape):

    model=Sequential()

    model.add(Conv2D(36, (4, 4), input_shape=inputShape,name="conv1"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="max2"))

    model.add(Conv2D(256, (4, 4),name="conv3",padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="max4"))

    model.add(Conv2D(65, (8, 8),name="conv5",padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name="max6"))

    model.add(Flatten())

    model.add(Dense(np.prod((1,imageInputSize[0],channels)), activation='sigmoid'))
    model.add(Reshape((1,imageInputSize[0],channels)))

    model.add(Activation('sigmoid'))

    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    
    return model