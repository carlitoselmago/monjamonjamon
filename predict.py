#TEST MODEL
from PIL import Image
from utils import *
from keras.models import load_model
import matplotlib.pyplot as plt
import json
from keras.models import model_from_json
from skimage import img_as_ubyte

#SETTINGS:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
modelname="model8"
imageInputSize=[80,40]
predictionH=40
"""
modelname="generator"
imageInputSize=[30,30]
predictionH=10

predictionIter=30
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


#model = load_model('trained/'+modelname+'.h5')
print ('trained/'+modelname+'.json')
#model = model_from_json('trained/'+modelname+'.json')
with open('trained/'+modelname+'.json', 'r') as json_file:
    architecture = json.load(json_file)
    model = model_from_json(json.dumps(architecture))

# load weights into new model
model.load_weights('trained/'+modelname+'_weights.hdf5')

#get first image and normalize the one for prediction
X,startImage=getRandomJamon(imageInputSize)
X=normalizeImage(X)

masked_X=maskimageMiddle(X[0])
masked_X=masked_X.reshape((1,)+masked_X.shape)

#startimage is a PIL version of the image
canvas=startImage
totalH=imageInputSize[1]+predictionH

for i in range(predictionIter):

    predicted=model.predict(masked_X)

    startimage=0.5 * X[0] + 0.5
    predicted= 0.5 * predicted[0] + 0.5

    canvas=concatenateWithMiddle(startimage,predicted)
    #imgplot=plt.imshow(canvas)
    #plt.show()

    denormalizeImage(canvas).show()



    ########################### aqui lo dejé #####################################
    im=np.array(canvas)

    xslice=im[totalH-imageInputSize[1]:totalH,:]

    #xslice=xslice.reshape(xslice.shape+(1,))



    #xslice=xslice.reshape((1,) + im.shape)
    totalH+=predictionH
    X=np.expand_dims(xslice, axis=0)

    masked_X=maskimageBottom(X[0])
    masked_X.resize((masked_X.shape[0]*2, masked_X.shape[1],masked_X.shape[2]))
    #print("masked_XR",masked_X)
    #print("masked_X shape",masked_X.shape)
    masked_X=masked_X.reshape((1,)+masked_X.shape)
    print("masked_X shape",masked_X.shape)
    showImage(masked_X[0])
    sleep(2)

    #sys.exit()
    #print("X shape",X.shape)
    #sys.exit()

    #canvas.show()
    #sleep(10)

canvas.show()
