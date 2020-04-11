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
imageInputSize=[60,60]
predictionH=20

predictionIter=50
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



#startimage is a PIL version of the image
canvas=startImage
totalH=imageInputSize[1]+predictionH

solvedImages=[]

for i in range(predictionIter):
    predicted=model.predict(masked_X.reshape((1,)+masked_X.shape))

    #not sure if this is necesary:
    #startimage=0.5 * X[0] + 0.5
    #predicted= 0.5 * predicted[0] + 0.5

    predicted=predicted[0]

    canvas=concatenateWithMiddle(masked_X,predicted)
    #imgplot=plt.imshow(canvas)
    #plt.show()

    #show image
    denormalizeImage(canvas).show()
    

    #define next masked_X

    #grab next random jamon
    X,startImage=getRandomJamon(imageInputSize)
    X=normalizeImage(X)

    #construct next batch
    masked_X=np.empty_like(X)[0]



    #top block
    masked_X[0:predictionH,0:imageInputSize[0],:]=canvas[canvas.shape[1]-predictionH:canvas.shape[1],0:imageInputSize[0],:]
    #middle block
    masked_X[predictionH:int(predictionH*2),0:imageInputSize[0],:]=0
    #bottom block
    masked_X[int(predictionH*2):imageInputSize[1],0:imageInputSize[0],:]=X[0][0:predictionH,0:imageInputSize[0],:]

    #add extra dimension
    #masked_X=masked_X.reshape(masked_X.shape+(1,))

    piledimage=denormalizeImage(canvas[predictionH:predictionH*2,0:imageInputSize[0],:])
    #piledimage.show()
    #sleep(5)
    solvedImages.append(piledimage)

    ########################### aqui lo dejé #####################################


finalImage=np.vstack(solvedImages )
print(finalImage.size)
finalImage = Image.fromarray(finalImage)
finalImage.show()
