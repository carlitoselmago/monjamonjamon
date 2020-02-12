import numpy as np
import matplotlib.pylab as plt
import glob
import sys
from PIL import Image
from PIL import Image, ImageOps 
import os, random
from time import sleep

def getRandomJamon(InputSize):
    InputHeight=InputSize[1]
    uri=random.choice(os.listdir("images"))
    image = Image.open("images/"+uri)
    
    oImage=image
    
    image=image.resize((InputSize[0],InputSize[0]))
    im=np.array(image)
    #im=normalize0to1(im)
    im=im.reshape((1,)+im.shape)
    return im,oImage

def getRandomStartSlice(InputSize):
    
    InputHeight=InputSize[1]
    uri=random.choice(os.listdir("images"))
    image = Image.open("images/"+uri)
        
    #convert to grayscale
    image=image.convert('L')
    image=image.resize((InputSize[0],InputSize[0]))
    
    #convert to numpy
    im=np.array(image)
    
    i=0
    xslice=im[i:i+InputHeight,:]
    xslice=xslice.reshape(xslice.shape+(1,))
    #xslice=xslice.reshape((1,) + im.shape)
    
    xslice=np.expand_dims(xslice, axis=0)
    
    return xslice,image

def concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def normalize0to1(data):
    info = np.iinfo(data.dtype)
    data=data.astype(np.float64) / info.max # normalize the data to 0 - 1
    return data

def normalizeMinus1to1(data):
    return (data.astype(np.float32) - 127.5) / 127.5

def loadDataReady(amount=5,InputSize=[28,28]):
    cut=int(InputSize[0]/2)
    
    cutsUp=[]
    cutsDown=[]
    
    oImgs=loadDataRaw(InputSize)
    for oI in oImgs:
        imgNP=oI[0:cut,:]
        cutsUp.append(imgNP)
        
        imgDNP=oI[cut:,:]
        cutsDown.append(imgDNP)

    cutsUp=np.array(cutsUp)
    cutsDown=np.array(cutsDown)
    #cuts=cuts.reshape(cuts.shape+(1,))
    
    # Rescale -1 to 1
    cutsUp = normalize0to1(cutsUp)#(cuts.astype(np.float32) - 127.5) / 127.5
    cutsUp = np.expand_dims(cutsUp, axis=3)
    
    cutsDown = normalize0to1(cutsDown)#(cuts.astype(np.float32) - 127.5) / 127.5
    cutsDown = np.expand_dims(cutsDown, axis=3)
    
    oImgs = normalize0to1(oImgs)#(oImgs.astype(np.float32) - 127.5) / 127.5
    oImgs = np.expand_dims(oImgs, axis=3)
    
    #split some for test
    splitP=int(len(oImgs)*0.8)
   
    X_train=cutsUp[:splitP]
    X_test=cutsUp[splitP:]
    
    Y_train=cutsDown[:splitP]
    Y_test=cutsDown[splitP:]
    
    #ground truths
    GT_train=oImgs[:splitP]
    GT_test=oImgs[splitP:]

    return X_train,X_test,Y_train,Y_test,GT_train,GT_test

def loadDataReady2(amount=5,InputSize=[28,28],mode="GREY"):
    cut=int(InputSize[0]/2)
    
    cutsUp=[]
    cutsDown=[]
    
    oImgs=loadDataRaw(InputSize,mode)
    for oI in oImgs:
        imgNP=oI[0:cut,:]
        cutsUp.append(imgNP)
        
        imgDNP=oI[cut:,:]
        cutsDown.append(imgDNP)

    cutsUp=np.array(cutsUp)
    cutsDown=np.array(cutsDown)
    #cuts=cuts.reshape(cuts.shape+(1,))
    
    # Rescale -1 to 1
    cutsUp = normalizeMinus1to1(cutsUp)#(cuts.astype(np.float32) - 127.5) / 127.5
    if mode=="GREY":
        cutsUp = np.expand_dims(cutsUp, axis=3)
    
    cutsDown = normalizeMinus1to1(cutsDown)#(cuts.astype(np.float32) - 127.5) / 127.5
    if mode=="GREY":
        cutsDown = np.expand_dims(cutsDown, axis=3)
    
    oImgs = normalizeMinus1to1(oImgs)#(oImgs.astype(np.float32) - 127.5) / 127.5
    if mode=="GREY":
        oImgs = np.expand_dims(oImgs, axis=3)
    
    #split some for test
    splitP=int(len(oImgs)*0.8)
   
    X_train=cutsUp[:splitP]
    X_test=cutsUp[splitP:]
    
    Y_train=cutsDown[:splitP]
    Y_test=cutsDown[splitP:]
    
    #ground truths
    GT_train=oImgs[:splitP]
    GT_test=oImgs[splitP:]

    return X_train,X_test,Y_train,Y_test,GT_train,GT_test

def maskimageBottom(img):
    
    y1 =int(img.shape[0]/2)
    y2 =img.shape[0]
    x1 =0
    x2 =img.shape[1]
    
    masked_imgs = np.empty_like(img)
    missing_parts = np.empty(( y1, img.shape[1],img.shape[2]))
    
    masked_img = img.copy()
    missing_parts = masked_img[y1:y2, x1:x2, :].copy()
    masked_img[y1:y2, x1:x2, :] = 0
    return masked_img

def addEmptyMaskBottom(img):
    return img.resize((img.shape[0]*2, img.shape[1]))

def cropImageInputs(amount=5,InputSize=[28,28]):
    cut=int(InputSize[0]/2)
    
    cuts=[]
    
    oImgs,_=loadData(InputSize)
    for oI in oImgs:
        imgNP=oI[0:cut,:]
        #print("imgNP shape",imgNP.shape)
        cuts.append(imgNP)

    cuts=np.array(cuts)
    
    cuts=cuts.reshape(cuts.shape+(1,))
    
    return cuts
        
def buildFakeImages(amount=5,InputSize=[28,28]):
    cut=int(InputSize[0]/2)
    oImgs=loadData(InputSize)
    #get some random original X images
    idx = np.random.randint(0,oImgs.shape[0], amount)
    oImgs = oImgs[idx]
    #print("oImgs shape",oImgs.shape)
    
    faked=[]
    
    for oI in oImgs:
        #print("oI shape",oI.shape)
        #print("cut",cut)
        #imgNP=oI[0,cut:0,oI.shape[0]]
        imgNP=oI[0:cut,:]
        #print("imgNP shape",imgNP.shape)
        Oimg = Image.fromarray(imgNP)
        #noise
        #noise=np.random.random((cut,InputSize[0])) 
        noise=noise_generator((cut,InputSize[1]))
        
        #print("noise.shape",noise.shape)
        
        Nimg=Image.fromarray(noise,"L")
        canvas=concat_v(Oimg,Nimg)
        canvas=canvas.convert('L')
        im=np.array(canvas)
        faked.append(im)
        #canvas.show()
        #sleep(2)
        
    faked=np.array(faked)
    
    return faked

def noise_generator(shape):
    
    row,col= shape
    ch=1
    mean = 0.0
    var = 0.01
    sigma = var**0.5
    gauss = np.array(shape)
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = gauss
    return noisy#.astype('uint8')
    

def loadData(InputSize=[28,28]):
    X=[]
    
    for uri in glob.glob("images/*.jpg"):
        im = Image.open(uri)
        
        #convert to grayscale
        im=im.convert('L')
        
        im=im.resize((InputSize[0],InputSize[0]))
        
        im=np.array(im) 
        
        X.append(im)
    
    X=np.array(X)
    
    #split some test
    cutP=int(len(X)*0.8)
    X_train=X[cutP:]
    X_test=X[:cutP]
    return X_test,X_train

def loadDataRaw(InputSize=[28,28],mode="GREY"):
    X=[]
   
   
    for uri in glob.glob("images/*.jpg"):
        im = Image.open(uri)
        
        if mode=="GREY":
            #convert to grayscale
            im=im.convert('L')
        
        im=im.resize((InputSize[0],InputSize[0]))
        
        im=np.array(im) 
        
        X.append(im)
    
    X=np.array(X)
    
    
    return X

def prepareData(InputSize=[800,20],predHeight=1):
    X,Y=[],[]
    
    InputHeight=InputSize[1]
    
    notifiedResized=False
    
    for uri in glob.glob("images/*.jpg"):
    #for uri in glob.glob('/content/drive/My Drive/jamonjamonjamon/images/*.jpg'):
    
        #im = plt.imread(uri)
        im = Image.open(uri)
        
        #convert to grayscale
        im=im.convert('L')

        if im.size[0]>InputSize[0]:
            if not notifiedResized:
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("Images will be resized from ",im.size[0],"to",InputSize[0],"width")
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("")
                notifiedResized=True
            im=im.resize((InputSize[0],InputSize[0])) #!important, assuming the original images will be squares
        
        #convert to numpy
        im=np.array(im) 
        
        #im=np.mean(im, axis=1) #(convert to grey)
        #print("image resized",im.shape)
        
        #im=rgb2gray(im)
        
        
        #showImage(im)
        
        
        for i in range(200):
            
            xslice=im[i:i+InputHeight,:]
            #yslice=im[i+InputHeight:(i+InputHeight)+predHeight,:]
            yslice=im[i+InputHeight:(i+InputHeight)+predHeight,:]
            
            if xslice.shape[0]<InputHeight:
                break
            else:
                #print(xslice.shape)
                #yslice=yslice.flatten(order='C')
                
                xslice=xslice.reshape(xslice.shape+(1,))
                #yslice=yslice.reshape(yslice.shape+(1,))
                
                if yslice.shape[0]>0:
                    
                    X.append(xslice)
                    Y.append(yslice.flatten())
                
        #print("")
    #sys.exit()
    X=np.array( X )
    Y=np.array( Y )
    return X,Y



"""
def prepareData(InputSize=[800,20],predHeight=1):
    X,Y=[],[]
    
    InputHeight=InputSize[1]
    
    notifiedResized=False
    
    for uri in glob.glob("images/*.jpg"):
        #im = plt.imread(uri)
        im = Image.open(uri)
     
        if im.size[0]>InputSize[0]:
            if not notifiedResized:
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("Images will be resized from ",im.size[0],"to",InputSize[0],"width")
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("")
                notifiedResized=True
            im=im.resize((InputSize[0],InputSize[0])) #!important, assuming the original images will be squares
        
        #convert to numpy
        im=np.array(im)
        #print("image resized",im.shape)
        
        for i in range(200):
            
            xslice=im[i:i+InputHeight,:,:]
            yslice=im[i+InputHeight:(i+InputHeight)+predHeight,:,:]
            
            if xslice.shape[0]<InputHeight:
                break
            else:
                #print(xslice.shape)
                X.append(xslice)
                Y.append(yslice)
                
        #print("")
    #sys.exit()
             
        
    return np.array(X),np.array(Y)

"""
def denormalizeImage(image, from_min, from_max, to_min, to_max):
    
    #example : interval_mapping(image, 0, 255, 0.0, 1.0)
    
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def showImageNormal(im,mode="L"):
    imtoShow = 255 * im#[:,:,0]#[0][:,:,0] # Now scale by 255
    imtoShow = imtoShow.astype(np.uint8)
    showImage(imtoShow,mode)

def showImage(im,mode="RGB"):
    # if black and white  showImage(img[:,:,0]*255,"F")
                    
    img = Image.fromarray(im, mode)
    """
    if len(im.shape)>2:
        img = Image.fromarray(im, 'RGB')
        img.show()
    else:
        #img = Image.fromarray(im, 'RGB')
        img = Image.fromarray(im, 'L')
    """
  
    #print(img)
    img.show()
  
