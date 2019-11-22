import numpy as np
from scipy.misc import imresize
import random
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=False)

#================get one/two/three array
Num=len(mnist.train.labels)

zeroArray=np.empty((Num//5,14,14))
oneArray=np.empty((Num//5,14,14))
twoArray=np.empty((Num//5,14,14))
zeroCount=0
oneCount=0
twoCount=0

for k in range(0,Num):
    if mnist.train.labels[k]==0:
        im = mnist.train.images[k].reshape(28, 28)
        im1_2=imresize(im ,0.5)
        zeroArray[zeroCount,:,:]=im1_2
        zeroCount=zeroCount+1
    elif mnist.train.labels[k]==1:
        im = mnist.train.images[k].reshape(28, 28)
        im1_2=imresize(im ,0.5)
        oneArray[oneCount,:,:]=im1_2
        oneCount=oneCount+1
    elif mnist.train.labels[k]==2:
        im = mnist.train.images[k].reshape(28, 28)
        im1_2=imresize(im ,0.5)
        twoArray[twoCount,:,:]=im1_2
        twoCount=twoCount+1
    else:
        pass

    if k%1000==0:
        print("porcessing..."+str(k))

print(zeroCount) #5444
print(oneCount)  #6179
print(twoCount)  #5470
maxValue=5300

#============get combined mnist outfit images===========
processNumber = 10000
width = 32
height = 32

MnistRandom012Array=np.zeros((processNumber,width,height))
label012Array=np.ones((processNumber,3))*10 # initial for rember 0 1 2

for i in range(processNumber):

    modeNum= random.randint(0, 3)
    if modeNum==0: # no digit
        pass
    elif modeNum==1: 
        # have one digit 
        modeWho=random.randint(0, 2)
        if modeWho==0:
            MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            label012Array[i,0]=0
        elif modeWho==1:
            MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            label012Array[i,1]=1
        elif modeWho==2:
            MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
            label012Array[i,2]=2
    elif modeNum==2: 
        # have two digit 
        modeWho=random.randint(0, 2)
        if modeWho==0: #0 1
            MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            label012Array[i,0]=0
            label012Array[i,1]=1
        elif modeWho==1:# 0 2
            MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
            label012Array[i,0]=0
            label012Array[i,2]=2
        elif modeWho==2:# 1 2
            MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
            label012Array[i,1]=1
            label012Array[i,2]=2
    else:
        # have three digit
        MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        label012Array[i,0]=0
        label012Array[i,1]=1
        label012Array[i,2]=2

#print("check the range of  image value is in 0~255 before Normalize")
#print(MnistRandom012Array[1:5,1:15,17:31])
# normalize to 0~1
#imgs=MnistRandom012Array/255.0
imgs=MnistRandom012Array
#=====================initial parameters===========================
w = 32
h = 32
ch=1
Nend=10000  
Nstart=0
partNum=3
N_p=Nend-Nstart
save_dir='./npz_datas/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#===================get the train image=========================
imgArray= np.empty((N_p,w,h,ch))

for k in range(Nstart,Nend):
    i=k-Nstart
    imgfloat=np.asarray(imgs[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray[i,:,:,:]=imgfloat

for unitLength in range(1,2,1):
    #unitLength=3  ## import parameter
    #=====================   get mask===============================
    maskArray=np.empty((N_p,unitLength*partNum))
    mask1=np.ones((unitLength*partNum))
    mask2=np.ones((unitLength*partNum))
    mask3=np.ones((unitLength*partNum))
    for k in range(unitLength,2*unitLength):
        mask1[k-unitLength]=0
        mask2[k]=0
        mask3[k+unitLength]=0

    numArray=np.arange(N_p)
    random.shuffle(numArray)
    for k in range(0,N_p):
        if numArray[k]> 2*N_p/3.0:
            maskArray[k]=mask3
        elif numArray[k]>1*N_p/3.0:
            maskArray[k]=mask2
        else:
            maskArray[k]=mask1

    # ===reset 64 images and mask in the head of array====
    # which used to show results in the train
    #reset 64 images in the head of array
    imgTemp64=np.zeros((64,width,height))
    for i in range(64):
        imgTemp64[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        imgTemp64[i,17:31,1:15]=oneArray[random.randint(0, maxValue)]
        imgTemp64[i,1:15,17:31]=twoArray[random.randint(0, maxValue)]
        imgfloat=np.asarray(imgTemp64[i],'f')
        imgfloat=np.reshape(imgfloat,(w,h,1))
        imgArray[i,:,:,:]=imgfloat
        # reset lablel
        label012Array[i,0]=0
        label012Array[i,1]=1
        label012Array[i,2]=2

    # reset mask
    for i in range(0,8):
        maskArray[i*8+0]=mask1
        maskArray[i*8+1]=mask2
        maskArray[i*8+2]=mask3
        maskArray[i*8+3]=np.zeros((unitLength*partNum))
        maskArray[i*8+4]=mask1
        maskArray[i*8+5]=mask2
        maskArray[i*8+6]=mask3
        maskArray[i*8+7]=np.zeros((unitLength*partNum))

    # Transform the lable into the onehot*part vector
    classNum=4 # (empty,0,1,2)
    labels=np.zeros((N_p,classNum*(classNum-1)))
    for k in range(N_p):
        #for digit 0
        if label012Array[k,0]==0:
            labels[k,1]=1
        else:
            labels[k,0]=1
        #for digit 1    
        if label012Array[k,1]==1:
            labels[k,classNum+2]=1
        else:
            labels[k,classNum+0]=1
        #for digit 2
        if label012Array[k,2]==2:
            labels[k,2*classNum+3]=1
        else:
            labels[k,2*classNum+0]=1
    #print('imgArray[1:5,1:15,17:31]')
    #print(imgArray[100:105,1:15,17:31])

    imgArrayNorm0_1=imgArray/255.0
    #print(labels[64:128])
    np.savez(save_dir+'SAE_mnistMultiAndMaskAndLabels_unitLength_'+str(unitLength)+'_'+str(Nend-Nstart)+'x'+str(w)+'x'+str(w)+'x'+str(ch)+'_train.npz',images=imgArrayNorm0_1,masks=maskArray,gts=labels)



