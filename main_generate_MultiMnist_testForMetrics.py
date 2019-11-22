import numpy as np
from scipy.misc import imresize
import random
from help_function import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

#================get one/two/three array
Num=len(mnist.test.labels)

zeroArray=np.empty((Num//5,14,14))
oneArray=np.empty((Num//5,14,14))
twoArray=np.empty((Num//5,14,14))
zeroCount=0
oneCount=0
twoCount=0

for k in range(0,Num):
    if mnist.test.labels[k]==0:
        im = mnist.test.images[k].reshape(28, 28)
        im1_2=imresize(im ,0.5)
        zeroArray[zeroCount,:,:]=im1_2
        zeroCount=zeroCount+1
    elif mnist.test.labels[k]==1:
        im = mnist.test.images[k].reshape(28, 28)
        im1_2=imresize(im ,0.5)
        oneArray[oneCount,:,:]=im1_2
        oneCount=oneCount+1
    elif mnist.test.labels[k]==2:
        im = mnist.test.images[k].reshape(28, 28)
        im1_2=imresize(im ,0.5)
        twoArray[twoCount,:,:]=im1_2
        twoCount=twoCount+1
    else:
        pass

    if k%1000==0:
        print("porcessing..."+str(k))

print(zeroCount) #980
print(oneCount)  #1135
print(twoCount)  #1032
maxValue=980

#============get combined mnist outfit images===========
batchSizeNum=47
batchSize=64
processNumber =batchSizeNum*batchSize #3008
width = 32
height = 32

imgArray=np.zeros((processNumber,width,height))
imgArrayResetGt=np.zeros((processNumber,width,height))
label012Array=-1*np.ones((processNumber,3)) # initial for rember 0 1 2
## fro DSD
imgArrayPaired=np.zeros((processNumber,width,height))
attrisArrayPaired=np.zeros(processNumber)
#===========for fixed digit 0 ========================
zero_fixed=np.zeros((14,14))
one=np.zeros((14,14))
two=np.zeros((14,14))
patchEmpty=np.zeros((14,14))
for k in range(0,10):
    zero_fixed=zeroArray[random.randint(0, maxValue)]
    for t in range(0,100):
        # label 0
        label012Array[k*100+t,0]=0
        # one / empty
        modeWho=random.randint(0,1)
        if modeWho==1:
            one=oneArray[random.randint(0, maxValue)]
            label012Array[k*100+t,1]=1
        else:
            one=patchEmpty
        # two / empty
        modeWho=random.randint(0,1)
        if modeWho==1:
            two=twoArray[random.randint(0, maxValue)]
            label012Array[k*100+t,2]=2
        else:
            two=patchEmpty
        # insert images
        imgArray[k*100+t,1:15,1:15]=zero_fixed
        imgArray[k*100+t,1:15,17:31]=one
        imgArray[k*100+t,17:31,1:15]=two
        # paired sample for swap DSD
        imgArrayPaired[k * 100 + t, 1:15, 1:15] = patchEmpty
        imgArrayPaired[k * 100 + t, 1:15, 17:31] = one
        imgArrayPaired[k * 100 + t, 17:31, 1:15] = two
        attrisArrayPaired[k * 100 + t] = 0

        imgArrayResetGt[k*100+t]=imgArray[k*100+t]
        imgArrayResetGt[k*100+t,1:15,1:15]=patchEmpty # remove digit zero

#===========for fixed digit 1 ========================
zero=np.zeros((14,14))
one_fixed=np.zeros((14,14))
two=np.zeros((14,14))
startIndex=1000
for k in range(0,10):
    one_fixed=oneArray[random.randint(0, maxValue)]
    for t in range(0,100):
        # label 1
        label012Array[startIndex+k*100+t,1]=1
        # zero / empty
        modeWho=random.randint(0,1)
        if modeWho==1:
            zero=zeroArray[random.randint(0, maxValue)]
            label012Array[startIndex+k*100+t,0]=0
        else:
            zero=patchEmpty
        # two / empty
        modeWho=random.randint(0,1)
        if modeWho==1:
            two=twoArray[random.randint(0, maxValue)]
            label012Array[startIndex+k*100+t,2]=2
        else:
            two=patchEmpty
        # insert images
        imgArray[startIndex+k*100+t,1:15,1:15]=zero
        imgArray[startIndex+k*100+t,1:15,17:31]=one_fixed
        imgArray[startIndex+k*100+t,17:31,1:15]=two
        # paired sample for swap DSD
        imgArrayPaired[startIndex + k * 100 + t, 1:15, 1:15] = zero
        imgArrayPaired[startIndex + k * 100 + t, 1:15, 17:31] = patchEmpty
        imgArrayPaired[startIndex + k * 100 + t, 17:31, 1:15] = two
        attrisArrayPaired[startIndex + k * 100 + t] = 1

        imgArrayResetGt[startIndex+k*100+t]=imgArray[startIndex+k*100+t]
        imgArrayResetGt[startIndex+k*100+t,1:15,17:31]=patchEmpty # remove digit one

#===========for fixed digit 2 ========================
zero=np.zeros((14,14))
one=np.zeros((14,14))
two_fixed=np.zeros((14,14))
startIndex=2000
for k in range(0,10):
    two_fixed=twoArray[random.randint(0, maxValue)]
    for t in range(0,100):
        # label 2
        label012Array[startIndex+k*100+t,2]=2
        # zero / empty
        modeWho=random.randint(0,1)
        if modeWho==1:
            zero=zeroArray[random.randint(0, maxValue)]
            label012Array[startIndex+k*100+t,0]=0
        else:
            zero=patchEmpty
        # one / empty
        modeWho=random.randint(0,1)
        if modeWho==1:
            one=oneArray[random.randint(0, maxValue)]
            label012Array[startIndex+k*100+t,1]=1
        else:
            one=patchEmpty
        # insert images
        imgArray[startIndex+k*100+t,1:15,1:15]=zero
        imgArray[startIndex+k*100+t,1:15,17:31]=one
        imgArray[startIndex+k*100+t,17:31,1:15]=two_fixed
        # DSD
        imgArrayPaired[startIndex + k * 100 + t, 1:15, 1:15] = zero
        imgArrayPaired[startIndex + k * 100 + t, 1:15, 17:31] = one
        imgArrayPaired[startIndex + k * 100 + t, 17:31, 1:15] = patchEmpty
        attrisArrayPaired[startIndex + k * 100 + t] = 1

        imgArrayResetGt[startIndex+k*100+t]=imgArray[startIndex+k*100+t]
        imgArrayResetGt[startIndex+k*100+t,17:31,1:15]=patchEmpty # remove digit two

#===================transfer the (Num,w,h) into float (Num,w,h,ch)====================
ch=1
imgArrayNorm0_255= np.empty((processNumber,width,height,ch))
imgArrayResetGtNorm0_255=np.empty((processNumber,width,height,ch))
# DSD
imgArrayPaired0_255= np.empty((processNumber,width,height,ch))
for k in range(processNumber):
    imgfloat=np.asarray(imgArray[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArrayNorm0_255[k,:,:,:]=imgfloat

    imgfloat = np.asarray(imgArrayPaired[k], 'f')
    imgfloat = np.reshape(imgfloat, (width, height, 1))
    imgArrayPaired0_255[k, :, :, :] = imgfloat

    imgfloat=np.asarray(imgArrayResetGt[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArrayResetGtNorm0_255[k,:,:,:]=imgfloat

# normalize to 0~1
imgArrayNorm0_1=imgArrayNorm0_255/255.0
imgArrayResetGtNorm0_1=imgArrayResetGtNorm0_255/255.0
imgArrayPaired0_1=imgArrayPaired0_255/255.2

#======******  get mask (mask is decided by visual the reconstructed result after train) *******========
# ===handle the left 8 (3008-3000) images ====
for t in range(3000,3008):
    imgArrayNorm0_1[t]=imgArrayNorm0_1[t-3000]
    imgArrayResetGtNorm0_1[t]=imgArrayResetGtNorm0_1[t-3000]

    imgArrayNorm0_255[t] = imgArrayNorm0_255[t - 3000]
    imgArrayResetGtNorm0_255[t] = imgArrayResetGtNorm0_255[t - 3000]

    label012Array[t]=label012Array[t-3000]

for k in range(1,2,1):
    unitLength=k
    partNum=3
    maskArray=np.empty((processNumber,unitLength*partNum))
    maskArraySAE=np.empty((processNumber,unitLength*partNum))

    mask1=np.ones((unitLength*partNum))
    mask2=np.ones((unitLength*partNum))
    mask3=np.ones((unitLength*partNum))

    for k in range(unitLength,2*unitLength):
        mask1[k-unitLength]=0 # [0,1,1]
        mask2[k]=0 # [1,0,1]
        mask3[k+unitLength]=0 # [1,1,0]
    for t in range(1000,2000):
        maskArray[t-1000]=mask3  # mask for digit 2
        maskArray[t]=mask2       # mask for digit 1
        maskArray[1000+t]=mask1  # mask for digit 0

        maskArraySAE[t-1000]=mask1  # mask for digit 0
        maskArraySAE[t]=mask2       # mask for digit 1
        maskArraySAE[1000+t]=mask3  # mask for digit 2

    # for UDOR, SAE
    np.savez('./npz_datas/mnistImgs_GTimages_mask_GTlabel_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+
             '_unitLength'+str(unitLength)+'_CodeImageDataset_forUDOR.npz',images=imgArrayNorm0_1,imagesGT=imgArrayResetGtNorm0_1,masks=maskArray,labelsGT=label012Array)
    np.savez('./npz_datas/mnistImgs_GTimages_mask_GTlabel_(' + str(batchSizeNum) + 'x' + str(batchSize) + ')x' + str(
        width) + 'x' + str(height) + 'x' + str(ch) + '_unitLength' + str(unitLength) + '_CodeImageDataset_forSAE.npz',
             images=imgArrayNorm0_1, imagesGT=imgArrayResetGtNorm0_1, masks=maskArraySAE, labelsGT=label012Array)

    # paired mask
    mask1ArrayDSD, mask2ArrayDSD = get3AttributeMaskFromLabels(attrisArrayPaired, unitLength, processNumber)
    np.savez('./npz_DSD_dataset/Arr1_mnistImgsForDSD_GTimgs_mask_GTlabel_(' + str(batchSizeNum) + 'x' + str(
        batchSize) + ')x' + str(width) + 'x' + str(height) + 'x' + str(ch) + '_unitLength' + str(
        unitLength) + '_dataset.npz', images=imgArrayNorm0_255, imagesGT=imgArrayResetGtNorm0_255, masks=mask1ArrayDSD,
             labelsGT=label012Array)
    np.savez('./npz_DSD_dataset/Arr2_mnistImgsForDSD_GTimgs_mask_GTlabel_(' + str(batchSizeNum) + 'x' + str(
        batchSize) + ')x' + str(width) + 'x' + str(height) + 'x' + str(ch) + '_unitLength' + str(
        unitLength) + '_dataset.npz', images=imgArrayPaired0_255, masks=mask2ArrayDSD)


