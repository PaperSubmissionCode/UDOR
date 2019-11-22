import numpy as np
from scipy.misc import imresize
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=False)

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

#============get combined mnist images===========
batchSizeNum=20
batchSize=64
processNumber =batchSizeNum*batchSize #1280
width = 32
height = 32

imgArray1=np.zeros((processNumber,width,height))
imgArray2=np.zeros((processNumber,width,height))

#=========== generate img1  and img2 ========================
zero1=np.zeros((14,14))
one1=np.zeros((14,14))
two1=np.zeros((14,14))
zero2=np.zeros((14,14))
one2=np.zeros((14,14))
two2=np.zeros((14,14))

imgTemp1=np.zeros((width,height))
imgTemp2=np.zeros((width,height))

for k in range(0,1280,4):
    zero1=zeroArray[random.randint(0, maxValue)]
    zero2=zeroArray[random.randint(0, maxValue)]
    one1=oneArray[random.randint(0, maxValue)]
    one2=oneArray[random.randint(0, maxValue)]
    two1=twoArray[random.randint(0, maxValue)]
    two2=twoArray[random.randint(0, maxValue)]
    # insert images
    imgTemp1[1:15,1:15]=zero1
    imgTemp1[1:15,17:31]=one1
    imgTemp1[17:31,1:15]=two1
    imgArray1[k]=imgTemp1
    imgArray1[k+1]=imgTemp1
    imgArray1[k+2]=imgTemp1
    imgArray1[k+3]=imgTemp1

    imgTemp2[1:15,1:15]=zero2
    imgTemp2[1:15,17:31]=one2
    imgTemp2[17:31,1:15]=two2
    imgArray2[k]=imgTemp2
    imgArray2[k+1]=imgTemp2
    imgArray2[k+2]=imgTemp2
    imgArray2[k+3]=imgTemp2

#===================transform the (Num,w,h) into float (Num,w,h,ch)====================
ch=1
imgArray1Norm0_255= np.empty((processNumber,width,height,ch))
imgArray2Norm0_255=np.empty((processNumber,width,height,ch))
for k in range(processNumber):
    imgfloat=np.asarray(imgArray1[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArray1Norm0_255[k,:,:,:]=imgfloat

    imgfloat=np.asarray(imgArray2[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArray2Norm0_255[k,:,:,:]=imgfloat

# normalize to 0~1
imgArray1Norm0_1=imgArray1Norm0_255/255.0
imgArray2Norm0_1=imgArray2Norm0_255/255.0

#=============================== get mask ================================
unitLength=1
partNum=3
maskArray1=np.empty((processNumber,unitLength*partNum))
maskArray2=np.empty((processNumber,unitLength*partNum))
maskall1=np.ones((unitLength*partNum))
maskall0=np.zeros((unitLength*partNum))

mask1=np.ones((unitLength*partNum))
mask2=np.ones((unitLength*partNum))
mask3=np.ones((unitLength*partNum))

mask1_rev=np.zeros((unitLength*partNum))
mask2_rev=np.zeros((unitLength*partNum))
mask3_rev=np.zeros((unitLength*partNum))

for k in range(unitLength,2*unitLength):
    mask1[k-unitLength]=0
    mask2[k]=0
    mask3[k+unitLength]=0
    mask1_rev[k-unitLength]=1
    mask2_rev[k]=1
    mask3_rev[k+unitLength]=1

for k in range(0,1280,4):
    maskArray1[k]=maskall1
    maskArray2[k]=maskall0
    maskArray1[k+1]=mask1
    maskArray2[k+1]=mask1_rev
    maskArray1[k+2]=mask2
    maskArray2[k+2]=mask2_rev
    maskArray1[k+3]=mask3
    maskArray2[k+3]=mask3_rev

np.savez('./npz_datas/mnist_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test_visualdata1.npz',images=imgArray1Norm0_1,masks=maskArray1)
np.savez('./npz_datas/mnist_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test_visualdata2.npz',images=imgArray2Norm0_1,masks=maskArray2)

imgArray3=np.zeros((processNumber,width,height,1))
np.savez('npz_datas/DSD_data1_3_mnist_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test.npz',images=imgArray1Norm0_255,masks=maskArray1)
np.savez('npz_datas/DSD_data2_mnist_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test.npz',images=imgArray3,masks=maskArray2)
np.savez('npz_datas/DSD_data4_mnist_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test.npz',images=imgArray2Norm0_255,masks=maskArray2)

# """
# check the testing set
# """
# import cv2
#
# test_data=np.load('./npz_datas/mnist_(20x64)x32x32x1_unitLength1_test_visualdata1.npz')
# test_imgs=test_data['images']
# test_masks=test_data['masks']
#
# for i in range(0,100,1):
#     cv2.imshow('test img',test_imgs[i])
#     print('test mask:', test_masks[i])
#     cv2.waitKey()