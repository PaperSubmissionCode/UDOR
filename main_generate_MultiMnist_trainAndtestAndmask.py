
import numpy as np
import random

# load 20000 samples 20000*32*32
MnistRandom012ArrayNorm0_1=np.load("./multiMnistDataset_32x32/MnistRandom012ArrayNorm0_1.npz") #
imgs=MnistRandom012ArrayNorm0_1['imgs']
w = 32
h = 32
ch=1
Nend=10000
Nstart=0 # Nend -Nstart is the number of training datasets

unitLength=1 # can be different

N_p=Nend-Nstart
#============get all images===============
imgArray= np.empty((N_p,w,h,ch))

for k in range(Nstart,Nend):
    i=k-Nstart
    imgfloat=np.asarray(imgs[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray[i,:,:,:]=imgfloat
#==============get mask==============
maskArray=np.empty((N_p,unitLength*3))
mask1=np.ones((unitLength*3))
mask2=np.ones((unitLength*3))
mask3=np.ones((unitLength*3))
for k in range(unitLength,2*unitLength): # 3:6
    mask1[k-unitLength]=0
    mask2[k]=0
    mask3[k+unitLength]=0
"""for unitLength=1
mask1 [0. 1. 1.]
mask2 [1. 0. 1.]
mask3 [1. 1. 0.]
"""
print("mask1",mask1)
print("mask2",mask2)
print("mask3",mask3)
numArray=np.arange(N_p)
random.shuffle(numArray) # (0,10000)
for k in range(0,N_p): # 0:10000
    if numArray[k]> 2*N_p/3.0: # > 6666.666666666667
        maskArray[k]=mask3
    elif numArray[k]> N_p/3.0: # > 3333.3333333333335
        maskArray[k]=mask2
    else:
        maskArray[k]=mask1

# reset  mask value 0~64 (in the training stage)
for i in range(0,8):
    for j in range(0,8):
        if j % 4==0:
            maskTemp=mask1
        elif j%4==1:
            maskTemp=mask2
        elif j%4==2:
            maskTemp=mask3
        else:
            maskTemp=np.zeros((unitLength*3))

        maskArray[i*8+j]=maskTemp
# ================================================================
# For resetting first 64 imgs for visualize the effect of the reconstruction in the training stage
Mnist012 = np.load('npz_datas/Mnist012.npz')
zeroArray = Mnist012['zeroArray']
oneArray = Mnist012['oneArray']
twoArray = Mnist012['twoArray']

processNumber = 64
width = 32
height = 32
'''
zeroMax=5444
oneMax=6179
twoMax=5470
'''
maxValue = 5400
MnistRandom012Array_first_64 = np.zeros((processNumber, width, height))
for i in range(processNumber):
    # have three digit
    MnistRandom012Array_first_64[i, 1:15, 1:15] = zeroArray[random.randint(0, maxValue)]
    MnistRandom012Array_first_64[i, 1:15, 17:31] = oneArray[random.randint(0, maxValue)]
    MnistRandom012Array_first_64[i, 17:31, 1:15] = twoArray[random.randint(0, maxValue)]

MnistRandom012Array_first_64_norm_0_1 = MnistRandom012Array_first_64 / 255.0
# np.savez('./multiMnistDataset_32x32/unitLength{}_MnistRandom012Array_norm_test64.npz'.format(unitLength), imgs=MnistRandom012Array_first_64_norm_0_1)

# reset first 64 iamges in training set
for k in range(0,64):
    imgfloat=np.asarray(MnistRandom012Array_first_64_norm_0_1[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray[k,:,:,:]=imgfloat

np.savez('./npz_datas/unitLength{}_mnistMultiAndMask_'.format(unitLength)+str(Nend-Nstart)+'x'+str(w)+'x'+str(w)+'x'+str(ch)+'_train.npz',images=imgArray,masks=maskArray) # training datasets npz format

"""
check the training set
"""
# import cv2
# train_data=np.load('./npz_datas/unitLength1_mnistMultiAndMask_10000x32x32x1_train.npz')
# train_imgs=train_data['images']
# train_masks=train_data['masks']
#
# for i in range(0,100,1):
#     cv2.imshow('train img',train_imgs[i])
#     print('train mask:',train_masks[i])
#     cv2.waitKey()