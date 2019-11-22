
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imresize
import random
import os

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=False)

# check
# print(mnist)
# print(mnist.train.labels[0:10])
# print(mnist.train.images[0].size)

im = mnist.train.images[7].reshape(28, 28)
im1_2=imresize(im ,0.5)

# visualize/check
# plt.subplot(1, 2, 1)
# plt.imshow(im,'gray')
# plt.subplot(1, 2, 2)
# plt.imshow(im1_2,'gray')
# plt.axis('off')
# plt.show()

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

# counts for later maxValue
# print(zeroCount) #5444
# print(oneCount)  #6179
# print(twoCount)  #5470
save_path='./npz_datas/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
np.savez(save_path+'Mnist012.npz',zeroArray=zeroArray,oneArray=oneArray,twoArray=twoArray)

#================================================================

Mnist012=np.load(save_path+'Mnist012.npz')
zeroArray=Mnist012['zeroArray']
oneArray=Mnist012['oneArray']
twoArray=Mnist012['twoArray']
'''
plt.subplot(1, 2, 1)
plt.imshow(zeroArray[0],'gray')
plt.subplot(1, 2, 2)
plt.imshow(twoArray[1],'gray')
plt.axis('off')
plt.show()
'''
#=================================================================

directoryName = './multiMnistDataset_32x32/'
if not os.path.exists(directoryName):
    os.mkdir(directoryName)

processNumber = 20000
width = 32
height = 32

MnistRandom012Array=np.zeros((processNumber,width,height))
'''
zeroMax=5444
oneMax=6179
twoMax=5470
'''
maxValue=5400

for i in range(processNumber):

    modeNum= random.randint(0, 3)
    if modeNum==0: # no digit
        pass
    elif modeNum==1: 
        # have one digit 
        modeWho=random.randint(0, 2) 
        if modeWho==0:
            MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        elif modeWho==1:
            MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
    elif modeNum==2: 
        # have two digit 
        modeWho=random.randint(0, 2) 
        if modeWho==0: #0 1
            MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==1:# 0 2
            MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        elif modeWho==2:# 1 2
            MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
    else:
        # have three digit
        MnistRandom012Array[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        MnistRandom012Array[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        MnistRandom012Array[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]

# middle file
# np.savez(directoryName+'MnistRandom012Array.npz',imgs=MnistRandom012Array)

imgs=MnistRandom012Array/255.0 # norm to 0-1
np.savez(directoryName+'MnistRandom012ArrayNorm0_1.npz',imgs=imgs)