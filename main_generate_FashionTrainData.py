import numpy as np
from scipy.misc import imresize
import random
import os
'''
Label 	Description
0 	T-shirt/top    1
1 	Trouser        1
2 	Pullover
3 	Dress
4 	Coat
5 	Sandal
6 	Shirt
7 	Sneaker
8 	Bag            1
9 	Ankle boot     1
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

fashion= input_data.read_data_sets('./Fashion_data/', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',one_hot=False)

#================get top/pants/bag/shoes array
Num=len(fashion.train.labels)
save_dir='./npz_datas/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

topArray=np.empty((Num//5,14,14))
downArray=np.empty((Num//5,14,14))
bagArray=np.empty((Num//5,14,14))
shoesArray=np.empty((Num//5,14,14))

topCount=0
downCount=0
bagCount=0
shoesCount=0

for k in range(0,Num):
    if fashion.train.labels[k]==0:
        im = fashion.train.images[k].reshape(28, 28)
        im1_2=imresize(im,0.5)
        topArray[topCount,:,:]=im1_2
        topCount=topCount+1
    elif fashion.train.labels[k]==1:
        im = fashion.train.images[k].reshape(28, 28)
        im1_2=imresize(im,0.5)
        downArray[downCount,:,:]=im1_2
        downCount=downCount+1
    elif fashion.train.labels[k]==8:
        im = fashion.train.images[k].reshape(28, 28)
        im1_2=imresize(im,0.5)
        bagArray[bagCount,:,:]=im1_2
        bagCount=bagCount+1
    elif fashion.train.labels[k]==9:
        im = fashion.train.images[k].reshape(28, 28)
        im1_2=imresize(im,0.5)
        shoesArray[shoesCount,:,:]=im1_2
        shoesCount=shoesCount+1
    else:
        pass

    if k%5000==0:
        print("porcessing..."+str(k))

print(topCount) #5444
print(downCount)  #6179
print(bagCount)  #5389
print(shoesCount)  #5454
maxValue=5300

#============get combined fashion outfit images===========
processNumber = 20000
width = 32
height = 32

FashionRandom1234Array=np.zeros((processNumber,width,height))

for i in range(processNumber):

    modeNum= random.randint(0, 4)
    if modeNum==0: # no digit
        pass
    elif modeNum==1:
        # have one clothing
        modeWho=random.randint(0, 3)
        if modeWho==0:
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
        elif modeWho==1:
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
        elif modeWho==2:
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
        elif modeWho==3:
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
    elif modeNum==2:
        # have two clothing
        modeWho=random.randint(0, 3)
        if modeWho==0:  #ab
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
        elif modeWho==1:#ac
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
        elif modeWho==2:#ad
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
        elif modeWho==3:#bc
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
        elif modeWho==4:#bd
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
        elif modeWho==5:#cd
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
    elif modeNum==3:
        # have three clothing
        modeWho=random.randint(0, 3)
        if modeWho==0: #
            #FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
        elif modeWho==1:#
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            #FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
        elif modeWho==2:# 1 2
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
            #FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
        else:
            FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
            FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
            #FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
    else:
        # have four clothing
        FashionRandom1234Array[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
        FashionRandom1234Array[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
        FashionRandom1234Array[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
        FashionRandom1234Array[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]

#print("check the range of  image value is in 0~255 before Normalize")
#print(FashionRandom1234Array[1:5,:,:])
# normalize to 0~1
imgs=FashionRandom1234Array/255.0

#=====================initial parameters===========================
w = 32
h = 32
ch=1
Nend=20000
Nstart=0
unitLength=1  ## import parameter
partNum=4
N_p=Nend-Nstart
#===================get the train image=========================
imgArray= np.empty((N_p,w,h,ch))

for k in range(Nstart,Nend):
    i=k-Nstart
    imgfloat=np.asarray(imgs[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray[i,:,:,:]=imgfloat

#=====================   get mask===============================
maskArray=np.empty((N_p,unitLength*partNum))
mask1=np.ones((unitLength*partNum))
mask2=np.ones((unitLength*partNum))
mask3=np.ones((unitLength*partNum))
mask4=np.ones((unitLength*partNum))
for k in range(unitLength,2*unitLength):
    mask1[k-unitLength]=0
    mask2[k]=0
    mask3[k+unitLength]=0
    mask4[k+2*unitLength]=0
print(mask1)
print(mask2)
print(mask3)
print(mask4)
numArray=np.arange(N_p)
random.shuffle(numArray)
for k in range(0,N_p):
    if numArray[k]> 3*N_p/4.0:
        maskArray[k]=mask4
    elif numArray[k]>2*N_p/4.0:
        maskArray[k]=mask3
    elif numArray[k]>N_p/4.0:
        maskArray[k]=mask2
    else:
        maskArray[k]=mask1


# ===reset 64 images and mask in the head of array====
# which used to show results in the train
#reset 64 images in the head of array
imgTemp64=np.zeros((64,width,height))
for i in range(0,64,4):
    imgTemp64[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
    imgTemp64[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
    imgTemp64[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
    imgTemp64[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
    imgfloat=np.asarray(imgTemp64[i],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray[i,:,:,:]=imgfloat/255.0
    imgArray[i+1, :, :, :] = imgfloat / 255.0
    imgArray[i+2, :, :, :] = imgfloat / 255.0
    imgArray[i+3, :, :, :] = imgfloat / 255.0

# reset mask
for i in range(0,8):
    maskArray[i*8+0]=mask1
    maskArray[i*8+1]=mask2
    maskArray[i*8+2]=mask3
    maskArray[i*8+3]=mask4
    maskArray[i*8+4]=mask1
    maskArray[i*8+5]=mask2
    maskArray[i*8+6]=mask3
    maskArray[i*8+7]=np.zeros((unitLength*partNum))


np.savez(save_dir+'FashionMultiAndMask_unitlength{}_'.format(unitLength)+str(Nend-Nstart)+'x'+str(w)+'x'+str(w)+'x'+str(ch)+'_train.npz',images=imgArray,masks=maskArray)

#========================for test============================
imgTemp64=np.zeros((64,width,height))
imgArrayTest64=np.zeros((64,width,height,1))
for i in range(0,64,4):
    imgTemp64[i,1:15,1:15]=topArray[random.randint(0, maxValue)]
    imgTemp64[i,17:31,1:15]=downArray[random.randint(0, maxValue)]
    imgTemp64[i,1:15,17:31]=bagArray[random.randint(0, maxValue)]
    imgTemp64[i,17:31,17:31]=shoesArray[random.randint(0, maxValue)]
    imgfloat=np.asarray(imgTemp64[i],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArrayTest64[i,:,:,:]=imgfloat/255.0
    imgArrayTest64[i+1,:,:,:]=imgfloat/255.0
    imgArrayTest64[i+2,:,:,:]=imgfloat/255.0
    imgArrayTest64[i+3,:,:,:]=imgfloat/255.0
# reset mask
maskArraytest=np.empty((64,unitLength*partNum))
for i in range(0,8):
    maskArraytest[i*8+0]=mask1
    maskArraytest[i*8+1]=mask2
    maskArraytest[i*8+2]=mask3
    maskArraytest[i*8+3]=mask4
    maskArraytest[i*8+4]=mask1
    maskArraytest[i*8+5]=mask2
    maskArraytest[i*8+6]=mask3
    maskArraytest[i*8+7]=mask4

np.savez(save_dir+'FashionMultiAndMask_unitlength{}_64x'.format(unitLength)+str(w)+'x'+str(w)+'x'+str(ch)+'_test.npz',images=imgArrayTest64,masks=maskArraytest)



