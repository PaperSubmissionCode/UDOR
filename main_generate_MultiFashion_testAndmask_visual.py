import numpy as np
from scipy.misc import imresize
import random

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


from tensorflow.examples.tutorials.mnist import input_data
fashion= input_data.read_data_sets('./Fashion_data/', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',one_hot=False)

#================get top/pants/bag/shoes array
Num=len(fashion.train.labels)

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
batchSizeNum=20 
batchSize=64
processNumber =batchSizeNum*batchSize #1280
width = 32
height = 32

imgArray1=np.zeros((processNumber,width,height))
imgArray2=np.zeros((processNumber,width,height))

top1=np.zeros((14,14))
down1=np.zeros((14,14))
bag1=np.zeros((14,14))
shoes1=np.zeros((14,14))
top2=np.zeros((14,14))
down2=np.zeros((14,14))
bag2=np.zeros((14,14))
shoes2=np.zeros((14,14))

imgTemp1=np.zeros((width,height))
imgTemp2=np.zeros((width,height))

for k in range(0,processNumber,4):

    top1=topArray[random.randint(0, maxValue)]
    down1=downArray[random.randint(0, maxValue)]
    bag1=bagArray[random.randint(0, maxValue)]
    shoes1=shoesArray[random.randint(0, maxValue)]

    top2=topArray[random.randint(0, maxValue)]
    down2=downArray[random.randint(0, maxValue)]
    bag2=bagArray[random.randint(0, maxValue)]
    shoes2=shoesArray[random.randint(0, maxValue)] 

    imgTemp1[1:15,1:15]=top1
    imgTemp1[17:31,1:15]=down1
    imgTemp1[1:15,17:31]=bag1
    imgTemp1[17:31,17:31]=shoes1

    imgTemp2[1:15,1:15]=top2
    imgTemp2[17:31,1:15]=down2
    imgTemp2[1:15,17:31]=bag2
    imgTemp2[17:31,17:31]=shoes2

    imgArray1[k]=imgTemp1
    imgArray1[k+1]=imgTemp1
    imgArray1[k+2]=imgTemp1
    imgArray1[k+3]=imgTemp1
        
    imgArray2[k]=imgTemp2
    imgArray2[k+1]=imgTemp2
    imgArray2[k+2]=imgTemp2
    imgArray2[k+3]=imgTemp2


#=====================initial parameters===========================
w = 32
h = 32
ch=1
partNum=4

#===================get the train image=========================

imgArray1Norm0_255= np.empty((processNumber,width,height,ch))
imgArray2Norm0_255=np.empty((processNumber,width,height,ch))

for k in range(0,processNumber):
    i=k
    imgfloat=np.asarray(imgArray1[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray1Norm0_255[i,:,:,:]=imgfloat

    imgfloat=np.asarray(imgArray2[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,1))
    imgArray2Norm0_255[i,:,:,:]=imgfloat

# normalize to 0~1
imgArray1Norm0_1=imgArray1Norm0_255/255.0
imgArray2Norm0_1=imgArray2Norm0_255/255.0

unitLength=1 ## import parameter
#=====================   get mask===============================
for unitLength in range(1,2,1):
    mask1=np.ones((unitLength*partNum))
    mask2=np.ones((unitLength*partNum))
    mask3=np.ones((unitLength*partNum))
    mask4=np.ones((unitLength*partNum))
    mask1_rev=np.zeros((unitLength*partNum))
    mask2_rev=np.zeros((unitLength*partNum))
    mask3_rev=np.zeros((unitLength*partNum))
    mask4_rev=np.zeros((unitLength*partNum))
    for k in range(unitLength,2*unitLength):
        mask1[k-unitLength]=0
        mask2[k]=0
        mask3[k+unitLength]=0
        mask4[k+2*unitLength]=0
        mask1_rev[k-unitLength]=1
        mask2_rev[k]=1
        mask3_rev[k+unitLength]=1
        mask4_rev[k+2*unitLength]=1

    maskArray1=np.empty((processNumber,unitLength*partNum))
    maskArray2=np.empty((processNumber,unitLength*partNum))
    for k in range(0,processNumber,4):
        maskArray1[k]=mask1
        maskArray2[k]=mask1_rev
        maskArray1[k+1]=mask2
        maskArray2[k+1]=mask2_rev
        maskArray1[k+2]=mask3
        maskArray2[k+2]=mask3_rev
        maskArray1[k+3]=mask4
        maskArray2[k+3]=mask4_rev


    np.savez('./npz_datas/fashion_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test_visualdata1.npz',images=imgArray1Norm0_1,masks=maskArray1)
    np.savez('./npz_datas/fashion_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test_visualdata2.npz',images=imgArray2Norm0_1,masks=maskArray2)