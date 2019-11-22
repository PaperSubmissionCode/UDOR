import numpy as np
import scipy.misc
from scipy.misc import imsave,imresize
from PIL import Image, ImageDraw
import random
import math
import shutil
import os

Num=10000

flowerArray=np.empty((Num,32,32,3))
leafArray1=np.empty((Num,32,32,3))
leafArray2=np.empty((Num,32,32,3))
treeArray=np.empty((Num,32,32,3))

for k in range(0,Num):
    Iflower=Image.open('pattern_flower_tree_leaf/flowers_new/'+str(k+1)+'.png')
    Iflower_array = np.array(Iflower)
    flowerArray[k,:,:,:]=Iflower_array

    Ileaf1=Image.open('pattern_flower_tree_leaf/leafs_new/'+str(k+1)+'_1.png')
    Ileaf2=Image.open('pattern_flower_tree_leaf/leafs_new/'+str(k+1)+'_2.png')
    Ileaf1_array = np.array(Ileaf1)
    Ileaf2_array = np.array(Ileaf2)
    leafArray1[k,:,:,:]=Ileaf2_array
    leafArray2[k,:,:,:]=Ileaf1_array

    Itree=Image.open('pattern_flower_tree_leaf/trees/'+str(k+1)+'.png')
    Itree_array = np.array(Itree)
    treeArray[k,:,:,:]=Itree_array

print('read object images finished')
#============get combined pattern design images===========


maxValue=1000

batchSizeNum=20 
batchSize=64
processNumber =batchSizeNum*batchSize #1280

width = 64
height = 64
ch=3


flower1=np.empty((32,32,3))
leaf1_1=np.empty((32,32,3))
leaf1_2=np.empty((32,32,3))
tree1=np.empty((32,32,3))

flower2=np.empty((32,32,3))
leaf2_1=np.empty((32,32,3))
leaf2_2=np.empty((32,32,3))
tree2=np.empty((32,32,3))

imgTemp1=np.zeros((width,height,ch))
imgTemp2=np.zeros((width,height,ch))


imgArray1=np.zeros((processNumber,width,height,ch))
imgArray2=np.zeros((processNumber,width,height,ch))

for k in range(0,processNumber,4):
    flower1=flowerArray[random.randint(0, maxValue)]
    temp=random.randint(0, maxValue)
    leaf1_1=leafArray1[temp]
    leaf1_2=leafArray2[temp]
    tree1=treeArray[random.randint(0, maxValue)]

    flower2=flowerArray[random.randint(0, maxValue)]
    temp=random.randint(0, maxValue)
    leaf2_1=leafArray1[temp]
    leaf2_2=leafArray2[temp]
    tree2=treeArray[random.randint(0, maxValue)]

    imgTemp1[0:32,0:32,:]=flower1
    imgTemp1[32:64,0:32,:]=leaf1_1
    imgTemp1[0:32,32:64,:]=leaf1_2
    imgTemp1[32:64,32:64,:]=tree1
    imgArray1[k]=imgTemp1
    imgArray1[k+1]=imgTemp1
    imgArray1[k+2]=imgTemp1
    imgArray1[k+3]=imgTemp1

    imgTemp2[0:32,0:32,:]=flower2
    imgTemp2[32:64,0:32,:]=leaf2_1
    imgTemp2[0:32,32:64,:]=leaf2_2
    imgTemp2[32:64,32:64,:]=tree2
    imgArray2[k]=imgTemp2
    imgArray2[k+1]=imgTemp2
    imgArray2[k+2]=imgTemp2
    imgArray2[k+3]=imgTemp2

#===================trasfor the (Num,w,h) into float (Num,w,h,ch)====================

imgArray1Norm0_255= np.empty((processNumber,width,height,ch))
imgArray2Norm0_255=np.empty((processNumber,width,height,ch))
for k in range(processNumber):
    imgfloat=np.asarray(imgArray1[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,ch))
    imgArray1Norm0_255[k,:,:,:]=imgfloat

    imgfloat=np.asarray(imgArray2[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,ch))
    imgArray2Norm0_255[k,:,:,:]=imgfloat

# normalize to 0~1
imgArray1Norm0_1=imgArray1Norm0_255/255.0
imgArray2Norm0_1=imgArray2Norm0_255/255.0


#=============================== get mask ================================
unitLength=9
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

np.savez('npz_datas/pattern_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test_visualdata1.npz',images=imgArray1Norm0_1,masks=maskArray1)
np.savez('npz_datas/pattern_('+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_test_visualdata2.npz',images=imgArray2Norm0_1,masks=maskArray2)
