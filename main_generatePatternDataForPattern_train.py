import random
import numpy as np
import os
from PIL import Image

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
#============get combined pattern images (pairs:5000 and unpairs:5000)===========
maxValue=Num-1
processNumber = Num
#half=np.int32(processNumber/2)
width = 64
height = 64
ch=3

flower1=np.empty((32,32,3))
leaf1_1=np.empty((32,32,3))
leaf1_2=np.empty((32,32,3))
tree1=np.empty((32,32,3))



PatternArray=np.ones((processNumber,width,height,ch))*255

label012Array=np.ones((processNumber,3))*10 # initial for rember 0 1 2
#=================generate paired samples============================
for i  in range(processNumber): # supervised rate 0.5
    modeNum= random.randint(0, 3)
    # initial empty image
    imgTemp=np.ones((width,height,ch))*255

    if modeNum==0: #no obeject
        pass
    elif modeNum==1: #one obeject
        modeWho=random.randint(0, 2)
        if modeWho==0:
            flower1=flowerArray[random.randint(0, maxValue)]
            imgTemp[0:32,0:32,:]=flower1
            label012Array[i,0]=0
        elif modeWho==1:
            temp=random.randint(0, maxValue)
            leaf1_1=leafArray1[temp]
            leaf1_2=leafArray2[temp]
            imgTemp[32:64,0:32,:]=leaf1_1
            imgTemp[0:32,32:64,:]=leaf1_2
            label012Array[i,1]=1
        elif modeWho==2:
            tree1=treeArray[random.randint(0, maxValue)]
            imgTemp[32:64,32:64,:]=tree1
            label012Array[i,2]=2
        PatternArray[i]=imgTemp
    elif modeNum==2: #two obeject
        modeWho=random.randint(0, 2)
        if modeWho==0:
            flower1=flowerArray[random.randint(0, maxValue)]
            imgTemp[0:32,0:32,:]=flower1

            temp=random.randint(0, maxValue)
            leaf1_1=leafArray1[temp]
            leaf1_2=leafArray2[temp]
            imgTemp[32:64,0:32,:]=leaf1_1
            imgTemp[0:32,32:64,:]=leaf1_2
            label012Array[i,1]=1
            label012Array[i,0]=0
        elif modeWho==1:
            temp=random.randint(0, maxValue)
            leaf1_1=leafArray1[temp]
            leaf1_2=leafArray2[temp]
            imgTemp[32:64,0:32,:]=leaf1_1
            imgTemp[0:32,32:64,:]=leaf1_2

            tree1=treeArray[random.randint(0, maxValue)]
            imgTemp[32:64,32:64,:]=tree1
            label012Array[i,2]=2
            label012Array[i,1]=1
        elif modeWho==2:
            flower1=flowerArray[random.randint(0, maxValue)]
            imgTemp[0:32,0:32,:]=flower1
            label012Array[i,0]=0
            tree1=treeArray[random.randint(0, maxValue)]
            imgTemp[32:64,32:64,:]=tree1
            label012Array[i,2]=2
        PatternArray[i]=imgTemp
    else: # three object
        flower1=flowerArray[random.randint(0, maxValue)]
        imgTemp[0:32,0:32,:]=flower1

        temp=random.randint(0, maxValue)
        leaf1_1=leafArray1[temp]
        leaf1_2=leafArray2[temp]
        imgTemp[32:64,0:32,:]=leaf1_1
        imgTemp[0:32,32:64,:]=leaf1_2

        tree1=treeArray[random.randint(0, maxValue)]
        imgTemp[32:64,32:64,:]=tree1
        label012Array[i,0]=0
        label012Array[i,1]=1
        label012Array[i,2]=2

        PatternArray[i]=imgTemp

# normlize to 0~1
imgs=PatternArray/255.0


#=====================initial parameters===========================
w = 64
h = 64
ch=3
unitLength=9  ## import parameter
partNum=3
N_p=Num #10000
#===================get the train image=========================
imgArray= np.empty((N_p,w,h,ch))

for k in range(0,N_p):
    imgfloat=np.asarray(imgs[k],'f')
    imgfloat=np.reshape(imgfloat,(w,h,ch))
    imgArray[k,:,:,:]=imgfloat


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
imgTemp=np.ones((width,height,ch))*255
for i in range(64):
    flower1=flowerArray[random.randint(0, maxValue)]
    imgTemp[0:32,0:32,:]=flower1
    temp=random.randint(0, maxValue)
    leaf1_1=leafArray1[temp]
    leaf1_2=leafArray2[temp]
    imgTemp[32:64,0:32,:]=leaf1_1
    imgTemp[0:32,32:64,:]=leaf1_2
    tree1=treeArray[random.randint(0, maxValue)]
    imgTemp[32:64,32:64,:]=tree1

    imgfloat=np.asarray(imgTemp,'f')
    imgfloat=np.reshape(imgfloat,(w,h,ch))
    imgArray[i,:,:,:]=imgfloat/255.0
    label012Array[i,0]=0
    label012Array[i,1]=1
    label012Array[i,2]=2
# reset mask
for i in range(0,8):
    maskArray[i*8+0]=mask1
    maskArray[i*8+1]=mask2
    maskArray[i*8+2]=mask3
    maskArray[i*8+3]=np.ones((unitLength*partNum))
    maskArray[i*8+4]=mask1
    maskArray[i*8+5]=mask2
    maskArray[i*8+6]=mask3
    maskArray[i*8+7]=np.zeros((unitLength*partNum))


#np.savez('npz_datas/FashionMultiAndMask_'+str(Nend-Nstart)+'x'+str(w)+'x'+str(w)+'x'+str(ch)+'_train.npz',images=imgArray,masks=maskArray)


# Transform the lable into the onehot*part vector
classNum=4 # (empty,0,1,2)
labels=np.zeros((N_p,classNum*(classNum-1)))
for k in range(N_p):
    #for flower
    if label012Array[k,0]==0:
        labels[k,1]=1
    else:
        labels[k,0]=1
    #for  leaf 
    if label012Array[k,1]==1:
        labels[k,classNum+2]=1
    else:
        labels[k,classNum+0]=1
    #for tree
    if label012Array[k,2]==2:
        labels[k,2*classNum+3]=1
    else:
        labels[k,2*classNum+0]=1

save_dir='./npz_datas/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
np.savez(save_dir+'Pattern_MultiAndMaskAndLabels_unitLength_'+str(unitLength)+'_'+str(N_p)+'x'+str(w)+'x'+str(w)+'x'+str(ch)+'_train.npz',images=imgArray,masks=maskArray,gts=labels)



