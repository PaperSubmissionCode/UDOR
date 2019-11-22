from scipy.misc import imresize
import random
from help_function import *
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
#============get combined mnist images (pairs:5000 and unpairs:5000)===========
processNumber = 5000
#half=np.int32(processNumber/2)
width = 32
height = 32
ch=1
MnistRandom012Array1=np.zeros((processNumber,width,height))
MnistRandom012Array2=np.zeros((processNumber,width,height))

AttrisArrayPaired=np.zeros(processNumber)
#=================generate paired samples============================
for i  in range(processNumber): # supervised rate 0.5
    modeNum= random.randint(0, 2)
    if modeNum==0: # same zero
        #========same digit zero============
        temp=random.randint(0, maxValue)
        MnistRandom012Array1[i,1:15,1:15]=zeroArray[temp]
        MnistRandom012Array2[i,1:15,1:15]=zeroArray[temp]
        AttrisArrayPaired[i]=0
        # random 1 2 for array1
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only two ; 3: one and two
        if modeWho==1:
            MnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            MnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            MnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        # random 1 2 for array2
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only two ; 3: one and two
        if modeWho==1:
            MnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            MnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            MnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]

    elif modeNum==1: # same one
        #========same digit one============
        temp=random.randint(0, maxValue)
        MnistRandom012Array1[i,1:15,17:31]=oneArray[temp]
        MnistRandom012Array2[i,1:15,17:31]=oneArray[temp]
        AttrisArrayPaired[i]=1
        # random 0 2 for array1
        modeWho=random.randint(1, 3) #0: empty  ;  1:only zero  2:only two ; 3: zero and two
        if modeWho==1:
            MnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            MnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        # random 0 2 for array2
        modeWho=random.randint(1, 3) #0: empty  ;  1:only zero  2:only two ; 3: zero and two
        if modeWho==1:
            MnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            MnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
    else: # same two
        #========same digit two============
        temp=random.randint(0, maxValue)
        MnistRandom012Array1[i,17:31,1:15]=twoArray[temp]
        MnistRandom012Array2[i,17:31,1:15]=twoArray[temp]
        AttrisArrayPaired[i]=2
        # random 0 1 for array1
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only zero ; 3: zero and one
        if modeWho==1:
            MnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        else:
            MnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        # random 0 1 for array2
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only zero ; 3: zero and one
        if modeWho==1:
            MnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            MnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        else:
            MnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            MnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]

#=================generate unpaired samples============================
unpairMnistRandom012Array1=np.zeros((processNumber,width,height))
unpairMnistRandom012Array2=np.zeros((processNumber,width,height))
AttrisArrayunPaired=np.zeros(processNumber)
for i  in range(processNumber): # unsupervised  data
    modeNum= random.randint(0, 2)
    if modeNum==0: # same zero
        #========same digit zero============
        unpairMnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        unpairMnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        AttrisArrayunPaired[i]=random.randint(0, 2)
        # random 1 2 for array1
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only two ; 3: one and two
        if modeWho==1:
            unpairMnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            unpairMnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            unpairMnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            unpairMnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        # random 1 2 for array2
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only two ; 3: one and two
        if modeWho==1:
            unpairMnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            unpairMnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            unpairMnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
            unpairMnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]

    elif modeNum==1: # same one
        #========same digit one============
        unpairMnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        unpairMnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        AttrisArrayunPaired[i]=random.randint(0, 2)
        # random 0 2 for array1
        modeWho=random.randint(1, 3) #0: empty  ;  1:only zero  2:only two ; 3: zero and two
        if modeWho==1:
            unpairMnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        elif modeWho==2:
            unpairMnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            unpairMnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            unpairMnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        # random 0 2 for array2
        modeWho=random.randint(1, 3) #0: empty  ;  1:only zero  2:only two ; 3: zero and two
        if modeWho==1:
            unpairMnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        elif modeWho==2:
            unpairMnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        else:
            unpairMnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            unpairMnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
    else: # same two
        #========same digit two============
        unpairMnistRandom012Array1[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        unpairMnistRandom012Array2[i,17:31,1:15]=twoArray[random.randint(0, maxValue)]
        AttrisArrayunPaired[i]=random.randint(0, 2)
        # random 0 1 for array1
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only zero ; 3: zero and one
        if modeWho==1:
            unpairMnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            unpairMnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        else:
            unpairMnistRandom012Array1[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            unpairMnistRandom012Array1[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        # random 0 1 for array2
        modeWho=random.randint(1, 3) #0: empty  ;  1:only one  2:only zero ; 3: zero and one
        if modeWho==1:
            unpairMnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]
        elif modeWho==2:
            unpairMnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
        else:
            unpairMnistRandom012Array2[i,1:15,1:15]=zeroArray[random.randint(0, maxValue)]
            unpairMnistRandom012Array2[i,1:15,17:31]=oneArray[random.randint(0, maxValue)]



imgArray1=np.zeros((processNumber,width,height,ch))
imgArray2=np.zeros((processNumber,width,height,ch))
imgArray3=np.zeros((processNumber,width,height,ch))
imgArray4=np.zeros((processNumber,width,height,ch))

for k in range(processNumber):
    imgfloat=np.asarray(MnistRandom012Array1[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArray1[k,:,:,:]=imgfloat
    imgfloat=np.asarray(MnistRandom012Array2[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArray2[k,:,:,:]=imgfloat
    imgfloat=np.asarray(unpairMnistRandom012Array1[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArray3[k,:,:,:]=imgfloat
    imgfloat=np.asarray(unpairMnistRandom012Array2[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArray4[k,:,:,:]=imgfloat


rate=0.5
for unitLatentLength in range(1,2,1):

    #paired mask
    mask1Array,mask2Array=get3AttributeMaskFromLabels(AttrisArrayPaired,unitLatentLength,processNumber)
    #unpaired mask
    mask3Array,mask4Array=get3AttributeMaskFromLabels(AttrisArrayunPaired,unitLatentLength,processNumber)

    save_dir='./npz_DSD_dataset/'
    if not os.path.exists(save_dir):\
        os.mkdir(save_dir)
    # paired
    np.savez(save_dir+'dataset1_255img32x32_mask_UnitLen_'+str(unitLatentLength)+'supervised='+str(rate)+'_N='+str(processNumber)+'.npz',images=imgArray1,masks=mask1Array)
    np.savez(save_dir+'dataset2_255img32x32_mask_UnitLen_'+str(unitLatentLength)+'supervised='+str(rate)+'_N='+str(processNumber)+'.npz',images=imgArray2,masks=mask2Array)
    #np.save('npz_DSD_dataset/AttrisArray12_255img32x32_UnitLen_'+str(unitLatentLength)+'supervised='+str(rate)+'_N='+str(processNumber)+'.npy',AttrisArrayPaired)
    #unpaired
    np.savez(save_dir+'dataset3_255img32x32_mask_UnitLen_'+str(unitLatentLength)+'supervised='+str(rate)+'_N='+str(processNumber)+'.npz',images=imgArray3,masks=mask3Array)
    np.savez(save_dir+'dataset4_255img32x32_mask_UnitLen_'+str(unitLatentLength)+'supervised='+str(rate)+'_N='+str(processNumber)+'.npz',images=imgArray4,masks=mask4Array)
    #np.save('npz_DSD_dataset/AttrisArray34_255img32x32_UnitLen_'+str(unitLatentLength)+'supervised='+str(rate)+'_N='+str(processNumber)+'.npy',AttrisArrayunPaired)




