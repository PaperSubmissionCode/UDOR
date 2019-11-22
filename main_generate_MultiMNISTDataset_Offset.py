import numpy as np
import random
import os
# '''
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
# print(mnist)
# print(mnist.train.labels[0:10])
# print(mnist.train.images[0].size)
# im = mnist.train.images[7].reshape(28, 28)
# im1_2=imresize(im ,0.5)
#
# plt.subplot(1, 2, 1)
# plt.imshow(im,'gray')
# plt.subplot(1, 2, 2)
# plt.imshow(im1_2,'gray')
# plt.axis('off')
# plt.show()
#
# Num=len(mnist.train.labels)
#
# zeroArray=np.empty((Num/5,14,14))
# oneArray=np.empty((Num/5,14,14))
# twoArray=np.empty((Num/5,14,14))
# zeroCount=0
# oneCount=0
# twoCount=0
# for k in range(0,Num):
#     if mnist.train.labels[k]==0:
#         im = mnist.train.images[k].reshape(28, 28)
#         im1_2=imresize(im ,0.5)
#         zeroArray[zeroCount,:,:]=im1_2
#         zeroCount=zeroCount+1
#     elif mnist.train.labels[k]==1:
#         im = mnist.train.images[k].reshape(28, 28)
#         im1_2=imresize(im ,0.5)
#         oneArray[oneCount,:,:]=im1_2
#         oneCount=oneCount+1
#     elif mnist.train.labels[k]==2:
#         im = mnist.train.images[k].reshape(28, 28)
#         im1_2=imresize(im ,0.5)
#         twoArray[twoCount,:,:]=im1_21
#         twoCount=twoCount+
#     else:
#         pass
#
#     if k%1000==0:
#         print("porcessing..."+str(k))
#
# print(zeroCount) #5444
# print(oneCount)  #6179
# print(twoCount)  #5470
# np.savez('npz_datas/Mnist012.npz',zeroArray=zeroArray,oneArray=oneArray,twoArray=twoArray)
# '''
# #================================================================
#
Mnist012=np.load('./npz_datas/Mnist012.npz')
zeroArray=Mnist012['zeroArray']
oneArray=Mnist012['oneArray']
# twoArray=Mnist012['twoArray']
'''
plt.subplot(1, 2, 1)
plt.imshow(zeroArray[0],'gray')
plt.subplot(1, 2, 2)
plt.imshow(twoArray[1],'gray')
plt.axis('off')
plt.show()
'''
#=================================================================

# directoryName = 'outputs'
# directoryName = 'multiMnistPositionDataset_32x32'
# shutil.rmtree(directoryName)
# os.mkdir(directoryName)

processNumber = 20000
width = 32
height = 32
max_offset=1

MnistRandom012Array=np.zeros((processNumber,width,height))
'''
zeroMax=5444
oneMax=6179
twoMax=5470
'''
maxValue=5400
data_dir='./multiMnistOffsetPositionDataset_32x32/'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
f_record=open(data_dir+'MnistOffset{}_PositionRandom01Array_record.txt'.format(max_offset),'a+')
for i in range(processNumber):

    modeNum= random.randint(0, 3)
    if modeNum==0: # no digit encoded by 00
        f_record.write('0,0,0,0') # 00 + location (x,y,height,width)
        f_record.write("\n")
        pass
    elif modeNum==1:
        # 0 digit encoded by 01
        center_x=8
        center_y=8 # center point
        # orientation
        """8 ori: 
        0:up, 1:bottom, 2:left, 3:right, 4:top-left, 5:top-right, 6:bottom-left, 7:bottom-right
        """
        f_record.write('0,1,')
        ori=random.randint(0,7)
        if ori==0: # up
            offset_up=random.randint(0,1)
            offset_center_x=center_x-offset_up
            offset_center_y=center_y
            MnistRandom012Array[i, offset_center_x-7:offset_center_x + 7,offset_center_y-7:offset_center_y + 7 ] = zeroArray[random.randint(0, maxValue)]
            f_record.write('0,{},{},{},{}'.format(offset_center_x-7,offset_center_x + 7,offset_center_y-7,offset_center_y + 7 ))
        elif ori==1: # bottom
            offset_bottom=random.randint(0,max_offset)
            offset_center_x = center_x + offset_bottom
            offset_center_y = center_y
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = zeroArray[random.randint(0, maxValue)]
            f_record.write('1,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori==2: # left
            offset_left=random.randint(0,1)
            offset_center_x = center_x
            offset_center_y = center_y-offset_left
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = zeroArray[random.randint(0, maxValue)]
            f_record.write('2,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 3:  # right
            offset_right = random.randint(0, max_offset)
            offset_center_x = center_x
            offset_center_y = center_y + offset_right
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
            f_record.write('3,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 4:  # top-left
            offset_top=random.randint(1,1)
            offset_left = random.randint(1,1)
            offset_center_x = center_x-offset_top
            offset_center_y = center_y - offset_left
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
            f_record.write('4,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 5:  # top-right
            offset_top=random.randint(1,1)
            offset_right = random.randint(1,max_offset)
            offset_center_x = center_x-offset_top
            offset_center_y = center_y + offset_right
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
            f_record.write('5,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 6:  # bottom-left
            offset_bottom=random.randint(1,max_offset)
            offset_left = random.randint(1,1)
            offset_center_x = center_x+offset_bottom
            offset_center_y = center_y +offset_left
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
            f_record.write('6,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 7:  # bottom-right
            offset_bottom=random.randint(1,max_offset)
            offset_right = random.randint(1,max_offset)
            offset_center_x = center_x+offset_bottom
            offset_center_y = center_y + offset_right
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
            f_record.write('7,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        f_record.write("\n")
    elif modeNum==2:
        # 1 digit encoded by 01
        center_x = 24
        center_y = 24  # center point
        # orientation
        """8 ori: 
        0:up, 1:bottom, 2:left, 3:right, 4:top-left, 5:top-right, 6:bottom-left, 7:bottom-right
        """
        f_record.write('1,0,')
        ori = random.randint(0, 7)
        if ori == 0:  # up
            offset_up = random.randint(0, max_offset)
            offset_center_x = center_x - offset_up
            offset_center_y = center_y
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            oneArray[random.randint(0, maxValue)]
            f_record.write('0,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori==1: # bottom
            offset_bottom=random.randint(0,1)
            offset_center_x = center_x + offset_bottom
            offset_center_y = center_y
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = oneArray[random.randint(0, maxValue)]
            f_record.write('1,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori==2: # left
            offset_left=random.randint(0,max_offset)
            offset_center_x = center_x
            offset_center_y = center_y-offset_left
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = oneArray[random.randint(0, maxValue)]
            f_record.write('2,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 3:  # right
            offset_right = random.randint(0, 1)
            offset_center_x = center_x
            offset_center_y = center_y + offset_right
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            oneArray[random.randint(0, maxValue)]
            f_record.write('3,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 4:  # top-left
            offset_top=random.randint(1,max_offset)
            offset_left = random.randint(1,max_offset)
            offset_center_x = center_x-offset_top
            offset_center_y = center_y - offset_left
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
            oneArray[random.randint(0, maxValue)]
            f_record.write('4,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 5:  # top-right
            offset_top=random.randint(1,max_offset)
            offset_right = random.randint(1,1)
            offset_center_x = center_x-offset_top
            offset_center_y = center_y + offset_right
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write('5,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 6:  # bottom-left
            offset_bottom=random.randint(1,1)
            offset_left = random.randint(1,max_offset)
            offset_center_x = center_x+offset_bottom
            # print(offset_center_x)
            offset_center_y = center_y -offset_left
            # print(offset_center_y)
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write('6,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        elif ori == 7:  # bottom-right
            offset_bottom=random.randint(1,1)
            offset_right = random.randint(1,1)
            offset_center_x = center_x+offset_bottom
            offset_center_y = center_y + offset_right
            MnistRandom012Array[i, offset_center_x - 7:offset_center_x + 7, offset_center_y - 7:offset_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write('7,{},{},{},{}'.format(offset_center_x - 7, offset_center_x + 7, offset_center_y - 7,
                                                  offset_center_y + 7))
        f_record.write("\n")


    else:
        # 0,1 digit
        zero_center_x = 8
        zero_center_y = 8  # zero center point
        one_center_x = 24
        one_center_y = 24  # one center point
        """"zero position first and one position last"""

        # =========================== zero orientation
        """8 ori:
        0:up, 1:bottom, 2:left, 3:right, 4:up-left, 5:up-right, 6:bottom-left, 7:bottom-right
        """
        f_record.write('1,1,')
        zero_ori = random.randint(0, 7)
        if zero_ori==0: # up
            zero_offset_up = random.randint(0, 1)
            offset_zero_center_x = zero_center_x - zero_offset_up
            offset_zero_center_y = zero_center_y
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7, offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write('0,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                                  offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori==0: # up
                one_offset_up = random.randint(0, max_offset)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==1: # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==2: # left
                one_offset_left = random.randint(0, max_offset)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y-one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==3: # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y+one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==4: # up-left
                one_offset_up=random.randint(0,max_offset)
                if one_offset_up<=2+zero_offset_up:
                    one_offset_left = random.randint(0, max_offset)
                else:
                    one_offset_left = random.randint(0, min(2,max_offset))
                offset_one_center_x = one_center_x-one_offset_up
                offset_one_center_y = one_center_y-one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==5: # up-right
                one_offset_up=random.randint(0,max_offset)
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x-one_offset_up
                offset_one_center_y = one_center_y+one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==6: # bottom-left
                one_offset_bottom=random.randint(0,1)
                one_offset_left = random.randint(0,max_offset)
                offset_one_center_x = one_center_x+one_offset_bottom
                offset_one_center_y = one_center_y-one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            elif one_ori==7: # bottom-right
                one_offset_bottom=random.randint(0,1)
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x+one_offset_bottom
                offset_one_center_y = one_center_y+one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                           offset_one_center_y + 7))
            f_record.write("\n")

        elif zero_ori ==1: # bottom
            zero_offset_bottom = random.randint(0, max_offset)
            offset_zero_center_x = zero_center_x + zero_offset_bottom
            offset_zero_center_y = zero_center_y
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '1,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori == 0:  # up
                one_offset_up = random.randint(0, max_offset)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 1:  # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 2:  # left
                if zero_offset_bottom <=2:
                    one_offset_left=random.randint(0, max_offset)
                else:
                    one_offset_left = random.randint(0, min(max_offset,2))
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 3:  # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 4:  # up-left
                # if zero_offset_bottom<=2:
                #     one_offset_left=random.randint(0,max_offset)
                #     one_offset_up=random.randint(0,1-zero_offset_bottom)
                one_offset_up = random.randint(1, max_offset)
                one_offset_left = random.randint(1, min(2, max_offset))
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 5:  # up-right
                one_offset_up = random.randint(1, max_offset)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 6:  # bottom-left
                one_offset_bottom = random.randint(1, 1)
                if zero_offset_bottom<=2:
                    one_offset_left = random.randint(1, max_offset)
                else:
                    one_offset_left = random.randint(1, min(2,max_offset))

                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 7:  # bottom-right
                one_offset_bottom = random.randint(1, 1)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            f_record.write("\n")
        elif zero_ori==2: #left
            zero_offset_left = random.randint(0,1)
            offset_zero_center_x = zero_center_x
            offset_zero_center_y = zero_center_y-zero_offset_left
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '2,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori == 0:  # up
                one_offset_up = random.randint(0, max_offset)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 1:  # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 2:  # left
                one_offset_left = random.randint(0, max_offset)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 3:  # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 4:  # up-left
                a=random.randint(0,1)
                if a ==0:
                    one_offset_up = random.randint(1,  min(2, max_offset))
                    one_offset_left = random.randint(1, max_offset)
                else:
                    one_offset_up = random.randint(1, max_offset)
                    one_offset_left = random.randint(1, min(2,max_offset))
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 5:  # up-right
                one_offset_up = random.randint(1, max_offset)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 6:  # bottom-left
                one_offset_bottom = random.randint(1, 1)
                one_offset_left = random.randint(1, max_offset)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 7:  # bottom-right
                one_offset_bottom = random.randint(1, 1)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            f_record.write("\n")
        elif zero_ori==3: # right
            zero_offset_right = random.randint(0, max_offset)
            offset_zero_center_x = zero_center_x
            offset_zero_center_y = zero_center_y + zero_offset_right
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '3,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori == 0:  # up
                if zero_offset_right<=2:
                    one_offset_up = random.randint(0, max_offset)
                else:
                    one_offset_up = random.randint(0, min(2,max_offset))
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 1:  # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 2:  # left
                one_offset_left = random.randint(0, max_offset)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 3:  # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 4:  # up-left
                one_offset_up = random.randint(1, min(2, max_offset))
                one_offset_left = random.randint(1, max_offset)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 5:  # up-right
                if zero_offset_right<=2:
                    one_offset_up = random.randint(1, max_offset)
                else:
                    one_offset_up = random.randint(1, min(2,max_offset))
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 6:  # bottom-left
                one_offset_bottom = random.randint(1, 1)
                one_offset_left = random.randint(1, max_offset)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 7:  # bottom-right
                one_offset_bottom = random.randint(1, 1)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            f_record.write("\n")

        elif zero_ori==4: # up-left
            zero_offset_up=random.randint(1,1)
            zero_offset_left = random.randint(1,1)
            offset_zero_center_x = zero_center_x - zero_offset_up
            offset_zero_center_y = zero_center_y - zero_offset_left
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '4,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori == 0:  # up
                one_offset_up = random.randint(0, max_offset)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 1:  # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 2:  # left
                one_offset_left = random.randint(0, max_offset)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 3:  # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 4:  # up-left
                a=random.randint(0,1)
                if a==0:
                    one_offset_up = random.randint(1, min(2+zero_offset_up, max_offset))
                    one_offset_left = random.randint(1, max_offset)
                else:
                    one_offset_up = random.randint(1, max_offset)
                    one_offset_left = random.randint(1, min(2+zero_offset_left, max_offset))
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 5:  # up-right
                one_offset_up = random.randint(1, max_offset)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 6:  # bottom-left
                one_offset_bottom = random.randint(1, 1)
                one_offset_left = random.randint(1, max_offset)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 7:  # bottom-right
                one_offset_bottom = random.randint(1, 1)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            f_record.write("\n")

        elif zero_ori == 5:  # up-right
            zero_offset_up = random.randint(1, 1)
            zero_offset_right = random.randint(1, max_offset)
            offset_zero_center_x = zero_center_x - zero_offset_up
            offset_zero_center_y = zero_center_y + zero_offset_right
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '5,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori == 0:  # up
                if zero_offset_right<=2:
                    one_offset_up = random.randint(0, max_offset)
                else:
                    one_offset_up = random.randint(0, min(2+zero_offset_up,max_offset))
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 1:  # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 2:  # left
                one_offset_left = random.randint(0, max_offset)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 3:  # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 4:  # up-left
                one_offset_up = random.randint(1, min(2 + zero_offset_up, max_offset))
                one_offset_left = random.randint(1, max_offset)

                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 5:  # up-right
                one_offset_up = random.randint(1, min(2 + zero_offset_up, max_offset))
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 6:  # bottom-left
                one_offset_bottom = random.randint(1, 1)
                one_offset_left = random.randint(1, max_offset)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 7:  # bottom-right
                one_offset_bottom = random.randint(1, 1)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            f_record.write("\n")

        elif zero_ori == 6:  # bottom-left
            zero_offset_bottom = random.randint(1, max_offset)
            zero_offset_left = random.randint(1, 1)
            offset_zero_center_x = zero_center_x + zero_offset_bottom
            offset_zero_center_y = zero_center_y - zero_offset_left
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '6,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # ================================= one orientation
            one_ori = random.randint(0, 7)
            if one_ori == 0:  # up
                one_offset_up = random.randint(0,max_offset)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 1:  # bottom
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 2:  # left
                if zero_offset_bottom<=2:
                    one_offset_left = random.randint(0, max_offset)
                else:
                    one_offset_left = random.randint(0, min(2+zero_offset_left,max_offset))
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 3:  # right
                one_offset_right = random.randint(0, 1)
                offset_one_center_x = one_center_x
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 4:  # up-left
                one_offset_up = random.randint(1, max_offset)
                one_offset_left = random.randint(1, min(2 + zero_offset_left, max_offset))

                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 5:  # up-right
                one_offset_up = random.randint(1, max_offset)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x - one_offset_up
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 6:  # bottom-left
                one_offset_bottom = random.randint(1, 1)
                if zero_offset_bottom<=2:
                    one_offset_left = random.randint(1, max_offset)
                else:
                    one_offset_left = random.randint(1, min(2,max_offset))
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y - one_offset_left
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif one_ori == 7:  # bottom-right
                one_offset_bottom = random.randint(1, 1)
                one_offset_right = random.randint(1, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
                offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            f_record.write("\n")

        elif zero_ori == 7:  # bottom-right
            if max_offset>=16:
                swap_if=random.randint(0,1)
                if swap_if==1:
                    temp=zeroArray
                    zeroArray=oneArray
                    oneArray=temp
            a=random.randint(0,1)
            if a==0:
                zero_offset_bottom = random.randint(1,1)
                zero_offset_right = random.randint(1,max_offset)
                offset_zero_center_x = zero_center_x + zero_offset_bottom
                offset_zero_center_y = zero_center_y + zero_offset_right
                MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
                offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                    zeroArray[random.randint(0, maxValue)]
                f_record.write(
                    '7,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                           offset_zero_center_y + 7))
                # one digit =======================
                if random.randint(0,1)==0:
                    one_offset_up = random.randint(0, 1)
                    offset_one_center_x = one_center_x - one_offset_up
                else:
                    one_offset_bottom = random.randint(0, 1)
                    offset_one_center_x = one_center_x + one_offset_bottom
                if random.randint(0,1)==0:
                    one_offset_left = random.randint(0, max_offset)

                    offset_one_center_y = one_center_y - one_offset_left
                else:
                    one_offset_right = random.randint(0, 1)
                    offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))
            elif a==1:
                zero_offset_bottom = random.randint(1, max_offset)
                zero_offset_right = random.randint(1, 1)
                offset_zero_center_x = zero_center_x + zero_offset_bottom
                offset_zero_center_y = zero_center_y + zero_offset_right
                MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
                offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                    zeroArray[random.randint(0, maxValue)]
                f_record.write(
                    '7,{},{},{},{}'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                           offset_zero_center_y + 7))
                # one digit =======================
                if random.randint(0, 1) == 0:
                    one_offset_up = random.randint(0, max_offset)
                    offset_one_center_x = one_center_x - one_offset_up
                else:
                    one_offset_bottom = random.randint(0, 1)
                    offset_one_center_x = one_center_x + one_offset_bottom
                if random.randint(0, 1) == 0:
                    one_offset_left = random.randint(0, 1)
                    offset_one_center_y = one_center_y - one_offset_left
                else:
                    one_offset_right = random.randint(0, 1)
                    offset_one_center_y = one_center_y + one_offset_right
                MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
                offset_one_center_y - 7:offset_one_center_y + 7] = \
                    oneArray[random.randint(0, maxValue)]
                f_record.write(
                    '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                         offset_one_center_y + 7))

            f_record.write("\n")

np.savez(data_dir+'MnistOffset{}_PositionRandom01Array.npz'.format(max_offset),imgs=MnistRandom012Array)
imgs=MnistRandom012Array/255.0
np.savez(data_dir+'MnistOffset{}_PositionRandom01ArrayNorm0_1.npz'.format(max_offset),imgs=imgs)
f_record.close()


"""----------------------test---------------------"""
# Mnist012=np.load('multiMnistPositionDataset_32x32/MnistPositionRandom01ArrayNorm0_1.npz')
# Array=Mnist012['imgs']
#
# # for i in range(20):
# #     plt.subplot(1, 2, 1)
# #     plt.imshow(Array[i], 'gray')
# #     plt.subplot(1, 2, 2)
# #     plt.imshow(Array[2*i], 'gray')
# #     plt.axis('off')
# #     plt.show()
# print(Array.shape)
# plt.subplot(1, 2, 1)
# plt.imshow(Array[49], 'gray')
# plt.subplot(1, 2, 2)
# plt.imshow(Array[2 ], 'gray')
# plt.axis('off')
# plt.show()
