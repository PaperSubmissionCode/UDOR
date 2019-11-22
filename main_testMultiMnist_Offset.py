import numpy as np
import random

# ================================================================
Mnist012 = np.load('npz_datas/Mnist012.npz')
zeroArray = Mnist012['zeroArray']
oneArray = Mnist012['oneArray']
twoArray = Mnist012['twoArray']

# directoryName = 'outputs'
# directoryName = 'multiMnistDataset_32x32'

processNumber = 64
width = 32
height = 32
max_offset=1

MnistRandom012Array = np.zeros((processNumber, width, height))
'''
zeroMax=5444
oneMax=6179
twoMax=5470
'''
maxValue = 5400
f_record=open('multiMnistOffsetPositionDataset_32x32/MnistOffset{}_PositionRandom012Array_norm_test64_record.txt'.format(max_offset),'a+')
for i in range(processNumber):
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
    if zero_ori == 0:  # up
        zero_offset_up = random.randint(0, 1)
        offset_zero_center_x = zero_center_x - zero_offset_up
        offset_zero_center_y = zero_center_y
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
        f_record.write(
            '0,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
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
            one_offset_up = random.randint(0, max_offset)
            if one_offset_up <= 2 + zero_offset_up:
                one_offset_left = random.randint(0, max_offset)
            else:
                one_offset_left = random.randint(0, min(2, max_offset))
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write(
                '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                     offset_one_center_y + 7))
        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(0, max_offset)
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write(
                '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                     offset_one_center_y + 7))
        elif one_ori == 6:  # bottom-left
            one_offset_bottom = random.randint(0, 1)
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write(
                '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                     offset_one_center_y + 7))
        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(0, 1)
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray[random.randint(0, maxValue)]
            f_record.write(
                '{},{},{},{}'.format(offset_one_center_x - 7, offset_one_center_x + 7, offset_one_center_y - 7,
                                     offset_one_center_y + 7))
        f_record.write("\n")

    elif zero_ori == 1:  # bottom
        zero_offset_bottom = random.randint(0, max_offset)
        offset_zero_center_x = zero_center_x + zero_offset_bottom
        offset_zero_center_y = zero_center_y
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
        f_record.write(
            '1,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
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
            if zero_offset_bottom <= 2:
                one_offset_left = random.randint(0, max_offset)
            else:
                one_offset_left = random.randint(0, min(max_offset, 2))
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
            if zero_offset_bottom <= 2:
                one_offset_left = random.randint(1, max_offset)
            else:
                one_offset_left = random.randint(1, min(2, max_offset))

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
    elif zero_ori == 2:  # left
        zero_offset_left = random.randint(0, 1)
        offset_zero_center_x = zero_center_x
        offset_zero_center_y = zero_center_y - zero_offset_left
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
        f_record.write(
            '2,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
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
            a = random.randint(0, 1)
            if a == 0:
                one_offset_up = random.randint(1, min(2, max_offset))
                one_offset_left = random.randint(1, max_offset)
            else:
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
    elif zero_ori == 3:  # right
        zero_offset_right = random.randint(0, max_offset)
        offset_zero_center_x = zero_center_x
        offset_zero_center_y = zero_center_y + zero_offset_right
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
        f_record.write(
            '3,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                   offset_zero_center_y + 7))
        # ================================= one orientation
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            if zero_offset_right <= 2:
                one_offset_up = random.randint(0, max_offset)
            else:
                one_offset_up = random.randint(0, min(2, max_offset))
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
            if zero_offset_right <= 2:
                one_offset_up = random.randint(1, max_offset)
            else:
                one_offset_up = random.randint(1, min(2, max_offset))
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

    elif zero_ori == 4:  # up-left
        zero_offset_up = random.randint(1, 1)
        zero_offset_left = random.randint(1, 1)
        offset_zero_center_x = zero_center_x - zero_offset_up
        offset_zero_center_y = zero_center_y - zero_offset_left
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray[random.randint(0, maxValue)]
        f_record.write(
            '4,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
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
            a = random.randint(0, 1)
            if a == 0:
                one_offset_up = random.randint(1, min(2 + zero_offset_up, max_offset))
                one_offset_left = random.randint(1, max_offset)
            else:
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
            '5,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                   offset_zero_center_y + 7))
        # ================================= one orientation
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            if zero_offset_right <= 2:
                one_offset_up = random.randint(0, max_offset)
            else:
                one_offset_up = random.randint(0, min(2 + zero_offset_up, max_offset))
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
            '6,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
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
            if zero_offset_bottom <= 2:
                one_offset_left = random.randint(0, max_offset)
            else:
                one_offset_left = random.randint(0, min(2 + zero_offset_left, max_offset))
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
            if zero_offset_bottom <= 2:
                one_offset_left = random.randint(1, max_offset)
            else:
                one_offset_left = random.randint(1, min(2, max_offset))
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
        if max_offset >= 16:
            swap_if = random.randint(0, 1)
            if swap_if == 1:
                temp = zeroArray
                zeroArray = oneArray
                oneArray = temp
        a = random.randint(0, 1)
        if a == 0:
            zero_offset_bottom = random.randint(1, 1)
            zero_offset_right = random.randint(1, max_offset)
            offset_zero_center_x = zero_center_x + zero_offset_bottom
            offset_zero_center_y = zero_center_y + zero_offset_right
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '7,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
                                       offset_zero_center_y + 7))
            # one digit =======================
            if random.randint(0, 1) == 0:
                one_offset_up = random.randint(0, 1)
                offset_one_center_x = one_center_x - one_offset_up
            else:
                one_offset_bottom = random.randint(0, 1)
                offset_one_center_x = one_center_x + one_offset_bottom
            if random.randint(0, 1) == 0:
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
        elif a == 1:
            zero_offset_bottom = random.randint(1, max_offset)
            zero_offset_right = random.randint(1, 1)
            offset_zero_center_x = zero_center_x + zero_offset_bottom
            offset_zero_center_y = zero_center_y + zero_offset_right
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray[random.randint(0, maxValue)]
            f_record.write(
                '7,{},{},{},{},'.format(offset_zero_center_x - 7, offset_zero_center_x + 7, offset_zero_center_y - 7,
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
f_record.close()
MnistRandom012Array = MnistRandom012Array / 255.0
np.savez('multiMnistOffsetPositionDataset_32x32/MnistOffset{}_PositionRandom012Array_norm_test64.npz'.format(max_offset), imgs=MnistRandom012Array)

# import cv2
# Mnist012=np.load('multiMnistPositionDataset_32x32/MnistPositionRandom012Array_norm_test64.npz')
# Array=Mnist012['imgs']
#
# for i in range(16):
#     cv2.imshow('tt',Array[i])
#     cv2.waitKey()