import numpy as np
import random
import os

zero_center_x = 8
zero_center_y = 8  # zero center point
one_center_x = 24
one_center_y = 24  # one center point

def draw_ossfet_mnist(i,MnistRandom012Array,zeroArray,oneArray,max_offset,set_0):
    """

    :param i:
    :param MnistRandom012Array:
    :param zeroArray:
    :param oneArray:
    :param set_0:
    :return: if set_0=0:offset_zero_center_x,offset_zero_center_y,
            else if set_o=1:offset_one_center_x,offset_one_center_y
    """
    zero_ori = random.randint(0, 7)
    if zero_ori == 0:  # up
        zero_offset_up = random.randint(0, 1)
        offset_zero_center_x = zero_center_x - zero_offset_up
        offset_zero_center_y = zero_center_y
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

        # ================================= one orientation
        # offset_one_center_x_return=0
        # offset_one_center_y_return=0
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            one_offset_up = random.randint(0, max_offset)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y
            # offset_one_center_x_return=offset_one_center_x
            # offset_one_center_y_return=offset_one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            offset_one_center_x_return = offset_one_center_x
            offset_one_center_y_return = offset_one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(0, max_offset)
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 6:  # bottom-left
            one_offset_bottom = random.randint(0, 1)
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(0, 1)
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y

    elif zero_ori == 1:  # bottom
        zero_offset_bottom = random.randint(0, max_offset)
        offset_zero_center_x = zero_center_x + zero_offset_bottom
        offset_zero_center_y = zero_center_y
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

        # ================================= one orientation
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            one_offset_up = random.randint(0, max_offset)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            if zero_offset_bottom <= 2:
                one_offset_left = random.randint(0, max_offset)
            else:
                one_offset_left = random.randint(0, min(max_offset, 2))
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(1, max_offset)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(1, 1)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y


    elif zero_ori == 2:  # left
        zero_offset_left = random.randint(0, 1)
        offset_zero_center_x = zero_center_x
        offset_zero_center_y = zero_center_y - zero_offset_left
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

        # ================================= one orientation
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            one_offset_up = random.randint(0, max_offset)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(1, max_offset)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 6:  # bottom-left
            one_offset_bottom = random.randint(1, 1)
            one_offset_left = random.randint(1, max_offset)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(1, 1)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y

    elif zero_ori == 3:  # right
        zero_offset_right = random.randint(0, max_offset)
        offset_zero_center_x = zero_center_x
        offset_zero_center_y = zero_center_y + zero_offset_right
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

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
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 4:  # up-left
            one_offset_up = random.randint(1, min(2, max_offset))
            one_offset_left = random.randint(1, max_offset)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 6:  # bottom-left
            one_offset_bottom = random.randint(1, 1)
            one_offset_left = random.randint(1, max_offset)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(1, 1)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y

    elif zero_ori == 4:  # up-left
        zero_offset_up = random.randint(1, 1)
        zero_offset_left = random.randint(1, 1)
        offset_zero_center_x = zero_center_x - zero_offset_up
        offset_zero_center_y = zero_center_y - zero_offset_left
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

        # ================================= one orientation
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            one_offset_up = random.randint(0, max_offset)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(1, max_offset)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 6:  # bottom-left
            one_offset_bottom = random.randint(1, 1)
            one_offset_left = random.randint(1, max_offset)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(1, 1)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y

    elif zero_ori == 5:  # up-right
        zero_offset_up = random.randint(1, 1)
        zero_offset_right = random.randint(1, max_offset)
        offset_zero_center_x = zero_center_x - zero_offset_up
        offset_zero_center_y = zero_center_y + zero_offset_right
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

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
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            one_offset_left = random.randint(0, max_offset)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 4:  # up-left
            one_offset_up = random.randint(1, min(2 + zero_offset_up, max_offset))
            one_offset_left = random.randint(1, max_offset)

            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(1, min(2 + zero_offset_up, max_offset))
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 6:  # bottom-left
            one_offset_bottom = random.randint(1, 1)
            one_offset_left = random.randint(1, max_offset)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(1, 1)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y

    elif zero_ori == 6:  # bottom-left
        zero_offset_bottom = random.randint(1, max_offset)
        zero_offset_left = random.randint(1, 1)
        offset_zero_center_x = zero_center_x + zero_offset_bottom
        offset_zero_center_y = zero_center_y - zero_offset_left
        MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = \
            zeroArray

        # ================================= one orientation
        one_ori = random.randint(0, 7)
        if one_ori == 0:  # up
            one_offset_up = random.randint(0, max_offset)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 1:  # bottom
            one_offset_bottom = random.randint(0, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 2:  # left
            if zero_offset_bottom <= 2:
                one_offset_left = random.randint(0, max_offset)
            else:
                one_offset_left = random.randint(0, min(2 + zero_offset_left, max_offset))
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 3:  # right
            one_offset_right = random.randint(0, 1)
            offset_one_center_x = one_center_x
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 4:  # up-left
            one_offset_up = random.randint(1, max_offset)
            one_offset_left = random.randint(1, min(2 + zero_offset_left, max_offset))

            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y - one_offset_left
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

        elif one_ori == 5:  # up-right
            one_offset_up = random.randint(1, max_offset)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x - one_offset_up
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray

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
                oneArray

        elif one_ori == 7:  # bottom-right
            one_offset_bottom = random.randint(1, 1)
            one_offset_right = random.randint(1, 1)
            offset_one_center_x = one_center_x + one_offset_bottom
            offset_one_center_y = one_center_y + one_offset_right
            MnistRandom012Array[i, offset_one_center_x - 7:offset_one_center_x + 7,
            offset_one_center_y - 7:offset_one_center_y + 7] = \
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y


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
                zeroArray

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
                oneArray

        elif a == 1:
            zero_offset_bottom = random.randint(1, max_offset)
            zero_offset_right = random.randint(1, 1)
            offset_zero_center_x = zero_center_x + zero_offset_bottom
            offset_zero_center_y = zero_center_y + zero_offset_right
            MnistRandom012Array[i, offset_zero_center_x - 7:offset_zero_center_x + 7,
            offset_zero_center_y - 7:offset_zero_center_y + 7] = \
                zeroArray

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
                oneArray
        if set_0==0:
            return offset_zero_center_x,offset_zero_center_y
        elif set_0==1:
            return offset_one_center_x,offset_one_center_y

# ================================================================
Mnist012 = np.load('npz_datas/Mnist012.npz')
zeroArray = Mnist012['zeroArray']
oneArray = Mnist012['oneArray']

# directoryName = 'outputs'
# directoryName = 'multiMnistDataset_32x32'
batchSizeNum=32
batchSize=64
processNumber = batchSizeNum*batchSize #2048
width = 32
height = 32
max_offset=1

imgArray=np.zeros((processNumber,width,height))
imgArrayResetGt=np.zeros((processNumber,width,height))
label01Array=-1*np.ones((processNumber,2)) # initial for rember 0 1
'''
zeroMax=5444
oneMax=6179
twoMax=5470
'''
maxValue = 5400


#============================= for fixed digit 0 =================================
zero_fixed=np.zeros((14,14))
one=np.zeros((14,14))
patchEmpty=np.zeros((14,14))
for k in range(0,10):
    zero_fixed=zeroArray[random.randint(0,maxValue)]
    for t in range(0,100):
        # label 0
        label01Array[k*100+t,0]=0
        # one / empty
        modeWho = random.randint(0, 1)
        if modeWho == 1:
            one = oneArray[random.randint(0, maxValue)]
            label01Array[k * 100 + t, 1] = 1
        else:
            one = patchEmpty
        # imgArray[k * 100 + t, 1:15, 1:15] = zero_fixed
        # imgArray[k * 100 + t, 1:15, 17:31] = one
        offset_zero_center_x, offset_zero_center_y=draw_ossfet_mnist(k*100+t,imgArray,zero_fixed,one,max_offset,set_0=0)
        # print(offset_zero_center_x, offset_zero_center_y)
        imgArrayResetGt[k * 100 + t] = imgArray[k * 100 + t]
        imgArrayResetGt[k * 100 + t, offset_zero_center_x - 7:offset_zero_center_x + 7,
        offset_zero_center_y - 7:offset_zero_center_y + 7] = patchEmpty  # remove digit zero

#============================= for fixed digit 1 =================================
zero=np.zeros((14,14))
one_fixed=np.zeros((14,14))
patchEmpty=np.zeros((14,14))
startIndex=1000
for k in range(0,10):
    one_fixed=oneArray[random.randint(0,maxValue)]
    for t in range(0,100):
        # label 1
        label01Array[startIndex + k * 100 + t, 1] = 1
        # zero / empty
        modeWho = random.randint(0, 1)
        if modeWho == 1:
            zero = zeroArray[random.randint(0, maxValue)]
            label01Array[startIndex + k * 100 + t, 0] = 0
        else:
            zero = patchEmpty
        # insert images

        offset_one_center_x, offset_one_center_y = draw_ossfet_mnist(startIndex + k * 100 + t, imgArray, zero, one_fixed,
                                                                       max_offset, set_0=1)
        # print(offset_zero_center_x, offset_zero_center_y)
        imgArrayResetGt[startIndex + k * 100 + t] = imgArray[startIndex + k * 100 + t]
        imgArrayResetGt[startIndex + k * 100 + t, offset_one_center_x - 7:offset_one_center_x + 7,
        offset_one_center_y - 7:offset_one_center_y + 7] = patchEmpty  # remove digit zero

#===================trasfor the (Num,w,h) into float (Num,w,h,ch)====================
ch=1
imgArrayNorm0_1= np.empty((processNumber,width,height,ch))
imgArrayResetGtNorm0_1=np.empty((processNumber,width,height,ch))
for k in range(processNumber):
    imgfloat=np.asarray(imgArray[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArrayNorm0_1[k,:,:,:]=imgfloat

    imgfloat=np.asarray(imgArrayResetGt[k],'f')
    imgfloat=np.reshape(imgfloat,(width,height,1))
    imgArrayResetGtNorm0_1[k,:,:,:]=imgfloat

# normalize to 0~1
imgArrayNorm0_1=imgArrayNorm0_1/255.0
imgArrayResetGtNorm0_1=imgArrayResetGtNorm0_1/255.0

#======******  get mask (mask is decided by visual the reconstructed result after train) *******========
# ===handle the left 8 (3008-3000) images ====
for t in range(2000,2048):
    imgArrayNorm0_1[t]=imgArrayNorm0_1[t-2000]
    imgArrayResetGtNorm0_1[t]=imgArrayResetGtNorm0_1[t-2000]
    label01Array[t]=label01Array[t-2000]


unitLength=1
partNum=2
maskArray=np.empty((processNumber,unitLength*partNum))
mask1=np.ones((unitLength*partNum))
mask2=np.ones((unitLength*partNum))

for k in range(unitLength,2*unitLength):
    mask1[k-unitLength]=0
    mask2[k]=0
for t in range(1000,2000):
    maskArray[t-1000]=mask1  # mask for digit 0
    maskArray[t]=mask2       # mask for digit 1

save_dir='npz_datas_single_test/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
np.savez(save_dir+'mnistOffset{}_Imgs_GTimages_mask_GTlabel_('.format(max_offset)+str(batchSizeNum)+'x'+str(batchSize)+')x'+str(width)+'x'+str(height)+'x'+str(ch)+'_unitLength'+str(unitLength)+'_CodeImageDataset.npz',images=imgArrayNorm0_1,imagesGT=imgArrayResetGtNorm0_1,masks=maskArray,labelsGT=label01Array)
