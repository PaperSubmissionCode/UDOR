import numpy as np
import random

# load 20000 samples 20000*32*32

for i in range(1,2,1):
    max_offset = i
    MnistRandom012ArrayNorm0_1 = np.load(
        'multiMnistOffsetPositionDataset_32x32/MnistOffset{}_PositionRandom01ArrayNorm0_1.npz'.format(max_offset))  #
    imgs = MnistRandom012ArrayNorm0_1['imgs']
    w = 32
    h = 32
    ch = 1
    Nend = 10000
    Nstart = 0
    unitLength = 1  # can change
    partNum = 2

    N_p = Nend - Nstart
    # ============get all images===============
    imgArray = np.empty((N_p, w, h, ch))

    for k in range(Nstart, Nend):
        i = k - Nstart
        imgfloat = np.asarray(imgs[k], 'f')
        imgfloat = np.reshape(imgfloat, (w, h, 1))
        imgArray[i, :, :, :] = imgfloat
    # ==============get mask==============
    maskArray = np.empty((N_p, unitLength * partNum))
    mask1 = np.ones((unitLength * partNum))
    mask2 = np.ones((unitLength * partNum))
    for k in range(unitLength, 2 * unitLength):  # 3:6
        mask1[k - unitLength] = 0
        mask2[k] = 0
    print(mask1)  # [0. 0. 0. 1. 1. 1.]
    print(mask2)  # [1. 1. 1. 0. 0. 0.]
    numArray = np.arange(N_p)
    random.shuffle(numArray)  # (0,10000)
    for k in range(0, N_p):  # 0:10000
        if numArray[k] > N_p / partNum:
            maskArray[k] = mask2
        else:
            maskArray[k] = mask1

    # reset  mask value 0~64 (in the training stage)
    for i in range(0, 8):
        for j in range(0, 8):
            if j % 4 == 0:
                maskTemp = np.ones((unitLength * partNum))
            elif j % 4 == 1:
                maskTemp = mask1
            elif j % 4 == 2:
                maskTemp = mask2
            else:
                maskTemp = np.zeros((unitLength * partNum))

            maskArray[i * 8 + j] = maskTemp
    # reset imgs
    MnistRandom012Array_norm_test64 = np.load(
        "multiMnistOffsetPositionDataset_32x32/MnistOffset{}_PositionRandom012Array_norm_test64.npz".format(
            max_offset))  #
    imgs = MnistRandom012Array_norm_test64['imgs']
    for k in range(0, 64):
        imgfloat = np.asarray(imgs[k], 'f')
        imgfloat = np.reshape(imgfloat, (w, h, 1))
        imgArray[k, :, :, :] = imgfloat

    np.savez('./npz_datas/unitLength{}_mnistOffset{}_PositionMultiAndMask_'.format(unitLength, max_offset) + str(
        Nend - Nstart) + 'x' + str(w) + 'x' + str(w) + 'x' + str(ch) + '.npz', images=imgArray, masks=maskArray)

    # ========================for test============================
    maskArray = np.empty((64, unitLength * partNum))
    # reset  mask value 0~64 (in the training stage)
    for i in range(0, 8):
        for j in range(0, 8):
            if j % 4 == 0:
                maskTemp = np.ones((unitLength * partNum))
            elif j % 4 == 1:
                maskTemp = mask1
            elif j % 4 == 2:
                maskTemp = mask2
            else:
                maskTemp = np.zeros((unitLength * partNum))

            maskArray[i * 8 + j] = maskTemp
    # reset imgs
    imgArray = np.empty((64, w, h, ch))
    # 让训练集的第一张大图都是三个数字的
    MnistRandom012Array_norm_test64 = np.load(
        "multiMnistOffsetPositionDataset_32x32/MnistOffset{}_PositionRandom012Array_norm_test64.npz".format(max_offset))
    imgs = MnistRandom012Array_norm_test64['imgs']
    for k in range(0, 64):
        imgfloat = np.asarray(imgs[k], 'f')
        imgfloat = np.reshape(imgfloat, (w, h, 1))
        imgArray[k, :, :, :] = imgfloat

    np.savez('./npz_datas/unitLength{}_mnistOffset{}_PositionMultiAndMask_64x'.format(unitLength, max_offset) + str(
        w) + 'x' + str(w) + 'x' + str(ch) + '_valid.npz', images=imgArray, masks=maskArray)


"""
stage 3: test
"""
# # np.squeeze
# Coco3animal_012 = np.load('./npz_datas/unitLength3_mnistPositionMultiAndMask_10000x32x32x1.npz')
# dogsArray_zero = Coco3animal_012['images']
# print(dogsArray_zero.shape)
# plt.subplot(1,2,1)
# plt.imshow(dogsArray_zero[0].squeeze(), 'gray')
# plt.subplot(1,2,2)
# plt.imshow(dogsArray_zero[1].squeeze(), 'gray')
# plt.axis('off')
# plt.show()
