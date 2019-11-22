import numpy as np

img_gt_path='../../npz_DSD_dataset/Arr1_mnistImgsForDSD_GTimgs_mask_GTlabel_(47x64)x32x32x1_unitLength1_dataset.npz'
img_pre_path='../../npz_DSD_dataset/codes/codes_DSD_codesAndImgForMetricsCal.npz'

img_gt=np.load(img_gt_path)['imagesGT']

img_pre=np.load(img_pre_path)['imagesNorm0_1']
spot=np.where(np.isinf(img_gt))[0]

# DSD already is 0-255
img_gt=np.delete(img_gt,spot,axis=0)
img_pre=np.delete(img_pre,spot,axis=0)

img_gt=img_gt[0:3000]
img_pre=img_pre[0:3000]

metri=abs(img_gt-img_pre).mean()
print("visual metrics:{}".format(metri))