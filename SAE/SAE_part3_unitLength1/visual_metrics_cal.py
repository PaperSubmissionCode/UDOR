import numpy as np

img_gt_path='../../npz_datas/mnistImgs_GTimages_mask_GTlabel_(47x64)x32x32x1_unitLength1_CodeImageDataset_forSAE.npz'
img_pre_path='../../npz_datas/codes/codes_SAE_codesAndImgForMetricsCal.npz'

img_gt=np.load(img_gt_path)['imagesGT']

img_pre=np.load(img_pre_path)['imagesNorm0_1']
spot=np.where(np.isinf(img_gt))[0]

img_gt=np.delete(img_gt,spot,axis=0)*255
img_pre=np.delete(img_pre,spot,axis=0)*255

img_gt=img_gt[0:3000]
img_pre=img_pre[0:3000]

metri=abs(img_gt-img_pre).mean()
print("visual metrics:{}".format(metri))