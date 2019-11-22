import numpy as np
np.set_printoptions(threshold = 1e6)
max_offset=1

img_gt_path='../../../npz_datas_single_test/mnistOffset{}_Imgs_GTimages_mask_GTlabel_(32x64)x32x32x1_unitLength1_CodeImageDataset.npz'.format(max_offset)
img_pre_path='../../../npz_datas_single_test/codes/codes_reconstrued_results_offset{}.npz'.format(max_offset)

img_gt=np.load(img_gt_path)['imagesGT']

img_pre=np.load(img_pre_path)['imagesNorm0_1']
spot=np.where(np.isinf(img_gt))[0]

img_gt=np.delete(img_gt,spot,axis=0)*255
img_pre=np.delete(img_pre,spot,axis=0)*255

metri=abs(img_gt-img_pre).mean()
print("visual quality:{}".format(metri))