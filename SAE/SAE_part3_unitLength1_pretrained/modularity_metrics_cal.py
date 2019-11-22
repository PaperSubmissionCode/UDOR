import numpy as np

codes_path='../../npz_datas/codes/codes_SAE_codesAndImgForMetricsCal.npz'
codes_arr=np.load(codes_path)['codes']
spot=np.where(np.isnan(codes_arr))[0]
codes_arr=np.delete(codes_arr,spot,axis=0)
print(codes_arr.shape)
# for 0
code_reprent_modular_0=codes_arr[0:1000]
mean_total=0
for i in range(10):
    data=code_reprent_modular_0[0+i*100:100+i*100]
    # print(data.shape)
    sum_mean=data.sum(axis=0)/data.size
    # print("it:{} sum_mean: {}".format(i,sum_mean))
    sub_ = abs(data - sum_mean)
    # print(sub_.shape)
    sum_sub_ = sub_.sum(axis=0)/sub_.shape[0]
    # print('it:{} sum_sub_: {} '.format(i,sum_sub_))
    mean_total=mean_total+sum_sub_
mean_total=mean_total/10
# print("Total sum_sub_mean for 2: {}".format(mean_total))
print('single mean for 0: {}'.format(mean_total.mean()))


# for 1
code_reprent_modular_1=codes_arr[1000:2000]
mean_total=0
for i in range(10):
    data=code_reprent_modular_1[0+i*100:100+i*100]
    # print(data.shape)
    sum_mean=data.sum(axis=0)/data.size
    # print("it:{} sum_mean: {}".format(i,sum_mean))
    sub_ = abs(data - sum_mean)
    # print(sub_.shape)
    sum_sub_ = sub_.sum(axis=0)/sub_.shape[0]
    # print('it:{} sum_sub_: {:.5f} ({})'.format(i,sum_sub_, sum_sub_))
    mean_total=mean_total+sum_sub_
mean_total=mean_total/10
# print("Total sum_sub_mean for 1: {}".format(mean_total))
print('single mean for 1: {}'.format(mean_total.mean()))


# for 2
code_reprent_modular_2=codes_arr[2000:3000]
mean_total=0
for i in range(10):
    data=code_reprent_modular_2[0+i*100:100+i*100]
    # print(data.shape)
    sum_mean=data.sum(axis=0)/data.size
    # print("it:{} sum_mean: {}".format(i,sum_mean))
    sub_ = abs(data - sum_mean)
    # print(sub_.shape)
    sum_sub_ = sub_.sum(axis=0)/sub_.shape[0]
    # print('it:{} sum_sub_: {:.5f} ({})'.format(i,sum_sub_, sum_sub_))
    mean_total=mean_total+sum_sub_
mean_total=mean_total/10
# print("Total sum_sub_mean for 0: {}".format(mean_total))
print('single mean for 2: {}'.format(mean_total.mean()))