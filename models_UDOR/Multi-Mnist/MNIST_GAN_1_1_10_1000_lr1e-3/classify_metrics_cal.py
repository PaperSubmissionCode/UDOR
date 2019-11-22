"""
this file is used for prediect the class, for class accuracy with representation variants
"""
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics

unitlLength=1
data_npz_path='../../../npz_datas/codes/codes_UDOR_resnet_3_32_lr_0.001.npz'
label_npz_path='../../../npz_datas/nistImgs_GTimages_mask_GTlabel_(47x64)x32x32x1_unitLength1_CodeImageDataset_forUDOR.npz'
data=np.load(data_npz_path)['codes']
label=np.load(label_npz_path)['labelsGT']

spot=np.where(np.isnan(data))[0]
data=np.delete(data,spot,axis=0)
label=np.delete(label,spot,axis=0)
# print("data shape:{}".format(data.shape))
# print("label sahpe:{}".format(label.shape))
assert len(data)>=3000,"len data is <3000"
assert len(label)>=3000, "len label is <3000"
assert len(data)==len(label),'len data != len label!'

state=np.random.get_state()
np.random.shuffle(data)

np.random.set_state(state)
np.random.shuffle(label)

train_num=1500
test_num=1500
data_train=data[0:train_num,]
label_train=label[0:train_num,]

data_test=data[train_num:train_num+test_num,]
label_test=label[train_num:train_num+test_num,]
label_test[:,0]=np.where(label_test[:,0]==-1,0,1)
label_test[:,1]=np.where(label_test[:,1]==-1,0,1)
label_test[:,2]=np.where(label_test[:,2]==-1,0,1)


# 3 segment for 3 svm model
## for 0
model_0 = SVC(kernel='linear', probability=True)
train_label_0=np.where(label_train[:,0]==-1,0,1)
model_0.fit(data_train[:,2*unitlLength:3*unitlLength], train_label_0)
pre_0 = model_0.predict_proba(data_test[:,2*unitlLength:3*unitlLength])
pre1_0 = model_0.predict(data_test[:,2*unitlLength:3*unitlLength])

## for 1
model_1 = SVC(kernel='linear', probability=True)
train_label_1=np.where(label_train[:,1]==-1,0,1)
model_1.fit(data_train[:,unitlLength:2*unitlLength], train_label_1)
pre_1 = model_1.predict_proba(data_test[:,unitlLength:2*unitlLength])
pre1_1 = model_1.predict(data_test[:,unitlLength:2*unitlLength])

# for 2
model_2 = SVC(kernel='linear', probability=True)
train_label_2=np.where(label_train[:,2]==-1,0,1)
model_2.fit(data_train[:,0:unitlLength], train_label_2)
pre_2 = model_2.predict_proba(data_test[:,0:unitlLength])
pre1_2 = model_2.predict(data_test[:,0:unitlLength])


pre1_0=pre1_0[:,np.newaxis]
pre1_1=pre1_1[:,np.newaxis]
pre1_2=pre1_2[:,np.newaxis]
pre_all=np.concatenate([pre1_0,pre1_1,pre1_2],axis=1)
# print(pre_all.shape)
# print(data_test.shape)


recall_0=metrics.recall_score(label_test[:,0],pre_all[:,0])
recall_1=metrics.recall_score(label_test[:,1],pre_all[:,1])
recall_2=metrics.recall_score(label_test[:,2],pre_all[:,2])

precision_0=metrics.precision_score(label_test[:,0],pre_all[:,0])
precision_1=metrics.precision_score(label_test[:,1],pre_all[:,1])
precision_2=metrics.precision_score(label_test[:,2],pre_all[:,2])

print("C-R: {}".format((recall_0+recall_1+recall_2)/3.0)) # == metrics.recall_score(label_test,pre_all,average='macro')
print("C-P: {}".format((precision_0+precision_1+precision_2)/3.0))

# print(label_test.shape)
# print(pre_all.shape)
recall_overall=metrics.recall_score(label_test,pre_all,average='micro')
precision_overall=metrics.precision_score(label_test,pre_all,average='micro')
micro_f1_score=metrics.f1_score(y_true=label_test,y_pred=pre_all,average='micro')
macro_f1_score=metrics.f1_score(y_true=label_test,y_pred=pre_all,average='macro')
print('micro_f1_score: {}'.format(micro_f1_score))
print('macro_f1_score: {}'.format(macro_f1_score))

print('O-R(globally): {}'.format(recall_overall))
print('O-P: {}'.format(precision_overall))
