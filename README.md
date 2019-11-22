## UDOR

Codes, datasets and guides of paper: **Disassembling Object Representation without Labels**

If you want to use pre-trained model, please download it from [One Drive](https://1drv.ms/u/s!AgDnjfnQi6vCa8V0gg6DBqknV_c?e=VOhw2P) and place the model to the corresponding path.

<!-- TOC -->

- [Dependencies](#Dependencies)

- [Multi-mnist Experiments](#Multi-mnist-Experiments)
- [Multi-Fashion Experiments](#Multi-Fashion-Experiments)
- [Object Position Experiments](#Object-Position-Experiments)
- [Pattern-Design Experiments](#Pattern-Design-Experiments)
- [Acknowledgement](#Acknowledgement)

<!-- TOC -->

## Dependencies

```xml
python3
tensorflow=1.8
pillow
py-opencv
scipy<1.3.0
scikit-learn
```

## Multi-mnist Experiments

Multi-mnist based experiments include our method UDOR and S-AE, DSD methods as paper discussed. 

Here, we set unitLength=1 as an example. Each part of the representation can be set to any integer  that is large than 0.

### Datasets

Here, we will produce the related datasets of this experiment. 

1. Run ```main_generate_MultiMNISTDataset.py``` to get  the **Multi-mnist datasets** from TensorFlow API (**only use train data through this API**), which will be used to produce the training datasets as ```npz``` format. It's saved in ```./multiMnistDataset_32x32/MnistRandom012ArrayNorm0_1.npz``` *(for UDOR)*

2. Run  ```main_generate_MultiMnist_trainAndtestAndmask.py``` to get the **training datasets** for UDOR which is saved in ```./npz_datas/unitLength1_mnistMultiAndMask_10000x32x32x1_train.npz``` *(for UDOR)*

3. Run ```main_generateMNISTBatchTrainData_SAE.py``` to get **training datasets** for S-AE which is saved in ```./npz_datas/SAE_mnistMultiAndMaskAndLabels_unitLength_1_10000x32x32x1_train.npz```. (for S-AE)

4. Run ```main_generateMultiMNISTforDSD.py``` to get **training datasets** for DSD which is saved in ```./npz_DSD_dataset/``` (for DSD)

5. Run  ```./main_generate_MultiMnist_testAndmask_visual.py``` to get the **testing  datasets** saved in ```./npz_datas/```, There are ```mnist_(20x64)x32x32x1_unitLength1_test_visualdata1.npz```, ```mnist_(20x64)x32x32x1_unitLength1_test_visualdata2.npz```, ```DSD_data1_3_mnist_(20x64)x32x32x1_unitLength1_test.npz```, ```DSD_data2_mnist_(20x64)x32x32x1_unitLength1_test.npz```, and ```DSD_data4_mnist_(20x64)x32x32x1_unitLength1_test.npz```. One of the **[...visualdata1] and [...visualdata2] are used for zero-reseting and both for object-swapping as our paper depicted by SAE and our method UDOR**. And we use other three **[DSD_....]** for DSD method. *(for SAE, DSD, UDOR)*

   Below is a simple description of these five testing datasets. See more details from ```main_generate_MultiMnist_testAndmask_visual.py```.

   |                                                     | [...visualdata1.npz]                                 | [...visualdata2.npz]                                   |
   | :-------------------------------------------------- | :--------------------------------------------------- | :----------------------------------------------------- |
   | Each image                                          | random of 0,1,2,<br/>three digits                    | random of 0,1,2,<br/>three digits                      |
   | Four masks compose <br>one group <br>16 same groups | [1, 1, 1],<br>[0, 1, 1],<br>[1, 0, 1],<br/>[1, 1, 0] | [0, 0, 0],<br/>[1, 0, 0],<br/>[0, 1, 0],<br/>[0, 0, 1] |

   |                                                    | [DSD_data1_3]                                        | [DSD_data2...]                                         | [DSD_data4...]                                         |
   | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
   | Each image                                         | random of 0,1,2,<br/>three digits                    | random of 0,1,2,<br/>three digits                      | random of 0,1,2,<br/>three digits                      |
   | Four masks compose<br>one group <br>16 same groups | [1, 1, 1],<br/>[0, 1, 1],<br>[1, 0, 1],<br>[1, 1, 0] | [0, 0, 0],<br/>[1, 0, 0],<br/>[0, 1, 0],<br/>[0, 0, 1] | [0, 0, 0],<br/>[1, 0, 0],<br/>[0, 1, 0],<br/>[0, 0, 1] |

   

### Train

After preparing the datasets, you can train your model with commands, see ```main.py``` in each experiment directory for more details for advanced training!

***Notes: In the training stage, you maybe encounter "NaN" logs. This is because of the classification loss. You can avoid it by decreasing epochs or learning rate.*** 

#### UDOR

```shell
cd models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3
python main.py
```

The intermediate result in training time will be saved in ```./models_UDOR/MNIST_GAN_1_1_10_1000_lr1e-3/samples/```

***Notes: And you maybe need to train several times to find the best model because of the GAN's unstable properties. From our knowledge,  it will have at least one good model in 5 times training.***

#### S-AE

```shell
cd SAE/SAE_part3_unitLength1
# train
python main.py
```

The intermediate result in training time will be saved in ```./SAE/SAE_part3_unitLength1/samples/```

#### DSD

```
cd DSD/dual_diaeMnist_unitLeng1
# train
python main.py
```

The intermediate result in training time will be saved in ```./DSD/dual_diaeMnist_unitLeng1/samples/```



### Test 

Now you can use the trained model to do visualization testing! Please use the best model which  can produce the best result in ```./samples/``` folder. For example, the best-reconstructed image is ```1000reset0_reconstructed.png```, which means that we have the best model's ckpt when step=1000.

For example, the step=1000, you need to modify one line in ```test_getVisualImage.py```

```python
# change the which iteration ckpt you want to use
saved_step = AE.load_fixedNum(inter_num=1000)
```

#### UDOR

```shell
cd models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3
# test
python test_getVisualImage.py
```

The results will be saved in ```./models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3/VisualImgsResults/```

#### S-AE

```shell
cd SAE/SAE_part3_unitLength1
# test
python test_getVisualImage.py
```

The results will in ```./SAE/SAE_part3_unitLength1/VisualImgsResults/```

#### DSD

```shell
cd DSD/dual_diaeMnist_unitLeng1
# test
python test_getVisualImage.py
```

The results will be saved  in ```./DSD/dual_diaeMnist_unitLeng1/VisualImgsResults/```

### Test from pre-trained model

We also have placed one pre-trained checkpoint with unitLength=1 in each of S-AE, DSD, and UDOR method for a quick visualization testing after datasets preparation. 

#### UDOR

```shell
cd models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3_pretrained
# test
python test_getVisualImage.py
```

The results can be found in ```./models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3/VisualImgsResults/```

#### S-AE

```shell
cd SAE/SAE_part3_unitLength1_pretrained
# test
python test_getVisualImage.py
```

The results can be found in ```./SAE/SAE_part3_unitLength1/VisualImgsResults/```

#### DSD

```shell
cd DSD/dual_diaeMnist_unitLeng1_pretrained
# test
python test_getVisualImage.py
```

The results can be found in ```./DSD/dual_diaeMnist_unitLeng1/VisualImgsResults/```

### Visualization Example

Here are examples by run ```test_getVisualImage.py```.

The top-left is the input with a ```batch_size=64``` of ```[...visualdata1.npz]```and ```[...visualdata2.npz]``` respectively, the bottom-left is the zero-resetting result and the bottom-right is the object-swapping result.

#### UDOR

![UDOR_1](models_UDOR/Multi-Mnist/Visualization%20Example/input.png)

![UDOR_2](models_UDOR/Multi-Mnist/Visualization%20Example/output.png)

#### S-AE

![SAE_0](SAE/Visualization%20Example/input.png)

![SAE_1](SAE/Visualization%20Example/output.png)

#### DSD

![DSD_1](DSD/Visualization%20Example/input.png)

![DSD_2](DSD/Visualization%20Example/output.png)

### Calculate metrics

Here is based on our pre-trained model. If you have trained a model, you can change the workspace, and the pipelines are the same.

First of all, please generate the datasets for calculating metrics.

```shell
# generate datasets for UDOR, S-AE and DSD to calculate metrics, for more dtails in paper
python main_generate_MultiMnist_testForMetrics.py
```

Then you can ```cd``` each method directory for calculating the metrics.

#### UDOR

```shell
# cd your_work_path
cd models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3_pretrained
# you should set the which ckpt to load in your_work_path
python test_getRepreCodes_forMetrics.py
# calculate visual metrics
python visual_metrics_cal.py
# calculate modularity metrics
python modularity_metrics_cal.py
# calculate classificaiton metrics
python classify_metrics_cal.py
```

*Note: If you use your trained model, please check which segment represents 0 digits, which is 1 digit and 2. According to this discovery, you should change the mask setting in ```main_generate_MultiMnist_testForMetrics.py``` and change 0,1,2 representation in modularity calculation  file ```modularity_metrics_cal.py``` for forwarding metrics calculation correctly.*

#### S-AE

```shell
# cd your_work_path
cd SAE/SAE_part3_unitLength1_pretrained
# codes and corresponding images will be saved in ./npz_datas/codes/
python test_getRepreCodes_forMetrics.py
# calculate visual metrics
python visual_metrics_cal.py
# calculate modularity metrics
python modularity_metrics_cal.py
# calculate classificaiton metrics
python classify_metrics_cal.py
```

Notes: Because the S-AE is a supervised training method, you don't have to modify ```main_generate_MultiMnist_testForMetrics.py```. 

#### DSD

```shell
# cd your_work_path
cd DSD/dual_diaeMnist_unitLeng1_pretrained
# codes and corresponding images will be saved in ./npz_datas/codes/
python test_getRepreCodes_forMetrics.py
# calculate visual metrics
python visual_metrics_cal.py
# calculate modularity metrics
python modularity_metrics_cal.py
# calculate classificaiton metrics
python classify_metrics_cal.py
```

*Note: If you use your trained model, please check which segment represents 0 digits, which is 1 digit and 2. According to this discovery, you should change the mask setting in ```main_generate_MultiMnist_testForMetrics.py``` and change 0,1,2 representation in modularity calculation  file ```modularity_metrics_cal.py``` for forwarding metrics calculation correctly.*

## Multi-Fashion Experiments

You should first ```cd models_UDOR/Multi-Fashion/``` directory.

### Datasets

1. Run ```main_generate_FashionTrainData.py``` 

2. Run ```main_generate_MultiFashion_testAndmask_visual.py```

   Below is a simple description of there two testing datasets. See more details from ```main_generate_MultiFashion_testAndmask_visual.py```.

|                                                     | [...visualdata1.npz]                                         | [...visualdata2.npz]                                         |
| :-------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Each image                                          | random of T-shirt,Trouser,Bag,Ankle boot<br/>four fashion objects | random of T-shirt,Trouser,Bag,Ankle boot<br/>four fashion objects |
| Four masks compose <br>one group <br>16 same groups | [0, 1, 1, 1],<br>[1, 0, 1, 1],<br>[1, 1, 0, 1],<br/>[1, 1, 1, 0] | [1, 0, 0, 0],<br/>[0, 1, 0, 0],<br/>[0, 0, 1, 0],<br/>[0, 0, 0, 1] |

### Train

After preparing the datasets, you can quicklytrain your model with commands, see ```main.py``` for more details for advanced training!

```shell
cd Fashion_GAN_1_1_10_1000_lr1e-3
python main.py
```

***Notes: And you maybe need to train several times to find the best model because of the GAN's unstable properties. From our knowledge, Fashion-based experiments are harder than Mnist-based in the training stage.***

### Test 

```python
# change the which iteration ckpt you want to use
saved_step = AE.load_fixedNum(inter_num=1000)
```

And, run below command to get the results.

```shell
python test_getVisualImage.py
```

The results are in ```./Fashion_GAN_1_1_10_1000_lr1e-3/VisualImgsResults/```.

### Test from pre-trained model

We also have placed one pre-trained checkpoint with unitLength=1 in ```./Fashion_GAN_1_1_10_1000_lr1e-3_pretrained/``` for a quickly visualization testing after datasets preparation. 

```shell
cd Fashion_GAN_1_1_10_1000_lr1e-3_pretrained
python test_getVisualImage.py
```

### Visualization Example

Here is our one example by run ```test_getVisualImage.py```.

The top-left is the input with a ```batch_size=64``` of ```[...visualdata1.npz]```and ```[...visualdata2.npz]``` respectively, the bottom-left is the zero-resetting result, and the bottom-right is the object-swapping result.

![input](models_UDOR/Multi-Fashion/Visualization%20Example/input.png)

![output](models_UDOR/Multi-Fashion/Visualization%20Example/output.png)



## Object Position Experiments

Here, we set position max_offset=1 with unitlength=1 as example. The max_offset can be set to any number that >=0 and the unitlength can be set to any positive integer.

#### Datasets

```shell
# generate multi-mnist with position offset=1 datasets
python main_generate_MultiMNISTDataset_Offset.py
# generate the valid dataset in training stage
python main_testMultiMnist_Offset.py
# generate the training and validating npz format datasets 
python main_mnist_npz2npz_Offset.py
```

#### Train

```shell
cd models_UDOR/Mnist-Offset/GAN_1_1_10_1000_offset1
# train
python main.py
```

#### Test  

```shell
# generate testing datasets
python main_generateSingleTest_Offset.py
# cd your_work_path, here is our providing pretrained dirctory
cd models_UDOR/Mnist-Offset/GAN_1_1_10_1000_offset1_pretrained
# test
python test_getVisualImage.py
```

### Calculate metrics

Here is based on our pre-trained model. If you have trained a model, you can change the workspace and the pipelines are the same.

```shell
# cd your_work_path
cd models_UDOR/Mnist-Offset/GAN_1_1_10_1000_offset1_pretrained
# codes and corresponding images will be saved in ./npz_datas/codes/
python test_getRepreCodes_forMetrics.py
# calculate visual metrics
python visual_metrics_cal.py
# calculate modularity metrics
python modularity_metrics_cal.py
```

*Note: If you use your trained model, please check which segment represents 0 digits and which is 1 digit. According to this discovery, you should change the mask setting in ```main_generateSingleTest_Offset.py``` and change which is 0,1 representation in modularity calculation  file ```modularity_metrics_cal.py``` for forwarding metrics calculation correctly.*

## Pattern-Design Experiments

Here, we set unitLength=9 as the example. The each part of the representation can be set to any integer  that is large than 0.

### Datasets

1. Uzip the datasets, 

   ```shell
   cd pattern_flower_tree_leaf
   # flower
   unzip flowers.zip
   # tree
   unzip trees.zip
   # leaf
   unzip leafs.zip
   ```

2. Run ```main_generatePatternDataForPattern_train.py``` 

### Train

```shell
cd models_UDOR/Pattern-Design/GAN_1_1_10_10000_u9_64x64
# trian
python main.py
```

### Test

```shell
# generate testing datasets
python main_generatePatternDataForPattern_test.py
# cd your_work_path, here is our providing pretrained dirctory
cd models_UDOR/Pattern-Design/GAN_1_1_10_10000_u9_64x64_pretrained
# test
python test_getVisualImage.py
```

### Visualization Example

![input](models_UDOR/Pattern-Design//Visualization%20Example/input.png)

![output](models_UDOR/Pattern-Design//Visualization%20Example/output.png)

## Acknowledgement

Thanks for cianeastwood, some of our code is based on the lib of [https://github.com/cianeastwood/qedr](https://github.com/cianeastwood/qedr).

