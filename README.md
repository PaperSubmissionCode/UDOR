## UDOR

Codes, datasets, pre-trained models and guides of paper: **Disassembling Object Representation without Labels**

## Index

<!-- TOC -->

- [Dependencies](#Dependencies)
- [Multi-MNIST Experiments](#Multi-MNIST-Experiments)
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

**Notes:** If you encounter the problem about ***the large file error*** with this repository when you *clone* by Git, you can refer to [Git Large File Storage (LFS)](<https://git-lfs.github.com/>) and  [Git LFS support](<https://github.com/rtyley/bfg-repo-cleaner/releases/tag/v1.12.5>)  to address this problem.

## Multi-MNIST Experiments

We give codes of three methods, including S-AE, DSD, UDOR on the Multi-MNIST dataset. 

Here, we set 'unitLength=1' as an example. The part length, which is the length of each part of the representation, can be set to any integer that is large than 0. In the code file,  the part length is denoted as 'unitLength'.

### Datasets

The dataset generation examples are give as follows:

1. Run ```main_generate_MultiMNISTDataset.py``` to get  the **Multi-mnist datasets** from TensorFlow API (*only use train data through this API*), which will be used to produce the training datasets as ```npz``` format. It's saved in ```./multiMnistDataset_32x32/MnistRandom012ArrayNorm0_1.npz``` *(for UDOR)*

2. Run  ```main_generate_MultiMnist_trainAndtestAndmask.py``` to get the **training datasets** for UDOR which is saved in ```./npz_datas/unitLength1_mnistMultiAndMask_10000x32x32x1_train.npz``` *(for UDOR)*

3. Run ```main_generateMNISTBatchTrainData_SAE.py``` to get **training datasets** for S-AE which is saved in ```./npz_datas/SAE_mnistMultiAndMaskAndLabels_unitLength_1_10000x32x32x1_train.npz```. (for S-AE)

4. Run ```main_generateMultiMNISTforDSD.py``` to get **training datasets** for DSD which is saved in ```./npz_DSD_dataset/``` (for DSD)

5. Run  ```./main_generate_MultiMnist_testAndmask_visual.py``` to get the **testing  datasets** saved in ```./npz_datas/```, There are ```mnist_(20x64)x32x32x1_unitLength1_test_visualdata1.npz```, ```mnist_(20x64)x32x32x1_unitLength1_test_visualdata2.npz```, ```DSD_data1_3_mnist_(20x64)x32x32x1_unitLength1_test.npz```, ```DSD_data2_mnist_(20x64)x32x32x1_unitLength1_test.npz```, and ```DSD_data4_mnist_(20x64)x32x32x1_unitLength1_test.npz```. The **[...visualdata1]** and **[...visualdata2]** are used for zero-reseting and object-swapping of SAE and our UDOR. And other three datasets ( **[DSD_....]** ) are generated for DSD. 

   We give a simple description of these five testing datasets in the following table. See more details from ```main_generate_MultiMnist_testAndmask_visual.py```.

   |                                                     | [...visualdata1.npz]                                 | [...visualdata2.npz]                                   |
   | :-------------------------------------------------- | :--------------------------------------------------- | :----------------------------------------------------- |
   | Each image                                          | composed by digital 0, 1, 2 randomly                 | composed by digital 0, 1, 2 randomly                   |
   | Four masks compose <br>one group <br>16 same groups | [1, 1, 1],<br>[0, 1, 1],<br>[1, 0, 1],<br/>[1, 1, 0] | [0, 0, 0],<br/>[1, 0, 0],<br/>[0, 1, 0],<br/>[0, 0, 1] |

   |                                                    | [DSD_data1_3]                                        | [DSD_data2...]                                         | [DSD_data4...]                                         |
   | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
   | Each image                                         | composed by digital 0, 1, 2 randomly                 | composed by digital 0, 1, 2 randomly                   | composed by digital 0, 1, 2 randomly                   |
   | Four masks compose<br>one group <br>16 same groups | [1, 1, 1],<br/>[0, 1, 1],<br>[1, 0, 1],<br>[1, 1, 0] | [0, 0, 0],<br/>[1, 0, 0],<br/>[0, 1, 0],<br/>[0, 0, 1] | [0, 0, 0],<br/>[1, 0, 0],<br/>[0, 1, 0],<br/>[0, 0, 1] |

   

### Train

After preparing the datasets, you can train the model with ```main.py```, which is given in the directory of each method.

***Notes: In the training stage, you maybe encounter with the "NaN" problem. The root cause is the 'log' function in the classification loss. You can alleviate it by decreasing epochs or learning rate.*** 

#### UDOR

```shell
cd models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3
python main.py
```

The intermediate results in training time will be saved in ```./models_UDOR/MNIST_GAN_1_1_10_1000_lr1e-3/samples/```

#### S-AE

```shell
cd SAE/SAE_part3_unitLength1
# train
python main.py
```

The intermediate results in training time will be saved in ```./SAE/SAE_part3_unitLength1/samples/```

#### DSD

```shell
cd DSD/dual_diaeMnist_unitLeng1
# train
python main.py
```

The intermediate results in training time will be saved in ```./DSD/dual_diaeMnist_unitLeng1/samples/```



### Test 

Now you can use the trained model to do visualization testing! Please choose the best model which  can produce the best result in ```./samples/``` folder. For example, the best reconstructed image is ```1000reset0_reconstructed.png```, which means that we have the best model's ckpt when step=1000.

For example, the step=1000, you need to modify one line in ```test_getVisualImage.py```

```python
# change the ckpt number that you want use
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

The results will be saved  in ```./SAE/SAE_part3_unitLength1/VisualImgsResults/```

#### DSD

```shell
cd DSD/dual_diaeMnist_unitLeng1
# test
python test_getVisualImage.py
```

The results will be saved  in ```./DSD/dual_diaeMnist_unitLeng1/VisualImgsResults/```

### Test from pre-trained model

For each method, we provide a pre-trained model, which can be used for quick visualization testing after datasets preparation.

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

The ```test_getVisualImage.py``` is a visual example.

For each method, the first row gives the original images and the swapping candidate images, respectively. The second row shows the zero-resetting results and the object-swapping results.

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

First of all, please generate the datasets for calculating metrics.

```shell
# generate datasets for UDOR, S-AE and DSD to calculate metrics
python main_generate_MultiMnist_testForMetrics.py
```

Then you can ```cd``` to each method directory for calculating the metric scores.

#### UDOR

```shell
# cd your_work_path
cd models_UDOR/Multi-Mnist/MMNIST_GAN_1_1_10_1000_lr1e-3_pretrained
# you should set the ckpt which to be loaded from your_work_path
python test_getRepreCodes_forMetrics.py
# calculate visual metrics
python visual_metrics_cal.py
# calculate modularity metrics
python modularity_metrics_cal.py
# calculate classificaiton metrics
python classify_metrics_cal.py
```

*Note: With the newly trained model, please check the corresponding relationship between  each part of the representation and each object category. According to the corresponding relationship, you should change the mask setting in ```main_generate_MultiMnist_testForMetrics.py```. With the newly generated testing dataset, the  file ```modularity_metrics_cal.py``` will  calculate the modularity scores correctly.*

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

Notes: The S-AE is a supervised method. The ```main_generate_MultiMnist_testForMetrics.py``` doesn't need to be modified. 

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

Notes: The DSD is a semi-supervised method. The file ```main_generate_MultiMnist_testForMetrics.py``` dosen't need to be modified.

## Multi-Fashion Experiments

You should first ```cd models_UDOR/Multi-Fashion/``` directory.

### Datasets

1. Run ```main_generate_FashionTrainData.py``` 

2. Run ```main_generate_MultiFashion_testAndmask_visual.py```

   The following table gives a simple description of two testing datasets. See more details from ```main_generate_MultiFashion_testAndmask_visual.py```.

|                                                     | [...visualdata1.npz]                                         | [...visualdata2.npz]                                         |
| :-------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Each image                                          | composed by T-shirt,Trouser,Bag,Ankle boot randomly<br/>four fashion objects | composed by T-shirt,Trouser,Bag,Ankle boot randomly<br/>four fashion objects |
| Four masks compose <br>one group <br>16 same groups | [0, 1, 1, 1],<br>[1, 0, 1, 1],<br>[1, 1, 0, 1],<br/>[1, 1, 1, 0] | [1, 0, 0, 0],<br/>[0, 1, 0, 0],<br/>[0, 0, 1, 0],<br/>[0, 0, 0, 1] |

### Train

After preparing the datasets, you can easily train your model with  ```main.py```.

```shell
cd Fashion_GAN_1_1_10_1000_lr1e-3
python main.py
```

### Test 

```python
# change the ckpt that you want to use
saved_step = AE.load_fixedNum(inter_num=1000)
```

Running the following command to get the results.

```shell
python test_getVisualImage.py
```

The results will be saved in ```./Fashion_GAN_1_1_10_1000_lr1e-3/VisualImgsResults/```.

### Test from pre-trained model

The pre-trained model with `unitLength=1' is given in ```./Fashion_GAN_1_1_10_1000_lr1e-3_pretrained/```, which can be used for the visualization testing. 

```shell
cd Fashion_GAN_1_1_10_1000_lr1e-3_pretrained
python test_getVisualImage.py
```

### Visualization Example

The ```test_getVisualImage.py``` is a visual example.

The first row gives the original images and the swapping candidate images, respectively. The second row shows the zero-resetting results and the object-swapping results.

![input](models_UDOR/Multi-Fashion/Visualization%20Example/input.png)

![output](models_UDOR/Multi-Fashion/Visualization%20Example/output.png)



## Object Position Experiments

Here, we set position offset to 1 (denoted by max_offset=1) as an example. For the 32x32 image with 14x14 digit image patch, position offset can be set to any positive integer that is less than 17.

#### Datasets

```shell
# generate Multi-MNIST dataset with max_offset=1
python main_generate_MultiMNISTDataset_Offset.py
# generate the validating dataset in training stage
python main_testMultiMnist_Offset.py
# generate the training and validating datasets in npz format   
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
# cd your_work_path, here is our pretrained dirctory
cd models_UDOR/Mnist-Offset/GAN_1_1_10_1000_offset1_pretrained
# test
python test_getVisualImage.py
```

### Calculate metrics

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

*Note: With the newly trained model, please check the corresponding relationship between  each part of the representation and each object category. According to the corresponding relationship, you should change the mask setting in ```main_generateSingleTest_Offset.py``` . With the newly generated testing dataset, the  file ```modularity_metrics_cal.py```  will  calculate the modularity scores correctly.*

## Pattern-Design Experiments

Here, we set 'unitLength=9' as an example. The part length can be set to any integer  that is large than 0.

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
# cd your_work_path, here is our dirctory
cd models_UDOR/Pattern-Design/GAN_1_1_10_10000_u9_64x64_pretrained
# test
python test_getVisualImage.py
```

### Visualization Example

![input](models_UDOR/Pattern-Design//Visualization%20Example/input.png)

![output](models_UDOR/Pattern-Design//Visualization%20Example/output.png)

## Acknowledgement

Thanks for cianeastwood, some codes are based on the lib of [https://github.com/cianeastwood/qedr](https://github.com/cianeastwood/qedr).

