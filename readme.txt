Introduction:

Pointnet ++ implementation for training on the ModelNet40 dataset with train test and val. Most changes can be made from 
the hyperparameters. 


Requirements:

CUDA GPU recommended but not necessary(Will take extremly long time if not avalible) 
torch (PyTorch) - A deep learning framework.
numpy - A package for scientific computing with Python.
tqdm - progress bar for loops
scikit-learn - Machine learning library for Python.
open3d - Library for 3D data processing.
matplotlib - Plotting library for Python.
argparse - Module for writing  user-friendly command-line interfaces.
seaborn - Visualization library based on matplotlib.



Installation:

pip install torch
pip install numpy
pip install tqdm
pip install scikit-learn
pip install open3d
pip install matplotlib
pip install seaborn



File descriptions : 

-hyperparameters.py  -  To change hyperparameters
- `model`: The model to use. Options : 'pointnet2_cls_ssg' & 'pointnet2_cls_msg'.
- `batch_size`: The size of the batch for training.
- `nepochs`: The number of epochs to train for.
- `nclasses`: The number of classes in the dataset.
- `lr`: The learning rate.
- `decay_rate`: The rate of decay for the learning rate.
- `step_size`: The step size for learning rate decay.
- `gamma`: The gamma for learning rate decay.
- `log_dir`: The directory for logs.
- `checkpoint_interval`: The interval to save model checkpoints.
- `optimizer`: The optimizer to use. Default is 'adam'.
- `beta1`: The momentum-like term for the Adam optimizer.
- `beta2`: The adaptive learning rate term for the Adam optimizer.
- `dropout_rate1`: The first dropout rate.
- `dropout_rate2`: The second dropout rate.
- `npoints`: The number of points for each datapoint .

-eval.py             -  Validation (final test) logic 
This Python file contains the evaluation code. Computes loss, accuracy, F1 score, and AUC (Area Under the ROC Curve) for the model's predictions.

-validate.py         -  To execute Validation(final test)


-ModelNet40.py       -  Dataloader
Defines custom PyTorch dataset for the ModelNet40 dataset 

-Provider.py         -  Data augmentation and normalitazion
Contains utility functions for manipulating and augmenting point cloud data

-Solver.py           -  Training logic 
Contains the train, test one epoch and train logic. 

-trainingfunc.py     -  Training functions
Contains functions logging and saving  that are used for training 

-trainingstats.py    - Training statistic functions 
Contain functions for calculating statistics for training and testing

-Visualize.py        - Genereate visual to showncase 1 datapoint and graph overview of all classes 
Functions for generating a visual representation of the dataset and overview of the classes.

-Models/pointnet.py  - Defines the pointnet++ neural network model 

Data/ModelNet40     - Data for training 





Configuration and customization:

Most of what is subject to change is in the hyperparameters.py 



Usage: 

1. Train.py:
Ensure you have the necessary dependencies installed. 

Download the  ModelNet40 dataset , place in the ./Data/ModelNet40_resampled directory. Dataset should be split into 'train','test' and 'val'.

Set your desired hyperparameters in the hyperparameters.py file.

Run the script. "python train.py".

Script assumes the existence of certain files and directories (like the ModelNet40 dataset and the ./checkpoints directory).



2. Validate.py:
Ensure you have the necessary dependencies installed.

Download the  ModelNet40 dataset , place in the ./Data/ModelNet40_resampled directory. Dataset should be split into 'train','test' and 'val'.

Run the script with the path to the trained model file as an argument. "python validate.py --model_path path_to_your_model".



Acknowledgements:

https://github.com/yanx27/Pointnet_Pointnet2_pytorch
MIT License

Copyright (c) 2019 benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Zhulf0804. (2023). Pointnet2.PyTorch [Computer software].
GitHub. https://github.com/zhulf0804/Pointnet2.PyTorch



https://github.com/charlesq34/pointnet

PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.

Copyright (c) 2017, Geometric Computation Group of Stanford University

The MIT License (MIT)

Copyright (c) 2017 Charles R. Qi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.