# Icevision
> a computer vision framework for end-to-end training of curated models


IceVision is the first agnostic computer vision framework to offer a curated collection with hundreds of high-quality pre-trained models from [torchvision](https://pytorch.org/vision/stable/index.html), [MMLab](https://openmmlab.com/), Ultralytics' [yolov5](https://github.com/ultralytics/yolov5) and Ross Wightman's [EfficientDet](https://github.com/rwightman/efficientdet-pytorch). It orchestrates the end-to-end deep learning workflow allowing to train networks with easy-to-use robust high-performance libraries such as [Pytorch-Lightning](https://www.pytorchlightning.ai/) and [Fastai](https://docs.fast.ai/).

#Features of Icevision

*   Data curation/cleaning with auto-fix
*   Access to an exploratory data analysis dashboard
*   Pluggable transforms for better model generalization
*   Access to hundreds of neural net models
*   Access to multiple training loop libraries
*   Multi-task training to efficiently combine object detection, segmentation, and classification models


















# [Join our Forum](https://discord.gg/JDBeZYK)

#Quick Example: How to train the Fridge Objects Dataset

```python
from icevision.all import *

#Model
model_type = models.mmdet.retinanet

#Backbone form MMDetection Example
backbone = model_type.backbones.resnet50_fpn_1x

#Loading Data
data_dir = icedata.fridge.load_data()

#Parsing
parser = icedata.fridge.parser(data_dir)
train_records, valid_records = parser.parse()

#Transforms
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

# Datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

#Create model object
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map)) 

#Learner
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

#LR Finder
learn.lr_find()

#Training
learn.fine_tune(20, 1e-4, freeze_epochs=1)
```

#Installing Icevision

{% include important.html content='We currently only support Linux/MacOS installations ' %}

{% include note.html content='Please do not forget to install the other optional dependencies if you would like to use them: MMCV+MMDetection, and/or YOLOv5' %}

##Installation on colab

```python
#check cuda version
import torch
cuda_version_major = int(torch.version.cuda.split('.')[0])
cuda_version_major
```

```python
#install packages based on your cuda version
!wget https://raw.githubusercontent.com/airctic/icevision/master/install_colab.sh
!bash install_colab.sh {cuda_version_major}
```

```python
# Restart kernel
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

##Installation using pip

Option 1: Installing from pypi repository [Stable Version]

`pip install icevision[all]`

Option 2: Installing an editable package locally [For Developers]

{% include note.html content='This method is used by developers who are usually either:' %}> actively contributing to icevision project by adding new features or fixing bugs, or
creating their own extensions, and making sure that their source code stay in sync with the icevision latest version.


```bash
$ got clone --depth=1
https://github.com/airctic/icevision.git

$ cd icevision
$ pip install -e .[all,dev]
$ pre-commit istall
```

Option 3: Installing a non-editable package from GitHub:

To install the icevision package from its GitHub repo, run the command here below. This option can be used in Google Colab, for example, where you might install the icevision latest version (from the master branch)

```pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] --upgrade```

##Installation using conda

Creating a conda environment is considered as a best practice because it avoids polluting the default (base) environment, and reduces dependencies conflicts. Use the following command in order to create a conda environment called icevision.

```
$ conda create -n icevision python=3.8 anaconda
$ conda activate icevision
$ pip install icevision[all]
```

#Installing optional dependencies 

##MMDetection Installation

We need to provide the appropriate version of the mmcv-full package as well as the cuda and the torch versions. Here are some examples for both the CUDA and the CPU versions.

{% include important.html content='For the torch version use `torch.__version__` and replace the last ' %}> number with 0. For the cuda version use:`torch.version.cuda.Example: TORCH_VERSION = torch1.8.0; CUDA_VERSION = cu101`

For CUDA version,
```python
$ pip install mmcv-full=="1.3.3" -f https://download.openmmlab.com/mmcv/dist/CUDA_VERSION/TORCH_VERSION/index.html --upgrade
$ pip install mmdet
```
For CPU version,
```python
$ pip install mmcv-full=="1.3.3+torch.1.8.0+cpu" -f https://download.openmmlab.com/mmcv/dist/index.html --upgrade
$ pip install mmdet
```

##YOLOv5 Installation

`pip install yolov5-icevision --upgrade`

#Troubleshooting

## MMCV is not installing with cuda support
If you are installing MMCV from the wheel like described above and still are having problems with CUDA you will probably have to compile it locally. Do that by running:

`pip install mmcv-full`

If you encounter the following error it means you will have to install CUDA manually (the one that comes with conda installation will not do).

`OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`

Try installing it with:

`sudo apt install nvidia-cuda-toolkit`

Check the installation by running:

`nvcc --version`

Error: Failed building wheel for pycocotools
If you encounter the following error, when installation process is building wheel for pycocotools:

```python
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

Try installing gcc with:

`sudo apt install gcc`

Check the installation by running:

`gcc --version`

It should return something similar:

```
gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

After that try installing icevision again.


If you need any assistance, feel free to:

# [Join our Forum](https://discord.gg/JDBeZYK)
