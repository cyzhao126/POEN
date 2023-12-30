# Trustworthy Video-based Sewer Inspection via Support Clip-based Pareto-optimal Evidential Network （submitted to ICME 2024）

## Introduction

This is the official repository for *Trustworthy Video-based Sewer Inspection via Support Clip-based Pareto-optimal Evidential Network*. In this repository, we release the splits of training set and test set for $\tau_{cls}$ and $\tau_{ood}$, as well as the source code.

## Installation

1、Create conda environment:
```
$ conda create -n POEN python=3.7
$ conda activate POEN
```
2、This code was implemented with python 3.7, torch 1.10.0+cu113
```
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
3、 Install other dependency
```
$ pip install -r requirements.txt
```

## Dataset

For data preparation of VideoPipe, we follow the dataset used in "VideoPipe Challenge @ ICPR2022，Challenge on Real-World Video Understanding for Urban Pipe Inspection" |[pdf](https://arxiv.org/pdf/2210.11158), [github](https://videopipe.github.io/)|. You can use the above official link to download VideoPipe dataset, since the annotations of testing set are not public, we conduct the experiments focused on the training sets. 

Our split files for $\tau_{cls}$ and $\tau_{ood}$ are provided in `/data`.

## Train 
The implementation of video swin transformer is based on MMAction2 project |[github](https://github.com/open-mmlab/mmaction2)|. The config file of this work could be found at `/configs`, config files show all detailed settings of the model. Some code designed for POEN are provide in `/mmaction`, which should overwrite its source files after downloading mmaction project.

In order to implement pareto optimization for $\tau_{cls}$ and $\tau_{ood}$, we have to modify the "optimizer.py" in "mmcv". We follow the method used in "Multi-Task Learning as Multi-Objective Optimization" |[pdf](https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html), [github](https://github.com/isl-org/MultiObjectiveOptimization)|. Furthermore, we provide our "optimizer.py" in `/mmcv` for video sewer inspection, which should overwrite its source files after installing the "mmcv" library.

1. Download the pretrained model from MMAction2 project and put it under the directory `/data`. The model we used is "swin_base_patch244_window877_kinetics600_22k.pth".
2. Train a model for $\tau_{cls}$ with 2 GPUs, by

   ```shell
   bash tools/dist_train1.sh configs/multiview_track1.py 2 --cfg-options load_from='data/swin_base_patch244_window877_kinetics600_22k.pth'
   ```
3. Train a model for $\tau_{ood}$ with 2 GPUs, by
   ```shell
   bash tools/dist_train1.sh configs/swin_edl_OOD_least4.py 2 --cfg-options load_from='data/swin_base_patch244_window877_kinetics600_22k.pth'
   ```
   
In default, checkpoints will be saved every 5 epochs.

## Test
To test on VideoPipe, you can run
   ```shell
   bash tools/dist_test.sh
   ```
