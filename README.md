# Thyroid Segmentation with Fast-SCNN (mmsegmentation)

This repository contains the configuration file and instructions for training a Fast-SCNN model for thyroid image segmentation using the mmsegmentation framework.

## 1. Introduction

This project aims to segment thyroid regions in medical images using the Fast-SCNN architecture.  The provided `thyroid_fastscnn.py` configuration file defines the model, training pipeline, and evaluation metrics specifically tailored for this task.

## 2. Prerequisites

*   Python: 3.7+
*   PyTorch: 1.8+
*   CUDA: 10.2+ (if using GPU)
*   mmcv-full:  See installation instructions in mmsegmentation documentation.
*   mmsegmentation: Refer to the official mmsegmentation documentation for installation instructions: [https://mmsegmentation.readthedocs.io/en/latest/get_started.html](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)

    ```bash
    pip install -r requirements.txt
    pip install -U openmim
    mim install mmcv-full
    pip install -v -e .
    ```

## 3. Dataset Preparation

Prepare your thyroid image segmentation dataset in the following directory structure:

