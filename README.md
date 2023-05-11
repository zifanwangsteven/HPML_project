## Comparing the Effectiveness of Convolutional Layers in Long Term Time Series Forecasting

### Introduction

This paper examines the performance of two models, SCINet and DLinear, applied to two time series data sets for high performance machine learning. The data sets used in the study are the electricity transformer temperature dataset. Time series forecasting is a critical task in various domains, including energy and finance, where accurate predictions can lead to significant benefits. The paper presents a detailed description of the models and data sets used in the study, the training and profiling methodology, the performance tuning methodology, and the experimental results. The experimental results demonstrate the effectiveness of both models on the time series data sets, with DLinear outperforming SCINet on certain metrics. Overall, this paper provides valuable insights into the performance of convolution-based and linear models on time series data sets and their potential applications in high performance machine learning.

### Dataset

The ETT (Electricity Transformer Temperature) [data set](https://github.com/zhouhaoyi/ETDataset) is a vital metric in long-term electric power deployment, consisting of two years of data from two counties in China. The data set offers two granularity levels for examination: hour ticks (ETTh1) and minute ticks (ETTm1). Each data point in the hourly subset (ETTh1) includes the target variable "oil temperature" and six power load features. For this paper, we will be using the hourly ETTh1 data. This dataset serves as a common benchmark for testing SOTA models in the field of Long Term Time Series Forecasting (LTS). We also referenced code from [Informer](https://github.com/zhouhaoyi/Informer2020) model for the DataLoaders for ETTh data.

### Repo Organization & Running the Code

We organized all code into a single Colab notebook which can be found here [notebook](https://colab.research.google.com/drive/1CbW_ecoqsgGxuSBfY8MlouSrE7_YrM5M#scrollTo=L7uKJbrCq630). The data used for this project can be found at `data/ETTh1.csv`. 

To run this Colab notebook

* first save a local copy of the notebook in Drive
* upload the dataset to Drive
* give authorization to access Drive as prompted

A walkthrough of the code sections

* `SCINet Model Definition` and `DLinear Model Definition`
  * contains the buildings blocks for the SCINet model and DLinear model, referencing code from [SCINet](https://github.com/cure-lab/SCINet) and [DLinear](https://github.com/cure-lab/LTSF-Linear) GitHub repos.
* `DataLoading and Preprocessing`
  * defines custom dataloader / parser for ETTh dataset
  * defines utilities for loading / saving models
* `Experiment`
  * `Training Definition`
    * defines training loop and hyperparameters
  * `Hyperparameter Sweep`
    * defines hyperparameter sweep with Weights & Biases. Notice the appeal of the DLinear model is that it generalizes well without hyperparameters tune (Except for batchsize and learning rate, which can be empirically set)
  * `PyTorch Profiling`
    * profiles CPU and GPU activity for both models for 20 iterations in training mode and inference mode
  * `Pruning`
    * defines model pruning for CPU

### Results & Conclusions