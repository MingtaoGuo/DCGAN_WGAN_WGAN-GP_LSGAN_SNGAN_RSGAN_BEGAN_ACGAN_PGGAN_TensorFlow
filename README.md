# DCGAN_LSGAN_WGAN_WGAN-GP_SNGAN
Implementation of some different variants of GANs 

Introduction
--------------

This code is mainly implement some basic GANs about 'DCGAN', 'WGAN', 'WGAN-GP', 'LSGAN', 'SNGAN'. 

More details of these GANs, please see follow papers:

1. DCGAN: [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/pdf/1511.06434.pdf%C3%AF%C2%BC%E2%80%B0)

2. WGAN: [Wasserstein gan](https://arxiv.org/pdf/1701.07875.pdf?__hstc=200028081.1bb630f9cde2cb5f07430159d50a3c91.1524009600081.1524009600082.1524009600083.1&__hssc=200028081.1.1524009600084&__hsfp=1773666937)

3. WGAN-GP: [Improved training of wasserstein gans](https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)

4. LSGAN: [Least Squares Generative Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf)

5. SNGAN: [Spectral normalization for generative adversarial networks](https://arxiv.org/pdf/1802.05957.pdf)

How to use 
----------
Firstly, you should download the data 'facedata.mat' from [BaiduYun](), then put the file 'facedata.mat' into the folder 'TrainingSet'.

Necessory python packages are as follow:

1. tensorflow

2. pillow

3. scipy

4. numpy

Results of this code
--------------------
<div align=center><img src="https://github.com/MingtaoGuo/DCGAN_LSGAN_WGAN_WGAN-GP_SNGAN/raw/master/Image/DCGAN.jpg"/></div>

Compare WGAN, WGAN-GP, SNGAN of different iteration
-----------------------------------------------------
![](https://github.com/MingtaoGuo/DCGAN_LSGAN_WGAN_WGAN-GP_SNGAN/raw/master/Image/GAN.jpg)
