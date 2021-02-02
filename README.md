# 心得：**Yolov3 图片-视频-摄像头物体检测 **

## News
* this code 
* [Opencv Version(延迟大)](https://github.com/xiaoxiaokaiyan/New_Opencv_Phone_Object_Detection)
&nbsp;
<br/>


## Dependencies:
* &gt; GeForce GTX 1660TI
* Windows10
* python==3.6.12
* torch==1.0.0
* GPU环境安装包，下载地址：https://pan.baidu.com/s/14Oisbo9cZpP7INQ6T-3vwA 提取码：z4pl （网上找的）
```
  Anaconda3-5.2.0-Windows-x86_64.exe
  cuda_10.0.130_411.31_win10.exe
  cudnn-10.0-windows10-x64-v7.4.2.24.zip
  h5py-2.8.0rc1-cp36-cp36m-win_amd64.whl
  numpy-1.16.4-cp36-cp36m-win_amd64.whl
  tensorflow_gpu-1.13.1-cp36-cp36m-win_amd64.whl
  torch-1.1.0-cp36-cp36m-win_amd64.whl
  torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```
<br/>



## Public Zoo:
* coco.names、yolov3.cfg、yolov3.weights。
  * YOLOv3模型文件 link1:[链接：https://pan.baidu.com/s/1M8EVfUZ7NCWV5yJMuK2LbQ 提取码：u41w](https://pan.baidu.com/s/1M8EVfUZ7NCWV5yJMuK2LbQ)
  * YOLOv3模型文件 link2:[https://pjreddie.com/darknet/yolo/]
<br/>


## Detection Example
### （1）On single or multiple images 
```  
  python detect.py --images imgs --det det 
```  
`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size (using `--bs` flag) , object threshold confidence can be tweaked with flags that can be looked up with.The result is below.
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/image_result1.PNG" width = 50% height =50%  div align=left />
``` 
  通过python detect.py -h，查看参数
``` 
### （2）Speed Accuracy Tradeoff
```
python detect.py --images imgs --det det --reso 320
```
You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. The result is below.
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/image_result2.PNG" width = 50% height =50%  div align=left />







## References:
* [WGAN-GP训练流程---对本代码的详细讲解](https://mathpretty.com/11133.html)
* [https://github.com/wmn7/ML_Practice/tree/master/2019_09_09](https://github.com/wmn7/ML_Practice/tree/master/2019_09_09)
* [RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 544 and 1935 in dimension 2 at ../aten/src/TH/generic/THTensor.cpp:711](https://www.cnblogs.com/zxj9487/p/11531888.html)
* [PyTorch修炼二、带你详细了解并使用Dataset以及DataLoader](https://zhuanlan.zhihu.com/p/128679151)
* [更多GAN变种的实现：https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2](https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2)
* [深度学习与TensorFlow 2入门实战（完整版）](https://www.bilibili.com/video/BV1HV411q7xD?from=search&seid=14089320887830328110)---龙曲良
* [https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) ---[Joseph Rocca](https://medium.com/@joseph.rocca)
* [https://zhuanlan.zhihu.com/p/24767059](https://zhuanlan.zhihu.com/p/24767059)
* [更多GAN变种的论文：https://github.com/hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)
* [https://reiinakano.github.io/gan-playground/在线构建GAN](https://reiinakano.github.io/gan-playground/)














# A PyTorch implementation of a YOLO v3 Object Detector

[UPDATE] : This repo serves as a driver code for my research. I just graduated college, and am very busy looking for research internship / fellowship roles before eventually applying for a masters. I won't have the time to look into issues for the time being. Thank you.


This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2). One of the goals of this code is to improve
upon the original port by removing redundant parts of the code (The official code is basically a fully blown deep learning 
library, and includes stuff like sequence models, which are not used in YOLO). I've also tried to keep the code minimal, and 
document it as well as I can. 

### Tutorial for building this detector from scratch
If you want to understand how to implement this detector by yourself from scratch, then you can go through this very detailed 5-part tutorial series I wrote on Paperspace. Perfect for someone who wants to move from beginner to intermediate pytorch skills. 

[Implement YOLO v3 from scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

As of now, the code only contains the detection module, but you should expect the training module soon. :) 





## Detection Example

![Detection Example](https://i.imgur.com/m2jwneng.png)
## Running the detector

### Speed Accuracy Tradeoff
You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. 

```
python detect.py --images imgs --det det --reso 320
```

### On Video
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .avi format
since openCV only accepts OpenCV as the input format. 

```
python video_demo.py --video video.avi
```

Tweakable settings can be seen with -h flag. 

### Speeding up Video Inference

To speed video inference, you can try using the video_demo_half.py file instead which does all the inference with 16-bit half 
precision floats instead of 32-bit float. I haven't seen big improvements, but I attribute that to having an older card 
(Tesla K80, Kepler arch). If you have one of cards with fast float16 support, try it out, and if possible, benchmark it. 

### On a Camera
Same as video module, but you don't have to specify the video file since feed will be taken from your camera. To be precise, 
feed will be taken from what the OpenCV, recognises as camera 0. The default image resolution is 160 here, though you can change it with `reso` flag.

```
python cam_demo.py
```
You can easily tweak the code to use different weightsfiles, available at [yolo website](https://pjreddie.com/darknet/yolo/)

NOTE: The scales features has been disabled for better refactoring.
### Detection across different scales
YOLO v3 makes detections across different scales, each of which deputise in detecting objects of different sizes depending upon whether they capture coarse features, fine grained features or something between. You can experiment with these scales by the `--scales` flag. 

```
python detect.py --scales 1,3
```


