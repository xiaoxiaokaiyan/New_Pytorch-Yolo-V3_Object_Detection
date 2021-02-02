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
&nbsp;
<br/>



## Public Zoo:
* coco.names、yolov3.cfg、yolov3.weights。
  * YOLOv3模型文件 link1:[链接：https://pan.baidu.com/s/1M8EVfUZ7NCWV5yJMuK2LbQ 提取码：u41w](https://pan.baidu.com/s/1M8EVfUZ7NCWV5yJMuK2LbQ)
  * YOLOv3模型文件 link2:[https://pjreddie.com/darknet/yolo/]
&nbsp;
<br/>


## Detection Example

### （1）On single or multiple images 
```  
  python detect.py --images imgs --det det 
```  
`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size (using `--bs` flag) , object threshold confidence can be tweaked with flags that can be looked up with.The result is below.
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/image_result1.PNG" width = 100% height =100%  div align=left />
``` 
  通过python detect.py -h，查看参数
``` 


### （2）Speed Accuracy Tradeoff
```
python detect.py --images imgs --det det --reso 320
```
You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. The result is below.
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/image_result2.PNG" width = 100% height =100%  div align=left />


### （3）On Video
```
python video_demo.py --video video.flv  #视频格式可以自己修改，如video.avi
```
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .avi format since openCV only accepts OpenCV as the input format. The result is below.
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/video_result1.PNG" width = 100% height =100%  div align=left />
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/video_result2.PNG" width = 100% height =100%  div align=left />

### （4）Speeding up Video Inference
```
python video_demo_half.py --video video.flv  #视频格式可以自己修改，如video.avi
```
To speed video inference, you can try using the video_demo_half.py file instead which does all the inference with 16-bit half precision floats instead of 32-bit float. I haven't seen big improvements, but I attribute that to having an older card (Tesla K80, Kepler arch). If you have one of cards with fast float16 support, try it out, and if possible, benchmark it. 

### （5）On a Camera
```
python cam_phone_demo.py  #调用手机摄像头，细节看 https://github.com/xiaoxiaokaiyan/New_Opencv_Phone_Object_Detection
```
```
python cam_demo.py   #笔记本自带摄像头
```
Same as video module, but you don't have to specify the video file since feed will be taken from your camera. To be precise, feed will be taken from what the OpenCV, recognises as camera 0. The default image resolution is 160 here, though you can change it with `reso` flag.The result is below.
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/camera_phone_result1.PNG" width = 100% height =100%  div align=left />
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/camera_phone_result2.PNG" width = 100% height =100%  div align=left />
<img src="https://github.com/xiaoxiaokaiyan/New_Pytorch-Yolo-V3_Object_Detection/blob/master/results/camera_result.PNG" width = 100% height =100%  div align=left />

**You can easily tweak the code to use different weightsfiles, available at [yolo website]**(https://pjreddie.com/darknet/yolo/)


### （6）Detection across different scales
```
python detect.py --scales 1,3
```
YOLO v3 makes detections across different scales, each of which deputise in detecting objects of different sizes depending upon whether they capture coarse features, fine grained features or something between. You can experiment with these scales by the `--scales` flag. 

&nbsp;
<br/>


## References:
* [yolov3官网](https://pjreddie.com/darknet/yolo/)
* [https://github.com/ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
* [https://github.com/wmn7/ML_Practice/tree/master/2019_09_09](https://github.com/wmn7/ML_Practice/tree/master/2019_09_09)
* [python+OpenCV+YOLOv3打开笔记本摄像头模型检测](https://blog.csdn.net/weixin_43590290/article/details/100736307)
* [Windows+Cygwin+Darknet+OpenCV 简易上手实现YOLOv3目标检测](https://www.bilibili.com/video/BV1o54y1X7nk)

