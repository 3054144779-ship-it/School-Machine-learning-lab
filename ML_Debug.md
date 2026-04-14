## login.bce.baidu.com 拒绝连接

直接访问百度 AI 开放平台的**控制台入口**：`https://console.bce.baidu.com/`，手动登录后再进入短语音识别页面；

## 创建好的应用在哪里？

![image-20260408105907329](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260408105907329.png)

左侧“公有云服务”->“应用列表”。

### 安装新版SDK

如果想在全局为python安装：

```cmd
pip install baidu-aip
```

如果是python3和pip2并存：

```cmd
pip3 install baidu-aip
```

如果想在Anaconda环境下安装：

```cmd
conda install -c conda-forge baidu-aip
```

检测是否安装好：

```cmd
python -c "from aip import AipSpeech; print('OK')"
```

如果安装好了，但是依然无法导入:

```cmd
pip -verson
```

检测文件路径，找到python.exe所在的文件路径，将内核切换到该路径就可以了。

### 安装`CV2`:

```cmd
pip install opencv-python
```

## lab1 语音识别(01)：

技术文档：[语音技术](https://ai.baidu.com/ai-doc/SPEECH/0lbxfnc9b)(左侧SDK文档 $\rightarrow$ 语音识别 $\rightarrow$ API-PythonSDK)。

## lab2 图像识别(02~06)：

技术文档：[接口说明 - 图像识别 | 百度智能云文档](https://cloud.baidu.com/doc/IMAGERECOGNITION/s/4k3bcxj1m)(左侧SDK文档 $\rightarrow$ PythonSDK)

## lab3 人脸识别(07~10)：

技术文档：[人脸识别-百度智能云](https://cloud.baidu.com/doc/FACE/index.html)(左侧API文档 $\rightarrow$ 人脸识别基础接口 $\rightarrow$ REST-API-SDK $\rightarrow$ Python-SDK)

cv2技术文档：

[图像入门 - 【布客】OpenCV 4.0.0 中文翻译](https://opencv.apachecn.org/4.0.0/2.1-tutorial_py_image_display/)

[OpenCV：绘图函数 --- OpenCV: Drawing Functions](https://docs.opencv.org/4.0.0/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576)

## lab4 各种算法

PPT213页应为18

Matplotlib 文档技术文档:[快速入门指南 — Matplotlib 3.10.3 文档 - Matplotlib 绘图库](https://matplotlib.net.cn/stable/users/explain/quick_start.html#types-of-inputs-to-plotting-functions)

小问题注意：numpy2.0版本后弃用了了`mat`等，大多数不常用不安全的函数

