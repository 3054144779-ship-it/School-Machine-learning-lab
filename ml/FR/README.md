# FR (Face Analysis) 人脸识别技术文档

## 1. 项目简介

本项目是专门针对人脸检测与属性分析开发的工具。它集成了 **百度 AI 开放平台** 的人脸检测能力，能够自动识别图像中的人脸，提取性别、年龄等关键信息，并利用 `OpenCV` 在原图上进行实时标注和可视化展示。

## 2. 基础架构
项目结构由一个本地库、一个核心处理类和一个程序入口组成：

```text
/IA ———|
        |——/image                  # 本地图库：存放待处理的图片 (.jpg, .png, .jpeg, .bmp)
        |
        |——main.py                 # 程序入口：负责实例化对象并启动识别逻辑
        |
        |——Face_Image_analysis.py  # 核心文件：包含 FaceRecognition 类及其所有处理方法
```

## 3. 环境准备
项目依赖百度 AI SDK 和 OpenCV 库，请确保已安装以下依赖：

```bash
pip install baidu-aip opencv-python
```

## 4. 核心类：FaceRecognition

所有功能均封装在 `FaceRecognition` 类中。

### 4.1 初始化参数

- `app_id` / `api_key` / `secret_key`: 百度 AI 平台的授权凭证。
- `image_path`: 指定存放图片的文件夹，默认为 `"image"`。

### 4.2 核心方法说明

| 方法名                                | 功能描述                                                     |
| :------------------------------------ | :----------------------------------------------------------- |
| `get_file_content`                    | 读取本地图片并转换为 **BASE64** 编码，供接口调用。           |
| `recognize_base64`                    | **单图检测**：通过图片编码识别单张本地图片的人脸属性。       |
| `detect_face`                         | **通用识别**：支持 BASE64、URL、FACE_TOKEN 三种模式的识别。  |
| `process_all_images`                  | **批量获取原始数据**：遍历文件夹，返回所有图片的 API 原始 JSON 结果。 |
| `process_all_images_face_information` | **批量获取文本信息**：提取所有图片的性别、年龄，返回格式化的字符串列表。 |
| `detect_and_draw`                     | **单图可视化**：识别并在图片上画出人脸框及标注性别、年龄文字。 |
| `process_all_draw_images`             | **批量可视化**：循环处理文件夹内所有图片并弹窗显示识别结果。 |

## 5. 核心逻辑流程
1. **数据准备**：程序通过 `os.listdir` 自动扫描 `/image` 文件夹下的图片文件。
2. **频率控制**：类中内置了 `API_DELAY = 1`（秒），在每次请求接口前会执行 `time.sleep`，有效防止触发免费接口的 QPS 限制。
3. **坐标换算与绘制**：从 API 返回的 `location`（left, top, width, height）中提取坐标，利用 OpenCV 绘制绿色矩形框，并在框上方标注红色信息文本。
4. **异常处理**：内置了 API 错误校验和“非人脸”校验（`Is not face`），确保程序在遇到模糊图片或接口报错时不会中断。

## 6. 使用示例 (main.py)
```python
from Face_Image_analysis import FaceRecognition

# 初始化
fr = FaceRecognition(app_id='你的ID', api_key='你的Key', secret_key='你的Secret')

# 场景 1：获取所有图片的文字描述
info = fr.process_all_images_face_information()
print(info)

# 场景 2：批量在图片上画框并显示
fr.process_all_draw_images()
```

---
*文档版本：V1.0*
*开发者：[花火]*