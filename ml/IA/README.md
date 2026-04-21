# IA (Image Analysis) 图像识别项目技术文档

## 1. 项目简介

本项目是一款基于 **百度 AI 开放平台** 图像识别接口开发的自动化处理工具。通过封装百度云提供的视觉能力，实现对本地图库中**动物、车辆、菜品及植物**的快速批量识别与分类。

## 2. 基础架构
项目采用模块化设计，确保了代码的高复用性和可维护性。其目录结构如下：

```text
/IA ———|
        |——/image                  # 输入文件夹：存放待识别的 (.jpg, .png) 图片
        |
        |——main.py                 # 程序主入口：负责整体业务流程调度
        |
        |——Image_analysis.py       # 基类：封装百度 AipImage 基础配置与通用方法
        |
        |——Animal_Image_analysis.py  # 子类：处理动物图像识别逻辑
        |——Car_Image_analysis.py     # 子类：处理车辆图像识别逻辑
        |——Dish_Image_analysis.py    # 子类：处理菜品图像识别逻辑
        |——Plant_Image_analysis.py   # 子类：处理植物图像识别逻辑
```

## 3. 环境准备
在使用本项目前，请确保 Python 环境中已安装百度 AI SDK (`baidu-aip`)。

### 安装命令：
* **全局安装：**
  ```bash
  pip install baidu-aip
  ```
* **Python3 环境：**
  ```bash
  pip3 install baidu-aip
  ```
* **Anaconda 环境：**
  ```bash
  conda install -c conda-forge baidu-aip
  ```

## 4. 核心类说明
每个业务模块均封装了独立的批处理器类：

| 模块文件                   | 核心类名                    | 职责描述                 |
| :------------------------- | :-------------------------- | :----------------------- |
| `Animal_Image_analysis.py` | `AnimalImageBatchProcessor` | 批量调用百度动物识别接口 |
| `Car_Image_analysis.py`    | `CarProcessor`              | 识别车辆品牌、型号等特征 |
| `Dish_Image_analysis.py`   | `DishBatchProcessor`        | 识别菜品名称、热量等信息 |
| `Plant_Image_analysis.py`  | `PlantImageBatchProcessor`  | 识别自然界植物种类       |

## 5. 核心功能逻辑

项目核心操作由以下三个关键方法组成：

1. **`get_file_content(file_path)`**
   - **功能**：读取本地图片文件。
   - **输出**：返回二进制流数据，供 API 调用。

2. **`recognize_image(image_data)`**
   - **功能**：将二进制图片数据发送至百度云接口。
   - **输出**：返回包含识别结果（如 Top 5 可能性及置信度）的 JSON 对象。

3. **`process_all_images()`**
   - **功能**：自动化批处理逻辑。
   - **流程**：遍历 `/image` 文件夹 -> 调用识别接口 -> 格式化输出/保存识别结果。

## 6. 使用注意事项
- **鉴权配置**：运行前需在基类或配置文件中填入百度 AI 平台的 `AppID`, `API Key` 和 `Secret Key`。
- **并发限制**：若使用免费额度接口，请注意控制 `process_all_images` 的调用频率（QPS）。
- **图片限制**：图片大小通常建议在 4MB 以内，且最短边不小于 15px。

---
*文档版本：V1.0*
*开发者：[花火]*