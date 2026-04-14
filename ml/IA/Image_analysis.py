import os
from aip import AipImageClassify

class ImageBatchProcessor:
    # 0.5s 响应一次
    API_DELAY = 0.5
    
    # 初始化密钥
    def __init__(self, app_id, api_key, secret_key, folder_path="image"):
        self.client = AipImageClassify(app_id, api_key, secret_key)
        self.folder_path = folder_path 
        self.image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    # 读取单个文件
    def get_file_content(self, file_path):
        with open(file_path, "rb") as fp:
            return fp.read()
    
    # 处理单张图片
    def recognize_image(self, file_path):
        img_data = self.get_file_content(file_path)
        return self.client.advancedGeneral(img_data)

    # 处理一组图片
    def process_all_images(self):
        # 遍历文件夹下的所有文件
        for file_name in os.listdir(self.folder_path):
            # 拼接路径
            file_path = os.path.join(self.folder_path, file_name)
            
            # 只处理允许格式图片
            if file_name.lower().endswith(self.image_exts):
                print(f"正在处理：{file_name}")
                result = self.recognize_image(file_path)
                print("识别结果：", result)
