import base64
import os
import time
import cv2
from aip import AipFace

class FaceRecognition:
    API_DELAY = 1  # 接口调用延迟，控制 QPS
    
    def __init__(self, app_id, api_key, secret_key, image_path="image"):
        self.client = AipFace(app_id, api_key, secret_key)
        self.image_path = image_path
        self.image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    # 读取图片并转 BASE64
    def get_file_content(self, file_path):
        with open(file_path, "rb") as fp:
            content = fp.read()
            return base64.b64encode(content).decode("utf-8")

    # base64 检测单张图片
    def recognize_base64(self, file_path, options=None):
        img_base = self.get_file_content(file_path)
        time.sleep(self.API_DELAY) # 控制QPS
        
        if options:
            return self.client.detect(img_base, "BASE64", options)
        return self.client.detect(img_base, "BASE64")

    # URL识别网络图片
    def recognize_url(self, image_url):
        return self.detect_face(image_url, "URL")

    # FACE_TOKEN识别人脸令牌
    def recognize_token(self, face_token):
        return self.detect_face(face_token, "FACE_TOKEN")
    
    # 通用人脸识别
    def detect_face(self, image: str, image_type: str):
        valid_types = ["BASE64", "URL", "FACE_TOKEN"]
        if image_type not in valid_types:
            raise ValueError(f"图片类型错误！仅支持：{valid_types}")
        
        time.sleep(self.API_DELAY)
        return self.client.detect(image, image_type)

    # 获取批量人脸信息文本
    def process_all_images_face_information(self):
        options = {"face_field": "gender,age,landmark,location", "max_face_num": 9, "face_type": "LIVE"}
        face_all_information = self.process_all_images(options) 
        output_information = []
        
        for result in face_all_information:
            # 增加 API 错误校验
            if result.get("error_code") != 0:
                output_information.append(f"API Error: {result.get('error_msg')}")
                continue

            face = result.get("result")
            if not face:
                output_information.append("Is not face")
                continue

            face_num = int(face.get("face_num", 0))
            if face_num == 0:
                output_information.append("Is not face")
                continue

            face_list = face.get("face_list", [])
            for i, f in enumerate(face_list):
                gender = f.get("gender", {}).get("type", "未知")
                age = int(f.get("age", 0))
                output_information.append(f"Face {i+1} -> Total: {face_num}, Gender: {gender}, Age: {age}")
        
        return output_information        
        
    # 批量处理并返回 API 原始数据
    def process_all_images(self, options=None):
        face_all_information = []
        # 确保文件夹存在
        if not os.path.exists(self.image_path):
            print(f"文件夹 {self.image_path} 不存在！")
            return face_all_information

        for file_name in os.listdir(self.image_path):
            img_data = os.path.join(self.image_path, file_name)
            if file_name.lower().endswith(self.image_exts):
                img_result = self.recognize_base64(img_data, options)
                face_all_information.append(img_result)

        return face_all_information

    # 绘制单张图片 
    def detect_and_draw(self, file_path):
        img = cv2.imread(file_path, 1)        
        if img is None:
            print(f"图片读取失败: {file_path}")
            return
        
        options = {"face_field": "gender,age"}
        image_information = self.recognize_base64(file_path, options)
        
        if image_information.get('error_code') != 0:
            print(f"识别失败 [{file_path}]: {image_information.get('error_msg')}")
            return

        result = image_information.get('result', {})
        if not result:
            print(f"未检测到人脸: {file_path}")
            return

        face_list = result.get('face_list', [])
        
        for face in face_list:
            # 获取坐标
            location = face.get('location', {})
            left   = int(location.get('left', 0))
            top    = int(location.get('top', 0))
            width  = int(location.get('width', 0))
            height = int(location.get('height', 0))
            
            right  = left + width
            bottom = top + height

            # 获取性别和年龄
            gender = face.get('gender', {}).get('type', 'Unknown')
            age = int(face.get('age', 0))
            text = f"Gender: {gender}  Age: {age}"

            # 画框和文字
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, text, (left, max(top - 10, 10)), # 确保文字不会跑到图片外面
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示图片
        cv2.imshow("Face Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 批量绘制
    def process_all_draw_images(self):
        if not os.path.exists(self.image_path):
            print(f"文件夹 {self.image_path} 不存在！")
            return
            
        for file_name in os.listdir(self.image_path):
            img_data = os.path.join(self.image_path, file_name)
            if file_name.lower().endswith(self.image_exts):
                self.detect_and_draw(img_data)