import base64
import os
import time
import cv2
from aip import AipFace

class FaceRecognition :
    API_DELAY = 1
    # image_path 传入BASE64字符串或URL字符串或FACE_TOKEN字符串
    # 创建密钥
    def __init__(self, app_id, api_key, secret_key, image_path = "image"):
        self.client = AipFace(app_id, api_key, secret_key)
        self.image_path = image_path
        self.image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    # 读取图片并转 BASE64
    def get_file_content(self, file_path) :
        with open(file_path, "rb") as fp :
            content = fp.read()
            return base64.b64encode(content).decode("utf-8")

    # base64 检测单张图片
    def recognize_base64(self, file_path):
        # 获取图片的 base64 值
        img_base = self.get_file_content(file_path)
        # 控制QPS
        time.sleep(FaceRecognition.API_DELAY)
        options = {"face_field": "gender,age,landmark,location", "max_face_num": 9, "face_type": "LIVE"}
        # 返回检测图片的信息        
        return self.client.detect(img_base, "BASE64", options)

    # URL识别网络图片
    def recognize_url(self, image_url):
        return self.detect_face(image_url, "URL")

    # FACE_TOKEN识别人脸令牌（复用已检测的人脸）
    def recognize_token(self, face_token):
        return self.detect_face(face_token, "FACE_TOKEN")
    
    # 通用人脸识别
    def detect_face(self, image: str, image_type: str) :
        valid_types = ["BASE64", "URL", "FACE_TOKEN"]
        if image_type not in valid_types:
            raise ValueError(f"图片类型错误！仅支持：{valid_types}")
        
        # 控制接口调用频率
        time.sleep(FaceRecognition.API_DELAY)
        
        # 调用百度人脸检测接口
        return self.client.detect(image, image_type)

    # 获取人脸信息(是否存在，数量，性别, 年龄)
    def process_all_images_face_information(self) :
        face_all_information = self.process_all_images()
        output_information = []
        for result in face_all_information :
            # 脸的数量
            face = result.get("result")

            # 如果 face 是 None 报错
            if not face:
                output_information.append("Is not face (API Error or No Result)")
                continue

            face_num = int(face.get("face_num", 0))
            # 没有脸就返回 "Is not face"            
            if face_num == 0 :
                output_information.append(f"Is not face")
                continue

            # 获取性别
            face_list = face.get("face_list", [])
            gender = "未知"
            if face_list:
                gender = face_list[0].get("gender", {}).get("type", "未知")
            
            # 获取年龄
            age = int(face_list[0].get("age", 0))

            output_information.append(f"count face: {face_num},\n sex: {gender},\n age:{age}")
        
        return output_information        
        
    # 批量处理
    def process_all_images(self) :
        face_all_information = []
        for file_name in os.listdir(self.image_path) :
            img_data = os.path.join(self.image_path, file_name)
            if file_name.lower().endswith(self.image_exts):
                img = self.recognize_base64(img_data)
                face_all_information.append(img)

        return face_all_information

    # 绘制单张图片 
    def detect_and_draw(self, file_path) :
        # 读取图片
        img = cv2.imread(file_path, 1)        
        if img is None:
            print("图片读取失败！")
            return
        
        # 获取人脸检测结果
        image_information = self.recognize_base64(file_path)
        result = image_information.get('result', {})
        face_list = result.get('face_list', [])
        
        # 遍历所有人脸
        for face in face_list:
            left   = int(face['location']['left'])      # 左上角X
            top    = int(face['location']['top'])       # 左上角Y
            width  = int(face['location']['width'])
            height = int(face['location']['height'])
            
            # 计算右下角坐标
            right  = left + width
            bottom = top + height

            gender = face['gender']['type']
            age = int(face['age'])
            text = f"Gender: {gender}  Age: {age}"

            # 画绿色人脸框
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # 画红色文字(图片，文本，左下角坐标，文字类型，缩放，颜色，粗细)
            cv2.putText(img, text, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 创建图片窗口
        cv2.imshow("Face Result", img)
        # 按任意键切换下一张
        cv2.waitKey(0)
        # 删除改张
        cv2.destroyAllWindows()

    # 批量绘制
    def process_all_draw_images(self) :
        for file_name in os.listdir(self.image_path) :
            img_data = os.path.join(self.image_path, file_name)
            if file_name.lower().endswith(self.image_exts):
                img = self.detect_and_draw(img_data)
