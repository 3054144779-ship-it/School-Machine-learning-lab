import os
import time
from Image_analysis import ImageBatchProcessor

class CarProcessor(ImageBatchProcessor) :
    # 设置最小置信度
    MIN_SCORE = 0.9

    def is_car(self, file_path) :
        # 获取图片信息
        img_data = self.get_file_content(file_path)
        
        # 控制QPS
        time.sleep(self.API_DELAY)
        
        # 获取车辆信息
        result = self.client.carDetect(img_data)
        # 获取车辆列表        
        car_result_list = result.get("result", [])
        if len(car_result_list) == 0:
            return False
        
        # 置信度要大于 0.9
        car_score = float(car_result_list[0].get("score", 0.0))
        if car_score <= CarProcessor.MIN_SCORE :
            return False

        # 车辆应该有高度和宽度
        car_location_result = result.get("location_result")
        car_height = car_location_result.get("height", 0)
        car_width = car_location_result.get("width", 0)

        if car_height <= 50 and car_width <= 50 :
            return False

        # 检查是否有非车类
        first_car_name = car_result_list[0].get("name", "")
        return "非车类" not in first_car_name
    
    def recognize_image(self, file_path) :
        # 先判断是不是车
        if not self.is_car(file_path) :
            print("Is not car")
            return
         
        # 获取图片信息
        img = self.get_file_content(file_path)
        # 返回车型
        return self.client.carDetect(img)
