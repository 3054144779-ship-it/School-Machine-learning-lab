import os
import time
from Image_analysis import ImageBatchProcessor

class PlantImageBatchProcessor(ImageBatchProcessor) :
    # 植物的最小置信度
    MIN_SCORE = 0.75
    
    def is_plant(self, file_path) :
        # 获取图片信息
        img_data = self.get_file_content(file_path)

        # 控制 QPS
        time.sleep(self.API_DELAY)

        # 获取植物信息
        result = self.client.plantDetect(img_data)
        # 获取植物信息列表
        result_plant_list = result.get("result", [])

        if len(result_plant_list) == 0 :
            return False
        
        # 植物的置信度不能低于最小置信度
        score = float(result_plant_list[0].get("score", 0.0))
        if score <= PlantImageBatchProcessor.MIN_SCORE :
            return False

        name = result_plant_list[0].get("name", "非植物类")
        return "非植物类" not in name
 
    def recognize_image(self, file_path) :
        if not self.is_plant(file_path) :
            print("Is not plant")
            return
        # 获取图片信息
        plant_image = self.get_file_content(file_path)
        # 返回植物图片信息
        return self.client.plantDetect(plant_image)
