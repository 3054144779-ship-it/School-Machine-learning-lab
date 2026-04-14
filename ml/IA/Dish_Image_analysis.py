import time
from Image_analysis import ImageBatchProcessor

class DishBatchProcessor(ImageBatchProcessor) :
    MIN_PROBABILITY = 0.65

    def is_dish(self, file_path) :
        # 获取图片路径
        img_data = self.get_file_content(file_path)
        # 设置 QPS 延迟
        time.sleep(self.API_DELAY)
        # 获取菜品信息
        result = self.client.dishDetect(img_data)
        # print(reslut)
        # 获取菜品信息列表
        result_dish_list = result.get("result", [])
        if len(result_dish_list) == 0 :
            return False
        
        # 菜品一定要有卡路里
        has_calorie = result_dish_list[0].get("has_calorie", False)
        if not has_calorie :
            return False
        
        # 概率
        probability = float((result_dish_list[0].get("probability", 0.0)))
        if probability <= DishBatchProcessor.MIN_PROBABILITY :
            return False
        
        name = result_dish_list[0].get("name", "非菜")
        return "非菜" not in name

    def recognize_image(self, file_path):
        if not self.is_dish(file_path) :
            print("Is not dish")
            return  
        
        img = self.get_file_content(file_path)
        return self.client.dishDetect(img)