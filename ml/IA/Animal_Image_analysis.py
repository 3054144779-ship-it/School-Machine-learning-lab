import time
from Image_analysis import ImageBatchProcessor

class AnimalImageBatchProcessor(ImageBatchProcessor) :
    MIN_SCORE = 0.9

    def is_animal(self, file_path) :
        # 获取图片路径
        img_data = self.get_file_content(file_path)
        
        # 控制QPS
        time.sleep(self.API_DELAY)
        
        # 获取动物信息
        result = self.client.animalDetect(img_data)

        # 获取动物信息列表
        result_animal_list = result.get("result", [])
        if len(result_animal_list) == 0 :
            return False

        # 如果当前置信度小于最小置信度的话就不行
        score = float(result_animal_list[0].get("score", 0.0))
        if score <= AnimalImageBatchProcessor.MIN_SCORE :
            return False        

        # 名字不是动物
        name = result_animal_list[0].get("name", "非动物")
        return "非动物" not in name

    def recognize_image(self, file_path):
        if not self.is_animal(file_path) :
            print("Is not animal")
            return

        # 获取动物图片信息
        img = self.get_file_content(file_path)
        return self.client.animalDetect(img)