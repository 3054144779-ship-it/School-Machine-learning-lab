from Image_analysis import ImageBatchProcessor
from Car_Image_analysis import CarProcessor
from Plant_Imge_analysis import PlantImageBatchProcessor
from Animal_Image_analysis import AnimalImageBatchProcessor
from Dish_Image_analysis import DishBatchProcessor

APP_ID = "7615563"
API_KEY = "9ddxioo3nfDbWV9mSJ3FPcA5"
SECRET_KEY = "o24YZIBahSKRGbkOFhvqfzNcifzAKmdw"

# 图片识别
# 创建对象
processor = ImageBatchProcessor(APP_ID, API_KEY, SECRET_KEY)
# 开始批量处理
processor.process_all_images()

# 车辆识别
car_processor = CarProcessor(APP_ID, API_KEY, SECRET_KEY)
car_processor.process_all_images()

# 植物识别
plant_processor = PlantImageBatchProcessor(APP_ID, API_KEY, SECRET_KEY)
plant_processor.process_all_images()

# 动物识别
animal_processor = AnimalImageBatchProcessor(APP_ID, API_KEY, SECRET_KEY)
animal_processor.process_all_images()

# 菜品识别
Dish_processor = DishBatchProcessor(APP_ID, API_KEY, SECRET_KEY)
Dish_processor.process_all_images()