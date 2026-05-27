from new_face_recognition import FaceRecognition

APP_ID = "122803311"
API_KEY = "qWfG7v7B1AAxsZhYNpwGw8Le"
SECRET_KEY = "vQFfesTst6PRXaj2iXMyLe1JA3S0wQat"

# 人脸检测
face_check = FaceRecognition(APP_ID, API_KEY, SECRET_KEY)
print(face_check.process_all_images())
print(face_check.process_all_images_face_information())
face_check.process_all_draw_images()
