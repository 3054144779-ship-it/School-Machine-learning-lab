from aip import AipFace
import base64

""" 你的 APPID AK SK """
APP_ID = '24777669'
API_KEY = 'woke0A2nZvDVoOnt78b4ZkdS'
SECRET_KEY = 'DDCCQvLa9GZd1npYqNXctt3P6Fx9LHV4'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

# """ 选择URL """
# image = "http://img31.mtime.cn/pi/2015/03/06/151408.15113572.jpg"
# imageType = "URL"

""" 选择BASE64 """
filePath ="huge.png"
with open(filePath,"rb") as f:
# b64encode是编码
    base64_data = base64.b64encode(f.read())
image = str(base64_data,'utf-8')
imageType = "BASE64"

""" 调用人脸检测 """
client.detect(image, imageType)

""" 如果有可选参数 """
options = {}
options["face_field"] = "age,beauty"
options["max_face_num"] = 1
options["face_type"] = "LIVE"

""" 带参数调用人脸检测 """
print (client.detect(image, imageType, options))


