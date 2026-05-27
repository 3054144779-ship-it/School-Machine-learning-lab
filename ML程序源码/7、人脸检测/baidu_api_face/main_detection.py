from aip import AipFace
import base64

""" 你的 APPID AK SK """
APP_ID = '你的 App ID'
API_KEY = '你的 Api Key'
SECRET_KEY = '你的 Secret Key'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

""" 选择BASE64 """
filePath ="huge.png"
with open(filePath,"rb") as f:  
# b64encode是编码
    base64_data = base64.b64encode(f.read())
image = str(base64_data,'utf-8')

imageType = "BASE64"

groupIdList = "Sherlock"

""" 调用人脸搜索 """
print (client.search(image, imageType, groupIdList))

