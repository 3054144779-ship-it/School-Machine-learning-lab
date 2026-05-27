from aip import AipImageClassify

""" 你的 APPID AK SK """
APP_ID = '24910506'
API_KEY = 'NvuW77M0jUqq2edREd1rgqYf'
SECRET_KEY = 'flFIKpBqtdMFUadu21lC0wCXqkKfTY31'

client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('yumi.png')

""" 调用植物识别 """
print (client.plantDetect(image))

""" 如果有可选参数 """
options = {}
options["baike_num"] = 1

""" 带参数调用植物识别 """
print (client.plantDetect(image, options))

