# from    aip import AipFace
# import base64
# """
# 人脸检测方式:
# 在百度人脸识别API上,有3种方式上传人脸图片:
# image = "取决于iamge_type参数,传入BASE64字符串或URL字符串或FACE_TOKEN字符串"
# imageType = "image_type参数"
# 即上传的图像3种方式:BASE64字符串,URL字符串,FACE_TOKEN字符串
# """
# """
# BASE64:图片的base64值,base64编码后的图片数据,需urlencode,编码后的图片大小不超过2M
# URL:图片的URL地址(可能由于网络等原因导致下载图片时间过长)
# FACE_TOKEN:人脸图片的唯一标识,调用人脸检测接口时,会为每个人脸图片赋予一个唯一的FACE_TOKEN,同一张图片多次检测得到的FACE_TOKEN是同一个
# """
# """
# 首先使用URL方式进行检测;当然,除了能直接返回人脸检测结果之外,还可以添加参数以及返回更为详细的信息
# """
# """APPID,AK,SK"""
# APP_ID = '23555132'
# API_KEY = '63S4vBrBliyEj6jazb3Liaiw'
# SECRET_KEY = 'FoVExS8rwpMyGrGGfQBHFDCCsTgeajMS'
# client = AipFace(APP_ID,API_KEY,SECRET_KEY)
# # # 该识别主要用于URL方式上传网络图片,从而进行人脸检测
# # image = "https://gss0.baidu.com/70cFfyinKgQFm2e88IuM_a/forum/w=580/sign=dfecbb34bb3533faf5b6932698d3fdca/f6d06efcc3cec3fd3a37954fdc88d43f869427c5.jpg"  #用百度图片里的图,这张就行
# # imageType = 'URL'
# # """如果有可选参数"""
# # options = {}
# # options['face_field'] = 'age,beauty'
# # options['max_face_num'] = 1
# # options['face_type'] = 'LIVE'
# # print(client.detect(image,imageType,options))
# # # 使用BASE64方式进行人脸检测
# # filePath = '叶奈法.jpg'
# # with    open(filePath,'rb') as f:
# #     # b64encode是编码
# #     base64_data = base64.b64encode(f.read())
# # image = str(base64_data,'utf-8')
# # imageType = 'BASE64'
# # """如果有可选参数"""
# # options = {}
# # options['face_field'] = 'age,beauty'
# # options['max_face_num'] = 1
# # options['face_type'] = 'LIVE'
# # """带参数调用人脸检测"""
# # print(client.detect(image,imageType,options))
# """选择BASE64"""
# filePath = '花生/花生8.jpg'#'叶奈法.jpg'
# with    open(filePath,'rb') as f:
#     # b64encode是编码
#     base64_data = base64.b64encode(f.read())
# image = str(base64_data,'utf-8')
# imageType = "BASE64"
# groupIdList = "Sherlock"
# """调用人脸搜索"""
# print(client.search(image,imageType,groupIdList))
from    aip import  AipFace
import base64
import cv2
from    matplotlib  import pyplot   as plt
# 填入百度AI开放平台密钥
APP_ID = '24777669'
API_KEY = 'woke0A2nZvDVoOnt78b4ZkdS'
SECRET_KEY = 'DDCCQvLa9GZd1npYqNXctt3P6Fx9LHV4'
# 实例化人脸的类
client = AipFace(APP_ID,API_KEY,SECRET_KEY)
# 加载图片并编码
img0 = cv2.imread('Sherlock_Wahson0.png',cv2.IMREAD_COLOR)  #必须是英文路径
print(img0)
img = cv2.imencode('.jpg',img0)[1]
image = str(base64.b64encode(img))[2:-1]
imageType = 'BASE64'
# 加入可选参数
options = {}
options['face_field'] = 'age,beauty,gender'
options['max_face_num'] = 9
options['face_type'] = 'LIVE'
# 识别
result = client.detect(image,imageType,options)
# 绘制出来
if  result['error_msg'] == 'SUCCESS':
    print(result)
    for i   in  range(result['result']['face_num']):
        x = result['result']['face_list'][i]['location']['left']
        y = result['result']['face_list'][i]['location']['top']
        w = result['result']['face_list'][i]['location']['width']
        h = result['result']['face_list'][i]['location']['height']
        cv2.rectangle(img0,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
        cv2.putText(img0,"beauty:"+str(result['result']['face_list'][i]['beauty']),(int(x),int(y+h)+15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(img0,"age:"+str(result['result']['face_list'][i]['age']),(int(x),int(y+h)+35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(img0,"gender:"+str(result['result']['face_list'][i]['gender']['type']),(int(x),int(y+h)+55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    plt.imshow(img0[:,:,::-1])
    plt.show()
else:
    print('no answer')