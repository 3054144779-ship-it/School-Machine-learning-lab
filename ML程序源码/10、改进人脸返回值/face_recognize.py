from aip import AipFace
import base64
import cv2
from    matplotlib  import pyplot   as plt

APP_ID = '24777669'
API_KEY = 'woke0A2nZvDVoOnt78b4ZkdS'
SECRET_KEY = 'DDCCQvLa9GZd1npYqNXctt3P6Fx9LHV4'

clt = AipFace(APP_ID, API_KEY, SECRET_KEY)

fP = './卷福/卷福0.jpg'

with open(fP, 'rb') as fp:
    bs64_dt = base64.b64encode(fp.read())

img = str(bs64_dt, 'utf-8')

imgTp = 'BASE64'

grpIdLst = 'Sherlock'

rslt = clt.search(img, imgTp, grpIdLst)

print(rslt)

if rslt['error_msg'] == 'SUCCESS':
    # pass
    print(rslt['result']['user_list'][0]['score'])
    if rslt['result']['user_list'][0]['score'] < 90:
        print('未找到指定人脸')
    else:
        print(rslt['result']['user_list'][0]['user_id'])
else:
    print('未找到人脸')