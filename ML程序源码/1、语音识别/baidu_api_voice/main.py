from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '24910443'
API_KEY = 'lO4E8a0SPGIoTwD8h7CWIg7c'
SECRET_KEY = '0gq5mpLIGrnkzWhT2XghIp8fNFzN4Zwh'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 读取文件
def get_file_content(filePath):
    with open(filePath,'rb') as fp:
        return fp.read()

# 识别本地文件
print (client.asr(get_file_content("test.wav"),'wav',16000,{'dev_pid':1537,}))


