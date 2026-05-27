from aip import AipSpeech

# 获取密钥
APP_ID = "122771829"
API_KEY = "jftMHq8XZnixJpgz0EAW1x0a"
SECRET_KEY = "YtS6TUKC9k5Ck43u4jgUEPoN0KUJ13CB"

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 读取音频文件
def getFileContent(filePath):
    with open(filePath, "rb") as fp:
        return fp.read()

audio_path = "test.wav"

# 调用语音识别
result = client.asr(
    getFileContent(audio_path), # 获取文件路径下语音信息
    'wav', # 文件类型
    16000, # 采样率
    {'dev_pid': 1537}  # 1537=普通话
)

print("识别结果：", result)