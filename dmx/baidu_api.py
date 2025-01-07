
import pyaudio
import wave
import time
from zhipuai import ZhipuAI
# import pyttsx3
import socket
import base64
import urllib
import requests
import json
import os
import sys
from filelock import FileLock
import traceback

file_path = "output.wav"
API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"
# wav 文件转 文本
def baidu_wav_to_words():

    def get_access_token():
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))


    url = "https://vop.baidu.com/server_api"

    
    speech = get_file_content_as_base64(file_path, False)
    sp_len = os.path.getsize(file_path)
        
    payload = json.dumps({
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": "vks6nBUXlchi2SekxmPHOuFoqW0UpeMe",
        "dev_pid": 1537,
        "speech": speech,
        "len": sp_len,# os.path.getsize(file_path),
        "token": get_access_token()
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.json())
    return(response.json().get('result')[0])


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

