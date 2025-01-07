
"""
    在windows上测试，大模型的窗口逻辑
"""
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
from baidu_api import baidu_wav_to_words

API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"

file_path = "output.wav"


def dmx_api(input_txt):
    conversation_id = None
    output=input_txt
    api_key = "299adac92d9b98c139f22fa1e22a8b2c.t7LzNyfNX49gsShG"
    url = "https://open.bigmodel.cn/api/paas/v4"
    client = ZhipuAI(api_key=api_key, base_url=url)
    prompt = output
    generate = client.assistant.conversation(
        assistant_id="659e54b1b8006379b4b2abd6",
        conversation_id=conversation_id,
        model="glm-4-assistant",
        messages=[
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            }
        ],
        stream=True,
        attachments=None,
        metadata=None
    )
    output = ""
    for resp in generate:
        if resp.choices[0].delta.type == 'content':
            output += resp.choices[0].delta.content
            conversation_id = resp.conversation_id
    return output


def lu_yin_and_save():    
    time.sleep(2)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 2 # 录音的时间
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    """ device_name = 'default'  # 你想使用的设备名称
    device_index = None
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if device_name in info['name']:
            device_index = i
            break """

    # 打开音频流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=0)  # windows 改成0 是可以工作的

    print("录音中...")

    frames = []
    # 录制音频
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # 如果 锁来了， 就break
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束")

    # 停止和关闭音频流

    audio.terminate()
    stream.stop_stream()
    stream.close()
    
    # 将录制的音频保存为wav文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def run():
    while True:
        try:
            lu_yin_and_save() # 录音

            text = baidu_wav_to_words() # 录音转文字

            text  = "从一个小狗的角度，判断下面这段话属于，1表扬我，2批评我，3正常谈话， 当中的哪个种类，只需要用编号1或2或3来回答我。" + text
            print(text)
            web_text = dmx_api(input_txt=text) # 
            print(web_text)

            if web_text == "1" or  web_text == "2" or web_text == "3":
                print("send")
                data = "dmx " + web_text + " " + "0"
                time.sleep(1)
            time.sleep(1) 
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))
            print("error")
            break
            




if __name__ == '__main__':
    run()