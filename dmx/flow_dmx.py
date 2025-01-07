"""
    这个代码打算能够，将录音不中断的进行， 只有在连续两个输入没有输入的时候，才会将之前的frames  保存wav 然后提交dmx，
    此外，如果一直是噪声输入，也不会被放进frames

    然后大模型调用由于是有延迟的， 因此可以做一个抢占，只要有新的来了老的就不必再等待了

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
import numpy as np



API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"

WAVE_OUTPUT_FILENAME = "output.wav"

record_input_th = 200

RECORD_STATE = 0 #

RECORD_STATE_DICT = {"Waiting_Input":0, "Input_Now":1, "Waiting_End":2}

# 0 表示 等待输入
# 1 表示 正在输入
# 2 表示 等待结束

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



def flow_record():    
    global WAVE_OUTPUT_FILENAME, record_input_th, RECORD_STATE
    time.sleep(2)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024 * 8 # 大改一个chunk 是 0.5s
    
    audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=0)  # windows 改成0 是可以工作的

    frames = []
    cnt = 0 # 用于标记这是第几个交给大模型的音频

    # 录制音频
    print("开始录音...")
    while(True):
        data = stream.read(CHUNK) # 录1个chunk

        data_int = np.abs(np.frombuffer(data, dtype=np.int16))
        # print(len(data), len(data_int)) 由于数据是16位，因此data的len是 8192 的double
        max_data_int = np.max(data_int)
        print("max is: ", np.max(data_int))

        if RECORD_STATE == RECORD_STATE_DICT['Waiting_Input']:
            print("waiting for input...")
            if max_data_int < record_input_th:
                continue 
            else:
                frames.append(data)
                RECORD_STATE = RECORD_STATE_DICT["Input_Now"]
        
        if RECORD_STATE == RECORD_STATE_DICT["Input_Now"]:
            frames.append(data)
            
            if max_data_int < record_input_th:
                RECORD_STATE = RECORD_STATE_DICT["Waiting_End"]
            else:
                continue
        
        if RECORD_STATE == RECORD_STATE_DICT["Waiting_End"]:
            frames.append(data)

            if max_data_int > record_input_th:
                RECORD_STATE = RECORD_STATE_DICT["Input_Now"]
            else:
                # 第二次小于则 则把数据提交给 dmx
                # 先转为 wav 文件，提交给百度
                wf_name = str(cnt) + WAVE_OUTPUT_FILENAME
                wf = wave.open(wf_name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                frames.clear() # 数据列表清空

                text = baidu_wav_to_words(file_name=wf_name) # 百度转文字

                text  = "从一个小狗的角度，判断下面这段话属于，1表扬我，2批评我，3正常谈话， 当中的哪个种类，只需要用编号1或2或3来回答我。" + text
                print(text)
                web_text = dmx_api(input_txt=text) # 
                print(web_text)

                cnt += 1
                RECORD_STATE = RECORD_STATE_DICT["Waiting_Input"]

        # frames.append(data)
        
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


if __name__ == '__main__':
    flow_record()