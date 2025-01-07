"""
    这个文件的功能是将录音与动态可视化结合起来
"""

import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024 * 10

audio = pyaudio.PyAudio() # 

fig, ax = plt.subplots() # 

x = np.arange(0, CHUNK) # 横坐标间隔

line, = ax.plot(x, np.random.rand(CHUNK)) # 生成随机数 plot 并取列表的第一个元素

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK) # 开始录音

# 相当于绘图前做一些初始化工作
def init():
    ax.set_ylim(-2**9, 2**9)
    fig.set_size_inches(10, 5) # 设置画布宽度
    

# 每次更新图像
def update(frame):
    data = stream.read(CHUNK) # 录一个chunk
    data_int = np.frombuffer(data, dtype=np.int16) # 把 bytes 数据 转为 int
    line.set_ydata(data_int) # 设置新的数据
    
ani = FuncAnimation(fig=fig, func=update, init_func=init) # blit=True 因此 update函数需要一个被修改对象的返回值
plt.show()

stream.stop_stream()
stream.close()
audio.terminate()