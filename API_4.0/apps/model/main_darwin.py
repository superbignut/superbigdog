"""
    这个文件相较于 main.py 目的在于把原来用  spaic 的网络传播 改成用 Darwin3 的完全替代 尽量做到接口函数都不变
    只修改 step 函数
"""
import os
import sys
import numpy as np
import torch
import time
# sys.path.append("../scripts")
from darwin3_runtime_api import darwin3_device
import numpy as np
from tqdm import tqdm
import os
import sys
import random
import torch

import torch.nn.functional as F
import multiprocessing
import csv
import pandas as pd
from collections import deque
import traceback
import threading
import socket
from Controller import Controller
import time
import subprocess
import wave
import pyaudio

"""
    | 0     1     2    3  |   4    5  |  6    7  |  9      10  |   11    12     13 |     14    15   |   8    |
    | 白    红    蓝  表扬 |  红    蓝 |  酒   酒 |  表扬   批评 |   上    下     挥  |    电低  电低  |   白    |     
"""

EMO = {"POSITIVE":0, "NEGATIVE":1, "ANGRY":2, "NULL_P":3, "NULL_N":4} # NULL(只在没有输入的时候使用 ), 积极，消极，愤怒

INTERACT = {"POSITIVE":0, "NEGATIVE":1, "ANGRY":2} # 用于对交互结果进行 积极和消极的判定 # 这里还要加一个

model_path = 'save/ysc_model'

buffer_path = 'ysc_buffer.pth'

device = torch.device("cuda:0")

input_node_num_origin = 16

input_num_mul_index = 16 #  把输入维度放大16倍

assign_label_len = 10 # 情感队列有效长度

input_node_num = input_node_num_origin * input_num_mul_index #  把输入维度放大16倍

output_node_num = 3 # 情感输出种类

label_num = 100 # 解码神经元数目


def _bo_fang(index):
    try:
        if index == 1:
            file_name = "wang_wang.wav"
        elif index == 2:
            file_name = "woof_sad.wav"
        # 打开.wav文件
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        wf = wave.open(file_path, 'rb')

        # 创建PyAudio对象
        p = pyaudio.PyAudio()

        # 打开音频流
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # 播放数据
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # 停止音频流
        stream.stop_stream()
        stream.close()

        # 关闭 PyAudio
        p.terminate()
        wf.close()
    except:
        print("audio played error!")
    finally:
        print("audio played over!")

"""
    | 0     1     2    3  |   4    5  |  6    7  |  9      10  |   11    12     13 |     14    15   |   8    |
    | 白    红    蓝  表扬 |  红    蓝 |  酒   酒 |  表扬   批评 |   上    下     挥  |    电低  电低  |   白    |     
"""
class Darwin_Net():
    def __init__(self):
        super().__init__()

        self.buffer = [[] for _ in range(output_node_num)] # 这里不能写成 [[]] * 4 的形式， 否则会右拷贝的问题

        self.assign_label = None # 统计结束的解码层神经元的的情感分组

        self.time_step = 25

        self.board = self.ysc_darwin_init() # 初始化了板子

        
    
    def ysc_darwin_init(self):
        # darwin 板子初始化 ，参考运行时手册
        board = darwin3_device.darwin3_device(app_path='API_4.0/apps/', step_size=1000_000, ip=['172.31.111.35'], spk_print=True) # 172.31.111.35

        time.sleep(1)
        board.reset() # 重置
        time.sleep(1)
        board.darwin3_init(333) # 时钟频率初始化
        time.sleep(1)
        board.deploy_config() # 板子权重、连接初始化
        time.sleep(1)

        return board

    def ysc_darwin_step(self, input_ls):
        # 一个时间步周期的运行， 输入数据维度 1 * 256
        # 返回脉冲输出情况 维度 1 * 256
        ls = np.zeros(100,)
        for _ in range(self.time_step):
            a = self.input_func(input_ls)
            out = self.board.run_darwin3_withoutfile(spike_neurons=[a])

            for i in range(len(out[0])):
                index = out[0][i][1]
                ls[index] += 1
        print(ls.nonzero())
        print(ls[ls.nonzero()])
        return np.array([ls])  # 

    def input_func(self, input_ls, unit_conversion=0.8, dt=0.1):# 这连个参数对标spaic的possion_encoder
        # 这里改成 numpy 的输入, 输入维度 1 * input_node_num
        # 返回 一个 list 也即发放脉冲的神经元编号
        a = (np.random.rand(*input_ls.shape) < input_ls * unit_conversion * dt)

        return np.nonzero(a[0].astype(int))[0]


    def step(self, data, reward=1): 
        # 对接到 darwin
        # Todo reward 暂未使用
        return self.ysc_darwin_step(data)
    
    def new_check_label_from_data(self, data):
        """
            这里暗含了优先级的概念在里面, 但要是能真正影响 情绪输出的还得是 权重
        """
        if data[0][2] == 1 or data[0][5] == 1 or data[0][10] == 1 or data[0][12] == 1:
            return EMO["ANGRY"] # 
        
        elif data[0][15] == 1 or  data[0][14] == 1 or data[0][7] == 1 or data[0][6] == 1 or data[0][4] == 1 or data[0][1] == 1:
            return EMO['NEGATIVE']
        
        elif data[0][0] == 1 or data[0][3] == 1 or data[0][8] == 1 or data[0][9] == 1 or data[0][11] == 1 or data[0][13] == 1:
            return EMO['POSITIVE']
        
        else:
            raise NotImplementedError


    def assign_label_update(self, newoutput=None, newlabel=None, weight=0):
        # 如果没有新的数据输入，则就是对 assign_label 进行一次计算，否则 会根据权重插入新数据，进而计算

        if newoutput != None:
            self.buffer[newlabel].append(newoutput)
        try:
            avg_buffer = [sum(self.buffer[i][-assign_label_len:]) / len(self.buffer[i][-assign_label_len:]) for i in range(len(self.buffer))] # sum_buffer 是一个求和之后 取平均的tensor  n * 1 * 100
            
            assign_label = torch.argmax(torch.cat(avg_buffer, 0), 0) # n 个1*100 的list在第0个维度合并 -> n*100的tensor, 进而在第0个维度比哪个更大, 返回一个1维的tensor， 内容是index，[0,n)， 目前是012
            
            self.assign_label = assign_label # 初始化结束s
        except ZeroDivisionError:
            # 如果分母是零 说明是刚开始数据还不够的时候，就暂时不管
            return     
          
    def predict_with_no_assign_label_update(self, output):
        # 根据输出 返回模型的预测
        if self.assign_label == None:
            raise ValueError("predict_with_no_assign_label_update error!")
        
        temp_cnt = [0 for _ in range(len(self.buffer))]
        temp_num = [0 for _ in range(len(self.buffer))]

        for i in range(len(self.assign_label)):
            # print(i)
            temp_cnt[self.assign_label[i]] += output[0, i]  # 第一个维度是batch, 
            temp_num[self.assign_label[i]] += 1
    
        predict_label = np.argmax(np.array(temp_cnt) / np.array(temp_num)) # 有待验证
        
        return predict_label
    
    def influence_all_buffer(self, interact, temp_output):
        # interact ： 0 积极交互 1 消极交互 2 愤怒交互，
        # 
        if interact == EMO['POSITIVE']:

            self.buffer[EMO['POSITIVE']].append(temp_output)
            
            self.buffer[EMO['NEGATIVE']][-1] += -1 * temp_output 

            self.buffer[EMO['ANGRY']][-1] += -1 * temp_output 
            
        elif interact == EMO["NEGATIVE"]:

            self.buffer[EMO['NEGATIVE']].append(temp_output)

            self.buffer[EMO['POSITIVE']][-1] += -1 * temp_output

            self.buffer[EMO['ANGRY']][-1] += -1 * temp_output 
        else:
            self.buffer[EMO['ANGRY']].append(temp_output)

            self.buffer[EMO['POSITIVE']][-1] += -1 * temp_output

            self.buffer[EMO['NEGATIVE']][-1] += -1 * temp_output 
            
        self.assign_label_update() # 施加了积极和消极得影响后 重新 assign label


    def single_test(self):
        
        print(self.assign_label)
        t = 1
        while t < 20:
            t+=1
            result_list = [0.0] * input_node_num_origin
            for i in range(input_node_num_origin):
                if i == 4: # 1 红， 9 10 抚摸
                    result_list[i] = 1.0
                else:
                    result_list[i] = random.uniform(0, 0.2)
            result_list = result_list * input_num_mul_index
            # print(result_list)
            
            # Todo 有待补充

class Gouzi:

    def __init__(self) -> None:

        #### 全局变量， 从socket线程中修改， 用于 self.net 的前向推理
        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
        self.color = 0 #  0 1 2  无， 蓝色， 红色                3
        self.alcohol = 0 #  0 1 2   无， 酒精 酒精               3
        self.dmx = 0 # 0 1 2 3 # 无，积极，消极                  3
        self.gesture = 0 # 0 1 2 3 无  上，下，挥手              4
        self.power = 0 # 0 1 # 20% 以下， 和以上的电量           2

        self.robot_net = Darwin_Net() # 
        
        self.state_update_lock = threading.Lock() # 这个lock 使用来检测 狗的状态的更新的， 在检测线程 和 clear 线程中使用

        self.emo_thread = threading.Thread(target=self.emo_handle_thread, name="emo_handle_thread")

        self.action_thread = threading.Thread(target=self.action_handle_thread, name="action_thread") # 动作线程还没做

        self.controller = None
        
        self.action_socket_init() # 初始化 Controller, 

        self.is_moving = False

    
    def action_socket_init(self):
        # 运动主机初始化、创建运动控制器、建立心跳
        # client_address = ("192.168.1.103", 43897)
        server_address = ("192.168.1.120", 43893)  # 运动主机端口
        self.controller = Controller(server_address) # 创建 控制器

        self.controller.heart_exchange_init() # 初始化心跳
        time.sleep(2)
        self.controller.stand_up()
        print('stand_up')
        # pack = struct.pack('<3i', 0x21010202, 0, 0)
        # controller.send(pack) # 
        time.sleep(2)
        self.controller.not_move() # 进入 静止状态
        print("动作socket初始化")

        
        """
            | 0     1     2    3  |   4    5  |  6    7  |  9      10  |   11    12     13 |     14    15   |   8    |
            | 白    红    蓝  表扬 |  红    蓝 |  酒   酒 |  表扬   批评 |   上    下     挥  |    电低  电低  |   白    |     
        """

    def start(self):

        self.robot_net.load_weight_and_buffer(model_path="save_600/real_ysc_model_mic", buffer_path='real_ysc_buffer_600_mic.pth') # 这里以后还可以换成 其他训练轮数的模型

        self.emo_thread.start() # 情感线程启动

        self.say_something(index=1)

        self.action_thread.start() # 启动 运动执行模块

        print("into server")
        
        self.start_server() # 启动监听线程 ， 线程中不断获取传感器数据

    def say_something(self, index):    
            
            # temp_p = multiprocessing.Process(target=_bo_fang, args=(index, ))
            temp_t = threading.Thread(target=_bo_fang, name="bo_fang_thread", args=(index, ))
            temp_t.start() # 这里 因为麦克风是 io 且是独占的，所有 多线程可以加速， 并且 需要join
            temp_t.join()
            # _bo_fang(index=index)
            print("播放结束")
    
    def positive_action_with_flag_changed(self): # 做一些积极的动作， 并标志flag
        #########################################
        self.is_moving = True
        #########################################

        self.controller.zuo_you_huang()
        time.sleep(1)
        self.controller.thread_active = False
        
        #########################################
        # 执行完所有动作后
        self.emo_queue.popleft()
        self.is_moving = False
        #
        ########################################
    def negative_action_with_flag_changed(self): # 做一些消极的动作， 并标志flag
        #########################################
        self.is_moving = True
        #########################################

        
        self.controller.low_height_of_dog()
        time.sleep(1)
        self.controller.thread_active = False
        time.sleep(1)

        #########################################
        # 执行完所有动作后
        self.emo_queue.popleft()
        self.is_moving = False
        #
        ########################################

    
    def clear(self):
        with self.state_update_lock:
            self.imu = 0
            self.color = 0
            self.alcohol = 0
            self.dmx = 0
            self.gesture = 0
            self.power = 0
                
            """
                | 0     1     2    3  |   4    5  |  6    7  |  9      10  |   11    12     13 |     14    15   |   8    |
                | 白    红    蓝  表扬 |  红    蓝 |  酒   酒 |  表扬   批评 |   上    下     挥  |    电低  电低  |   白    |     
            """

    def cmd_waiting_thread(self):
        """
            0 每次检测各个传感器信号， 判断是不是指令
            1 如果是指令，则找到情感输出
            2 融合后动作输出
        """
        while True:
            
            temp_input = [0 for _ in range(input_node_num_origin)]
            is_activate = None
            
            def _check_input():
                if self.color == 2:
                    temp_input[4] = 1 # 红
                    temp_input[1] = 1

                if self.alcohol == 1:
                    temp_input[6] = 1
                    temp_input[7] = 1
                
                if self.dmx == 1:
                    temp_input[9] = 1
                    temp_input[3] = 1                    
                elif self.dmx ==2:
                    temp_input[10] = 1

                if self.gesture == 1 or self.gesture == 3:
                    temp_input[11] = 1
                    temp_input[13] = 1
                elif self.gesture == 2:
                    temp_input[12] = 1
                
                if self.power == 1:
                    temp_input[14] = 1 
                    temp_input[15] = 1            
            
            
            is_moving_flag = False
            # 这里打算等待imu 等待 1 秒钟， 如果有imu 交互输入 就 在线学习， 否则就 正常推理
            start_time = time.time()

            while time.time() - start_time < 1: # 在检查imu 的间隙 检查 其他输入
                if self.is_moving: # 如果正在运动，也是不进行前向传播
                    self.clear()
                    is_moving_flag = True
                    break
                
                _check_input() # 继续检测输入

                if self.imu == 1: # 摸
                    is_activate = INTERACT["POSITIVE"]
                    break
                elif self.imu == 2:
                    is_activate = INTERACT['NEGATIVE']
                    break
                
                time.sleep(0.1)  # 每 0.1 秒检测一次

            if is_moving_flag == True:
                continue 
            elif sum(temp_input) < 1:  # 
                if is_activate is INTERACT['POSITIVE']: # 如果没有输入 只有交互 那就 制作简答的动作
                    print("简单 输出")
                    self.emo_queue_add_in_lock(data=EMO["NULL_P"]) # 表示只有只需要简答的动作
                    self.clear()
                    continue
                elif is_activate is INTERACT['NEGATIVE']:
                    self.emo_queue_add_in_lock(data=EMO["NULL_N"]) # 表示只有简单的动作
                    self.clear()
                    print("简单 输出")
                    continue
                else:
                    continue

            print("input is : ", temp_input) # 有输入的情况

            self.clear() # 清除状态
            
            temp_input = torch.tensor(temp_input * input_num_mul_index, device=device).unsqueeze(0) # 增加了一个维度  

            temp_output = self.robot_net.step(data=temp_input, reward=1) # 前向传播
            print("temp_output is :",temp_output.shape)

            darwin_output = self.ysc_darwin_step(input_ls=temp_input) # 这里的输出是numpy，可能需要转成tensor 才行
            print("darwin_output is: " ,darwin_output.shape)

            temp_predict = self.robot_net.just_predict_with_no_assign_label_update(output=temp_output) # 得到预测

            real_label = None

            if is_activate == INTERACT['POSITIVE']:
                # 抚摸输入
                self.robot_net.influence_all_buffer(interact=INTERACT["POSITIVE"], temp_output=temp_output)
                print("向积极方向修正")
            elif is_activate == INTERACT["NEGATIVE"]:
                # 踢打输入
                self.robot_net.influence_all_buffer(interact=INTERACT["NEGATIVE"], temp_output=temp_output)
                print("向消极方向修正")
            else:
                real_label = self.robot_net.new_check_label_from_data(data=temp_input)
                self.robot_net.buffer[temp_predict].append(temp_output) # 根据 预测的结果正向强化
                print("根据预测结果正向强化")
                

            self.emo_queue_add_in_lock(temp_predict) # 这里暂时还是 每次产生一个情感吧
                
            print("predict_label is: ", temp_predict, "real_label is: ", real_label)
            

    def emo_queue_add_in_lock(self, data):
        with self.emo_queue_lock:
            self.emo_queue.append(data)

    def client_handle_thread(self, client_socket):
        # 处理不同客户端上报的环境数据， 修改self 的全局变量
        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                if self.is_moving:
                    continue
                # print(data)
                command, args1, args2 = data.decode('utf-8').split()  # 假设数据格式为 "COMMAND arg1 arg2"
                # print(command, args1, args2)
                args1 = int(args1)
                # args2 = int(args2)
                print(f"Received command: {command}, args: {args1}, {args2}")

                """
                        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
                        self.color = 0 #  0 1 2   无， 蓝色， 红色               3
                        self.alcohol = 0 #  0 1 2   无， 酒精 酒精               3
                        self.dmx = 0 #    1 2 3 #  积极，消极, 正常               3
                        self.gesture = 0 # 0 1 2 3 无  上，下，挥手              4
                        self.power = 0 # 0 1 # 20% 以下， 和以上的电量           2
                """
                with self.state_update_lock:  # 修改状态 上锁
                    if command == "gesture":
                        if args1 == 4: #  nice 表扬 手势
                            self.gesture = 1
                            print("up_gesture")
                        elif args1 == 5: #  批评手势
                            print("down_gesture") 
                            self.gesture = 2
                        elif args1 == 1: # 手掌手势
                            print("hello_gesture") 
                            self.gesture = 3
                        else:
                            self.gesture = 0


                    elif command == "imu":
                        if args1 == 1: # 抚摸
                            print("touching_imu")
                            self.imu = 1
                        elif args1 == 2: # 敲打
                            print("was kicked_imu")
                            self.imu = 2
                        else:
                            self.imu = 0 


                    elif command == "color":
                        if args1 == 1:
                            print("red_color")
                            self.color = 2 
                        else:
                            self.color = 0


                    elif command == "alco":
                        if args1 == 1:
                            print("drink_alco")
                            self.alcohol = 1
                        else:
                            self.alcohol = 0


                    elif command == "dmx":
                        if args1 == 1:
                            print("biao_yang_dmx")
                            self.dmx = 1
                        elif args1 == 2:
                            print("pi_ping_dmx")
                            self.dmx = 2 
                        elif args1 == 3:
                            print("nothing_dmx")
                            self.dmx = 3


                    elif command == "power":
                        if args1 == 1:
                            self.power = 1
                        else:
                            self.power = 0 # 正常
                    
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
            # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))

        finally:
            client_socket.close()


    def start_server(self, host='192.168.1.103', port=12345):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5) # 等待数， 连接数
        print(f"Server listening on {host}:{port}...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            client_handler = threading.Thread(target=self.client_handle_thread, args=(client_socket,))
            client_handler.start()




if __name__ == "__main__":
    
    # 如果需要重新构造数据集的话，需要重新打开这个函数， 把其余部分注释掉
    # 这个文件需要在 API_4.0 外面执行
    # ysc_darwin_init()
    xiaobai = Gouzi()
    xiaobai.start()


