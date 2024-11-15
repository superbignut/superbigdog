"""
    先使用 spaic搭建最简单的分类网络，期望能够实现一个2分类效果


"""
from hang_zhou_spaic.SPAIC import spaic
import torch
import math
from hang_zhou_spaic.SPAIC.spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum

log_path = './log/easyspaic'
writer = SummaryWriter(log_path)

# tensorboard.exe --logdir=./log/easyspaic

device = 'cuda'

backend = spaic.Torch_Backend(device)

backend.dt = 0.1

run_time = 256 * backend.dt

bat_size = 1

input_dim = 10 # 这个应该不影响

layer_dim = 2

class EasyNet(spaic.Network):
    def __init__(self):
        super().__init__()

        self.input = spaic.Encoder(num=input_dim, coding_method='poisson')

        self.layer1 = spaic.NeuronGroup(num=layer_dim, model='lif')

        # self.layer2 = spaic.NeuronGroup(num=layer_dim, model='lif')

        self.output = spaic.Decoder(num=layer_dim, dec_target=self.layer1, coding_method='spike_counts') 

        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full')

        # self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full')

        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        
        self.learner.set_optimizer('Adam', 0.001)

        self.set_backend(backend)
        
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')


Net = EasyNet()

def train():

    _input = [[1,1,1,1,1, 0,0,0,0,0],[0,0,0,0,0, 1,1,1,1,1]] # 最简单的输入

    acc = deque(maxlen=100)
    
    for i in tqdm(range(400)):
        _num = i % 2    
        data = temp_input = torch.tensor(_input[_num], device=device).unsqueeze(0) # 增加了一个维度  

        Net.input(data)
        Net.run(run_time)
        output = Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        # print(output)
        label = torch.tensor(_num, device=device).unsqueeze(0)
        batch_loss = F.cross_entropy(output, label) # 计算交叉熵？

        Net.learner.optim_zero_grad()
        batch_loss.backward(retain_graph=False)
        Net.learner.optim_step()
        
        writer.add_scalar(tag="loss", scalar_value=batch_loss.item(), global_step=i) # 每次打印准确率
        
        predict_labels = torch.argmax(output, 1)
        if predict_labels == label:
            acc.append(1)
        else:
            acc.append(0)
        writer.add_scalar(tag="acc", scalar_value=sum(acc) / len(acc), global_step=i) # 每次打印准确率
        if i == 200:
            Net.save_state(filename = 'save_200/easyspaic_200') # 这里需要手动删除保存的文件夹
            break
    
def single_test():
    Net.state_from_dict(filename="save_200/easyspaic_200", device=device)
    for i in range(10):
        data =  torch.tensor([1,1,1,1,1, 0,0,0,0,0], device=device).unsqueeze(0)
        Net.input(data)
        Net.run(run_time)
        output = Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        print(output)
        predict_labels = torch.argmax(output, 1)
        print(predict_labels)

        time_line = Net.mon_V.times  # 取layer1层时间窗的坐标序号
        value_line = Net.mon_V.values[0][0]
        plt.plot(time_line, value_line)
        plt.show() # """ #  观察一下电压情况

        data =  torch.tensor([0,0,0,0,0, 1,1,1,1,1], device=device).unsqueeze(0)
        Net.input(data)
        Net.run(run_time)
        output = Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        print(output)
        predict_labels = torch.argmax(output, 1)
        print(predict_labels)


"""
class LIFModel(NeuronModel):
    
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm)
    O^n[t] = spike_func(V^n[t-1])

    def __init__(self, **kwargs):
        super(LIFModel, self).__init__()
        # initial value for state variables
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0


        self._constant_variables['Vth'] = kwargs.get('v_th', 1.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

        self._tau_variables['tauM'] = kwargs.get('tau_m', 8.0)

        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]')) # 脉冲来了，电压衰减、 增加
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth')) # 超过阈值发射
        self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]')) # 发射后的重置

NeuronModel.register("lif", LIFModel)

    # vth taum vreset
"""
def quant():
    class Symmetric_Quantize:
        class Target_Precision(Enum):
            INT8=127
    
        def __init__(self, precision):
            self.upper_bound = precision.value
            self.lower_bound = - precision.value
            self._scalar_factor = None
            self._bias = None
            self.q = 0.97
        
        def _get_scale_factor_score_weight(self, q, weight): # 这里我的权重本来就是 2维的，flatten 之后变为 1 维
            # print(weight)
            topq = torch.abs(weight).flatten(1).quantile(dim=1, q=q) # 这里的quantile 其实是找到一个分位数的概念
            _scalar_factor = self.upper_bound / topq # 计算放大因数
            # print("scalar_factor is :", _scalar_factor)
            _new_weight = torch.clamp(torch.round(_scalar_factor[:, None] * weight),min=self.lower_bound, max=self.upper_bound)
            _quant_back = _new_weight / _scalar_factor[:, None]
            _quant_loss = torch.nn.functional.mse_loss(_quant_back, weight)

            return _quant_loss, _scalar_factor, _new_weight # 返回量化损失，缩放因子，缩放后的权重
            
        
        
        def search_reasonable_quant_weight(self, weight):
            # 这里就不选了， 直接用0.97
            _score, _scalar_factor, _new_weight = self._get_scale_factor_score_weight(q=self.q, weight=weight)
            print("quant_loss is: ", _score)
            print("scalar_factor is: ", _scalar_factor)
            print("_new_weight is: ", _new_weight)
            
            return _scalar_factor, _new_weight

        def __call__(self, origin_weight):
            # print(origin_weight)
           return self.search_reasonable_quant_weight(weight=origin_weight)

    Net.state_from_dict(filename="save_200/easyspaic_200", device=device) # 加载权重

    quant_obj = Symmetric_Quantize(Symmetric_Quantize.Target_Precision.INT8) # 量化权重

    _scalar_factor, _new_weight= quant_obj(Net.connection1.weight.value) # 量化权重

    class QuantEasyNet(spaic.Network): # 用于保存量化后模型的新网络
        def __init__(self):
            super().__init__()

            self.input = spaic.Encoder(num=input_dim, coding_method='poisson')

            self.layer1 = spaic.NeuronGroup(num=layer_dim, model='lif', v_th=1.0*_scalar_factor, tau_m=8.0*_scalar_factor, v_reset=0.0) # 1 8 是原默认设置

            self.output = spaic.Decoder(num=layer_dim, dec_target=self.layer1, coding_method='spike_counts') 

            self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=_new_weight.detach().cpu().numpy()) # 量化后的权重

            # self.learner = spaic.Learner(trainable=self, algorithm='STCA')
            
            # self.learner.set_optimizer('Adam', 0.001) # 量化后 先不用训练
            
            self.mon_V = spaic.StateMonitor(self.layer1, 'V')

            new_backend = spaic.Torch_Backend(device)

            new_backend.dt = 0.1

            self.set_backend(new_backend)

    # vth taum vreset 应该都要全比例的放大， reset=0 应该不用管吧

    Quant_Net = QuantEasyNet()

    # 测试SPAIC 量化后的模型效果: # 

    for i in range(2):
        data =  torch.tensor([1,1,1,1,1, 0,0,0,0,0], device=device).unsqueeze(0)
        Quant_Net.input(data)
        Quant_Net.run(run_time)
        output = Quant_Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        print(output)
        predict_labels = torch.argmax(output, 1)
        print(predict_labels)

        time_line = Quant_Net.mon_V.times  # 取layer1层时间窗的坐标序号
        value_line = Quant_Net.mon_V.values[0][0]

        """plt.plot(time_line, value_line)
        plt.show() # """ #  观察一下电压情况
        

        data =  torch.tensor([0,0,0,0,0, 1,1,1,1,1], device=device).unsqueeze(0)
        Quant_Net.input(data)
        Quant_Net.run(run_time)
        output = Quant_Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        print(output)
        predict_labels = torch.argmax(output, 1)
        print(predict_labels)

    # 竟然成功了

    
    

if __name__ == "__main__":
    # train()
    single_test()
    # quant()
    
    