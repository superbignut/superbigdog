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

        data =  torch.tensor([0,0,0,0,0, 1,1,1,1,1], device=device).unsqueeze(0)
        Net.input(data)
        Net.run(run_time)
        output = Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        print(output)
        predict_labels = torch.argmax(output, 1)
        print(predict_labels)

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
        
        def _get_scale_factor_score_weight(self, q, weight):
            topq = torch.abs(weight).flatten(1).quantile(dim=1, q=q) # Todo
            pass
        
        
        def search_reasonable_quant_weight(self, weight):
            # 这里就不选了， 直接用0.97
            _score, _new_weight = Symmetric_Quantize._get_scale_factor(q=self.q, weight=weight)
            

        def __call__(self, origin_weight):
            print(origin_weight)
            _scalar_factor, _new_weight = Symmetric_Quantize.search_reasonable_quant_weight(weight=origin_weight)

    quant_obj = Symmetric_Quantize(Symmetric_Quantize.Target_Precision.INT8)
    quant_obj(Net.connection1.weight) # 把权重传进去
    
    
    

if __name__ == "__main__":
    # train()
    # single_test()
    quant()
    
    