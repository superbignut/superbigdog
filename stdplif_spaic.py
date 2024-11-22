"""
在easy_spaic 的基础上， 把if改成 lif 来测试一下， 神经元写的对不对

虽然写完了，但是发现了挺多问题

+ 首先是spaic的原本的衰减是 0.987 但是这其实 基本没怎么衰减，
+ 为了考虑到写到darwin3 上 的汇编好实现一点 我就 把衰减改成了 3/4 但是发现衰减太大了， 导致本地训练都出问题
+ 就又改成了15/16 = 0.93 虽有所好转，但是还是 没有 0.987 效果好， 我猜测就是 由于衰减太大了，或者是我的损失函数写的不对
+ 但没有原则上的错误

test.get_neuron_state(pop_name="layer1", state=[ [0, ['vt',],],  [1, ['vt',],] ]) # 读存储器

test.get_neuron_state(pop_name="layer1", state=[ [0, ['npu_reg',],],  [1, ['npu_reg',],] ], offset=0x00) # 读寄存器

+ 使用这两个读取方式，可以测试出膜电压的衰减 满足 15/16的比例
+ 还有一个地方就是 知道了spaic 的 _tau_dict 是有一个自动的衰减的，但是我debug 没有找到 衰减的位置，很奇怪， 只看到一个np.exp却又没跳进去
+ 感觉还挺简单的，
+ Todo 的话， 可以考虑 写一个 自适应阈值，或者看一下，stdpex 去掉自适应阈值会怎么样，stdpex里面的衰减太小了，感觉不那么重要
+ 然后的话，stdpex 是有一个 如果脉冲 则 vth ++， 如果不脉冲 vth -- 这么一个操作
+ 先提交了，以后再看
"""
"""
    先使用 spaic搭建最简单的分类网络，期望能够实现一个2分类效果


"""
from hang_zhou_spaic.SPAIC import spaic
# from hang_zhou_spaic.SPAIC.spaic
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


log_path = './log/lifspaic'
writer = SummaryWriter(log_path)

# tensorboard.exe --logdir=./log/lifspaic

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

        self.input = spaic.Encoder(num=input_dim, coding_method='poisson') # tau_m=(-0.1 / np.log(3 / 4)

        self.layer1 = spaic.NeuronGroup(num=layer_dim, model='lif', tau_m=(-0.1 / np.log(15/16)), v_th=1.0) # 这里有一个变换 真正起作用的是 np.exp(-1 / tau_m)
        # tau_m=(-0.1 / np.log(3 / 4)) = 0.773 这么做的目的就是 为了 让实际的衰减率 达到 * 3/4 使用 移位实现
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

    _input = [[5,5,5,5,5, 0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2, 5,5,5,5,5]] # 最简单的输入

    acc = deque(maxlen=100)
    
    for i in tqdm(range(401)):
        _num = i % 2    
        data = temp_input = torch.tensor(_input[_num], device=device).unsqueeze(0) # 增加了一个维度  

        Net.input(data) # 
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


        """ time_line = Net.mon_V.times  # 取layer1层时间窗的坐标序号
        value_line = Net.mon_V.values[0][0]
        plt.plot(time_line, value_line)
        plt.show()  """

        if i == 400:
            # Net.save_state(filename = 'save_200/easyspaic_200') # 这里需要手动删除保存的文件夹
            spaic.Network_saver.network_save(Net=Net, filename='save_lif_200', save=True)
            break
    
def single_test():
    # Net.state_from_dict(filename="save_200/easyspaic_200", device=device)
    Net = spaic.Network_loader.network_load(filename='save_lif_200',device=device)
    for i in range(10):
        data =  torch.tensor([5,5,5,5,5, 0,0,0,0,0], device=device).unsqueeze(0)
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

        data =  torch.tensor([0,0,0,0,0, 5,5,5,5,5], device=device).unsqueeze(0)
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


"""
class LIFModel(NeuronModel):
    
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm)
    # 上面写的应该是错的 ： 
    # 正确的表达式为 dV/dt = - 1/tao V + I 
    # 这个 tau 的合理 的值 应该也是 小于1 的, 但其实 如果不考虑 物理意义 tao 的值 或大或小 对训练又有什么影响呢

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

            _new_weight = torch.clamp(torch.round(_scalar_factor[:, None] * weight), min=self.lower_bound, max=self.upper_bound)
            # print(_new_weight)

            _quant_back = _new_weight / _scalar_factor[:, None]
            _quant_loss = torch.nn.functional.mse_loss(_quant_back, weight)

            return _quant_loss, _scalar_factor, _new_weight # 返回量化损失，缩放因子，缩放后的权重
            
        
        
        def search_reasonable_quant_weight(self, weight):
            # 这里就不选了， 直接用0.97
            _score, _scalar_factor, _new_weight = self._get_scale_factor_score_weight(q=self.q, weight=weight)
            print("origin_weight", weight)
            print("quant_loss is: ", _score)
            print("scalar_factor is: ", _scalar_factor)
            print("_new_weight is: ", _new_weight)
            
            return _scalar_factor, _new_weight

        def __call__(self, origin_weight):
            # print(origin_weight)
           return self.search_reasonable_quant_weight(weight=origin_weight)

    # Net.state_from_dict(filename="save_200/easyspaic_200", device=device) # 加载权重
    Net = spaic.Network_loader.network_load(filename='save_lif_200', device=device)

    quant_obj = Symmetric_Quantize(Symmetric_Quantize.Target_Precision.INT8) # 量化权重

    _scalar_factor, _new_weight= quant_obj(Net.connection1.weight.value) # 量化权重

    vth_level = Symmetric_Quantize.Target_Precision.INT8.value

    vth_origin = Net.layer1._var_dict['autoname1<net>_layer1<neg>:{Vth}'].value
    # print(vth_origin)
    _new_vth = torch.clamp(vth_origin * _scalar_factor, min=-vth_level, max=vth_level).detach().cpu().numpy()[0].item()

    # _tau_origin = Net.layer1._var_dict['autoname1<net>_layer1<neg>:{tauM}'].value # 0.987应该是不衰减的
    
    # _new_tau = torch.clamp(_tau_origin * _scalar_factor, min=-vth_level, max=vth_level).detach().cpu().numpy()[0].item()


    print("_new_vth is: ", _new_vth)
    class QuantEasyNet(spaic.Network): # 用于保存量化后模型的新网络
        def __init__(self):
            super().__init__()

            self.input = spaic.Encoder(num=input_dim, coding_method='poisson')

            self.layer1 = spaic.NeuronGroup(num=layer_dim, model='lif', v_th=_new_vth, v_reset=0.0, tau_m=(-0.1 / np.log(15/16))) # tao 应该不变化

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
    Quant_Net.build()
    spaic.Network_saver.network_save(Net=Quant_Net, filename='save_lif_200_quant', save=True)

    """
        实际测试发现，_variables.pt 的 tau_M 是不准确的，diff_para_dict里面的是正确的    
    """
    # 测试SPAIC 量化后的模型效果: # 
    # return
    for i in range(1):
        data =  torch.tensor([5,5,5,5,5, 0,0,0,0,0], device=device).unsqueeze(0)
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
        # Quant_Net.save_state(filename = 'save_200_quant/easyspaic_200_quant')
        # spaic.Network_saver.network_save(Net=Quant_Net, filename='save_200_quant', save=True)

        data =  torch.tensor([0,0,0,0,0, 5,5,5,5,5], device=device).unsqueeze(0)
        Quant_Net.input(data)
        Quant_Net.run(run_time)
        output = Quant_Net.output.predict
        # print(output)
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        print(output)
        predict_labels = torch.argmax(output, 1)
        print(predict_labels)

    
def compile_to_darwin():

    from darwin3_deployment.codegen import dump_input_neuron, add_connections, dump_pop, dump_output_neuron
    from darwin3_deployment.ir.net_population import PhysicalPopulation
    from darwin3_deployment.core_config.get_model_config import get_model_config
    from darwin3_deployment.pops_data import PopsDataConfig
    from core_configs.lif_config import spaic_lif_config
    from core_configs.stdpexlif_config import spaic_stdpexlif_config

    input0_neurons = PhysicalPopulation(shape=[10, ], coord=[-1, 1], pop_position="input") # 使用负坐标， 表示不在darwin上
    layer1_neurons = PhysicalPopulation(shape=[2,  ], coord=[ 0, 2]) #
    output_neurons = PhysicalPopulation(shape=[2,  ], coord=[-1, 3], pop_position='output') # 使用负坐标 

    emo_net_weight = torch.load(r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\save_lif_200_quant\parameters\_variables.pt") # 加载权重

    weight_input0_to_layer1 = emo_net_weight[r'autoname14<net>_connection1<con>:autoname14<net>_layer1<neg><-autoname14<net>_input<nod>:{weight}'].detach().cpu() # 
    layer1_vth = emo_net_weight[r'autoname14<net>_layer1<neg>:{Vth}'].detach().cpu().item()
    # print(layer1_vth)

    add_connections('full',   weight=weight_input0_to_layer1, pre_pops=input0_neurons, post_pops=layer1_neurons) # 
    add_connections('output', weight=None,                    pre_pops=layer1_neurons, post_pops=output_neurons) # 添加轴突是什么意思

    pops_data = {}
    pops_data['layer1'] = {}
    pops_data['layer1']['core_config'] = spaic_stdpexlif_config()
    pops_data['layer1']['vth_theta'] = (np.zeros((2,)) + 0) # 所有的vth_theta最开始都是0， 每次脉冲增加 0.5
    pops_data['layer1']['vth'] = (np.zeros(2,) - 52) # 初始化是-52

    pops_data_config = PopsDataConfig()
    pops_data_config.set_pops_data(pops_data) # 注册

    dump_pop(pop_list=layer1_neurons, pop_name='layer1',output_dir='./save_stdp') # Todo  这个要怎么设定
    dump_input_neuron(input_pops=input0_neurons, output_dir='./save_stdp')
    dump_output_neuron(output_pops=layer1_neurons, pop_name="output_pop", output_dir='./save_stdp')

if __name__ == "__main__":
    # train()
    # single_test()
    # quant()
    compile_to_darwin()