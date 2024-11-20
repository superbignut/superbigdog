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
from darwin3_deployment.core_config.core_config import CoreConfig
from darwin3_deployment.ir.dinst import *


"""
class LIFModel(NeuronModel):
    
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm) # tag
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

        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]'))

NeuronModel.register("lif", LIFModel)

"""
def spaic_lif_config(vth, tauM=8):
    # 用于实现spaic的LIF 神经元
    """
    vt = vt * tauM
    vt = vt + wgsum
    O = func(vt)
    
    vt = vt + wgtsum if FCS == 1
    vt = p1 + vt x p0 + wgtsum x c6 + i x c7 if FCS ==0 其中 c6 = CPAR6 c7 = CPAR7
    """
    core_config = CoreConfig(
            assembly_program=[
                LSIS(ls=LSIS.LS.LOAD, nph=0b0100_0100), # vt weightsum
                UPTVT(fcs=0, fcph=0b00), # 使用完整公式 使用cpar0 和cpar1 作为参数 
                # 因此是 ： vt = cpar1 + vt x cpar0 + wgtsum x c6 + i x c7
                GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=1, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位 发放脉冲 比较 重置
                LSIS(ls=LSIS.LS.STORE, nph=0b0100_0000), # 把膜电压存回去 
                NPC(),
            ]
        )
    
    cpar0 = int(tauM)
    cpar1 = 0 # 
    cpar6 = 1
    cpar7 = 0 # 这两个都是0的，不写也是ok的
    core_config.set_register("CR_CPARA", cpar0 | (cpar1 << 8)) # cpar0 不能大于255 否则超过限制了
    core_config.set_register("CR_CPARB", cpar6 << 16) 
    core_config.set_register("R1", int(vth))
    return core_config


def compile_to_darwin():

    from darwin3_deployment.codegen import dump_input_neuron, add_connections, dump_pop, dump_output_neuron
    from darwin3_deployment.ir.net_population import PhysicalPopulation
    from darwin3_deployment.core_config.get_model_config import get_model_config
    from darwin3_deployment.pops_data import PopsDataConfig


    input0_neurons = PhysicalPopulation(shape=[10, ], coord=[-1, 1], pop_position="input") # 使用负坐标， 表示不在darwin上
    layer1_neurons = PhysicalPopulation(shape=[2,  ], coord=[ 0, 2]) #
    output_neurons = PhysicalPopulation(shape=[2,  ], coord=[-1, 3], pop_position='output') # 使用负坐标 

    emo_net_weight = torch.load(r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\save_200_quant\parameters\_variables.pt") # 加载权重

    weight_input0_to_layer1 = emo_net_weight[r'autoname14<net>_connection1<con>:autoname14<net>_layer1<neg><-autoname14<net>_input<nod>:{weight}'].detach().cpu() # 
    layer1_vth = emo_net_weight[r'autoname14<net>_layer1<neg>:{Vth}'].detach().cpu().item()
    # print(layer1_vth)

    add_connections('full',   weight=weight_input0_to_layer1, pre_pops=input0_neurons, post_pops=layer1_neurons) # 
    add_connections('output', weight=None,                    pre_pops=layer1_neurons, post_pops=output_neurons) # 添加轴突是什么意思

    pops_data = {}
    pops_data['layer1'] = {}
    my_config = get_model_config('if', vth=layer1_vth) # 这里暂时用if
    my_config.set_register(name='CR_VTDEC', value = -60 ) # 重置电压被设置为 -60


    pops_data['layer1']['core_config'] = my_config


    pops_data_config = PopsDataConfig()
    pops_data_config.set_pops_data(pops_data) # 注册

    dump_pop(pop_list=layer1_neurons, pop_name='layer1',output_dir='./save_darwin') # Todo  这个要怎么设定
    dump_input_neuron(input_pops=input0_neurons, output_dir='./save_darwin')
    dump_output_neuron(output_pops=layer1_neurons, pop_name="output_pop", output_dir='./save_darwin')

if __name__ == "__main__":
    # train()
    # single_test()
    # quant()
    compile_to_darwin()