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
# from darwin3_deployment.ir.dinst import ADD as _ADD

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


register = {
    "vt":0,
    "vth":1,
    "i":2,
    'rpd':3,
    'wgtsum':4,
    'res':5,
    'rand':6,
    'wgt':7,
    'pra01':0xD,
    'pra23':0xE,
    'pra45':0xF
}
""" for i in range(len(register)):
    register.i =  """


def spaic_stdpihlif_ts_learn_config(timestep=25, vreset=-100):
    # 输入量化后的阈值， 衰减固定是15/16 在spaic 中需要 修改lif 的 tau_m 为 tau_m=(-0.1 / np.log(3 / 4)) 因为 spaic 自带一层 np.exp(-dt / tau)
    # 用于实现spaic的LIF 神经元
    """
    vt = vt * tauM
    vt = vt + wgsum
    O = func(vt)
    
    vt = vt + wgtsum if FCS == 1
    vt = p1 + vt x p0 + wgtsum x c6 + i x c7 if FCS ==0 其中 c6 = CPAR6 c7 = CPAR7
    """
    """ inference_state_settings={
        "vth": "vth",
    }, """




    core_config = CoreConfig(

        inference_state_settings={
            # "vth": "my_vth", # 阈值存储器
            # "i":"my_loop_index",
        },

        assembly_program=[
            
            NPC(),

        ]
    )
    
    core_config.set_register("CR_VTDEC", int(hex((vreset & 0xffff)<<16), 16)) 
    # core_config.set_learning_mode(True)
    # core_config.set_register("CR_LI", 0x01)
    return core_config