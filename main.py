"""
    将已有的模型编译到darwin3芯片上

    创建神经元群
    创建连接
    神经元信息注入

              output
                ^
                |
    input -> layer1 -> layer2    
                ^         |
                |_________|

"""
import os, torch, numpy, einops, yaml
from pathlib import Path
from darwin3_deployment.codegen import dump_input_neuron, add_connections, dump_pop, add_to_dendrite_full
from darwin3_deployment.ir.net_population import PhysicalPopulation
# from y_darwin_config.l_d3b3_config import config1 as POP_CONFIG1
from darwin3_deployment.core_config.get_model_config import get_model_config

input0_neurons = PhysicalPopulation(shape=[12, ], coord=[0, 1], pop_position="input0")
layer1_neurons = PhysicalPopulation(shape=[100,], coord=[0, 2], pop_position="layer1")
layer2_neurons = PhysicalPopulation(shape=[100,], coord=[0, 3], pop_position='layer2')
output_neurons = PhysicalPopulation(shape=[100,], coord=[0, 4], pop_position='output')

emo_net_weight = torch.load(f="mydog_variables.pt")

weight_input0_to_layer1 = emo_net_weight[r'autoname1<net>_connection1<con>:autoname1<net>_layer1<neg><-autoname1<net>_input<nod>:{weight}'].detach().cpu()
weight_layer1_to_layer2 = emo_net_weight[r'autoname1<net>_connection2<con>:autoname1<net>_layer2<neg><-autoname1<net>_layer1<neg>:{weight}'].detach().cpu()
weight_layer2_to_layer1 = emo_net_weight[r'autoname1<net>_connection3<con>:autoname1<net>_layer1<neg><-autoname1<net>_layer2<neg>:{weight}'].detach().cpu()

add_connections('full',   weight=weight_input0_to_layer1, pre_pops=input0_neurons, post_pops=layer1_neurons)
add_connections('full',   weight=weight_layer1_to_layer2, pre_pops=layer1_neurons, post_pops=layer2_neurons)
add_connections('full',   weight=weight_layer2_to_layer1, pre_pops=layer2_neurons, post_pops=layer1_neurons)
add_connections('output', weight=None,                    pre_pops=layer1_neurons, post_pops=output_neurons)

stdp_neuron_info = {}
stdp_neuron_info['core_config'] = get_model_config('if') # 这里要参照 l_d3_b3_config.py 来配置神经元

dump_pop(pop_list=layer1_neurons, pop_info=stdp_neuron_info, output_dir='./save') # info暂不设定
dump_pop(pop_list=layer2_neurons, pop_info=stdp_neuron_info, output_dir='./save') # info暂不设定
dump_input_neuron(input_pops=input0_neurons, output_dir='./save')


"""
    Todo:
        是否编译成功 ？
        权重可以更新 ？
"""