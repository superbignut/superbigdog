import os
import sys
sys.path.append("../scripts")
from darwin3_runtime_api import darwin3_device
from contextlib import redirect_stdout

# test = darwin3_device.darwin3_device(app_path='../', step_size=10000, ip=['192.168.1.90']) # 172.31.111.35
board = darwin3_device.darwin3_device(app_path='C:\\Users\\bignuts\\Desktop\\ZJU\\hang_zhou\\alcohol\API_4.0\\apps\\', step_size=1000_000, ip=['172.31.111.35'], spk_print=True) # 172.31.111.35
board.reset()
board.darwin3_init(333)
board.get_neuron_state(pop_name="layer1", state=[ [0, ['vt',],],  [1, ['vt',],] ])