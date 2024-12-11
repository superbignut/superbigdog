"""
    将 权重 画出来


"""

import torch
import matplotlib.pyplot as plt
import numpy as np
# 画权重函数
def inner_plot_func(weight):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(weight, cmap='hot_r', vmin=0, vmax=256)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect("auto")

    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.show()


# 加载量化后的权重
def plot_weight_func():
    w = torch.load(r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\quant_input_layer1.pth").detach().cpu().numpy()
    w_new = w.reshape(10, 10, 16, 16).transpose(0, 2, 1, 3).reshape(160, 160)
    inner_plot_func(weight=w_new)
    
# 加载从darwin上读出来的权重
def darwin_plot_weight_func():
    ls = []
    temp_ls = []
    hex_data = None
    with open('weight.txt', 'r') as file:
        hex_data = file.read().split()
    
    

    for index, item in enumerate(hex_data):
        # print(item)
        
        for i in range(len(item) // 2):
            num = int(item[i*2: i*2+2], 16) # 把权重分割
            if num > 127:
                num -= 256
            
            temp_ls.append(num)
            if len(temp_ls) == 102:
                temp_ls.pop(-1)
                temp_ls.pop(-1)
                ls.extend(temp_ls)          # 把权重放到ls中
                temp_ls.clear()        
                
            # print(num)
        
        # print(ls)
    print(len(ls))
    

    ls_new = np.array(ls)
    np.savetxt('new_w.txt', ls_new, fmt='%d', delimiter=',')
    ls_new = ls_new.reshape(256, 100).T.reshape(10, 10, 16, 16).transpose(0, 2, 1, 3).reshape(160, 160)
    
    inner_plot_func(weight=ls_new)

if __name__ == '__main__':

    print("successfully!")
    plot_weight_func()
    darwin_plot_weight_func()