"""
    将 权重 画出来


"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_weight_func():
    w = torch.load(r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\quant_input_layer1.pth").detach().cpu().numpy()
    w_new = w.reshape(10, 10, 16, 16).transpose(0, 2, 1, 3).reshape(160, 160)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(w_new, cmap='hot_r', vmin=0, vmax=256)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect("auto")

    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.show()

def darwin_plot_weight_func():

    pass

if __name__ == '__main__':

    print("successfully!")
    plot_weight_func()