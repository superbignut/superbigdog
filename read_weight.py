"""
    把darwin上的权重数据读出来， 这里主要是 还是通过 get neuron state 的函数，去读树突表， 每次从表的权重地址 读100 个
    sort 是由于 app.log 返回的总是乱序， 因此 先排了一个序

    最终的 256 个权重  在 weight.txt 中， 我是用的是 8位有符号来表示 所以 大致是 -127 到 127  每个权重占 8位

    weight.txt  中一行 是 6 个 8位 也就是 6个权重
"""

import os 
import re

file_path = r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\API_4.0\apps\model\app.log"

sorted_file_path = "app.txt"

addr_pattern = r'addr=0x([0-9a-f]+)'

weight_pattern = r'value=0x[0-9a-f]+([0-9a-f]{4})'

temp_re_pattern = r'addr=(0x[0-9a-f]+)'

temp_weight_pattern = r'value=0x([0-9a-f]+)'

def sort_applog():
    ls = []
    with open(file_path, 'r') as f:
        with open(sorted_file_path, 'w') as new_app_f:

            for index, line in enumerate(f):
                ls.append(line)
                # print(line[81:86])
            ls = sorted(ls, key=lambda x : int(x[81:86], 16))

            new_app_f.write(''.join(ls))
        

def main():


    print("successful")
    ls = []  # ls 由于 存在一点乱序， 需要排序一下
    real_addr = []
    with open(sorted_file_path, 'r') as f:
        # ls = [] # ls 用来存储app.log 读到的权重地址
        
        for index, line in enumerate(f):
            if index > 300:
                break
            addr = re.search(addr_pattern, line)
            weight = re.search(weight_pattern, line)
            # print(addr.group(1), weight.group(1),end=" ")
            # print(hex(int(addr.group(1), 16) + int(weight.group(1), 16)))
            
            # ls.append(hex(int(addr.group(1), 16) + int(weight.group(1), 16)))
            
            ls.append((int(addr.group(1), 16), hex(0x10000 + int(weight.group(1), 16))))
        
        sorted_ls = sorted(ls, key=lambda x: x[0]) # (addr_ori, new_addr) 第一个参数 用作 排序， 第二个才是真的地址
        
        for i in range(len(sorted_ls)-1):
            if i > 260:
                break
            if sorted_ls[i+1][0] - sorted_ls[i][0] != 1:
                print(sorted_ls[i], " ", sorted_ls[i+1])
                print("some error occured.")
                break
            else:
                real_addr.append(sorted_ls[i][1]) # 把 排序后的 真实地址拿出来
    # 使用ls 去寻找 真实的权重
    ls_addr_weight = [] # 地址与权重的对应 list
    num_of_weight_17_lines = 255 # 总共有 256 个 大小为 100 的树突
    num_of_weight_17_lines_cnt = 0


    with open(sorted_file_path, 'r') as f:
        with open("weight.txt", 'w') as wf:
            find_flag = False
            weight_17_lines = [] #
            for index, line in enumerate(f):

                if find_flag == False:
                    temp_re = re.search(temp_re_pattern, line)

                    if temp_re != None and temp_re.group(1) in real_addr: # 从这行开始的 包括这行在内的 17 行 每行 6个权重，共100个， 最后一行有4个可用
                        print(temp_re.group(1), "yes")
                        weight_17_lines.append(re.search(temp_weight_pattern, line).group(1))
                        find_flag = True
                        continue
                elif len(weight_17_lines) < 17:
                        weight_17_lines.append(re.search(temp_weight_pattern, line).group(1))
                
                # else:
                    
                if len(weight_17_lines) == 17 :    
                    wf.write('\n'.join(weight_17_lines) + '\n'*2)
                    weight_17_lines.clear()
                    find_flag = False
                    num_of_weight_17_lines_cnt += 1
                elif num_of_weight_17_lines_cnt > num_of_weight_17_lines:
                    break
                    
                    
                        # print(temp_re.group(1), weight.group(1))

if __name__ == '__main__':
    
    sort_applog()
    main()