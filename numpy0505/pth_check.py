import torch
import numpy as np
import math
'''
pthfile = r'/Users/zengowen/Desktop/本科毕业论文/网络net参数提取/pth_check//lsq_ckpt_94.pth'

model = torch.load(pthfile, torch.device('cpu'))

file = "./out_file.txt"

with open(file,'w') as outfile:
    for k in model:
        outfile.write(k)
        outfile.write('\n')
        outfile.write(str(model[k]))
        outfile.write('\n')
        
outfile.close()
'''
eps = 1e-5
M = 13

def count_bias(bn_m, bn_b, bn_var, bn_w, sa, sw):
    
    # all is np array
    bn_m_np = bn_m.numpy()
    bn_b_np = bn_b.numpy()
    bn_var_np = bn_var.numpy()
    bn_w_np = bn_w.numpy()
    sa_np = sa.numpy()
    sw_np = sw.numpy() #float
    
    bias_np = np.rint((- bn_m_np + bn_b_np * np.sqrt(bn_var_np + eps) / bn_w_np) / (sa_np * sw_np))
    return bias_np.astype(np.int16)

def count_bias1(b, sa, sw):
    
    # all is np array
    b_np = b.numpy()
    sa_np = sa.numpy()
    sw_np = sw.numpy() #float
    
    bias_np = np.rint(b_np/(sa_np*sw_np))
    return bias_np.astype(np.int16)    

def count_gam(sa, sw, bn_w, bn_var, sa1, M):
    bn_var_np = bn_var.numpy()
    bn_w_np = bn_w.numpy()
    sa_np = sa.numpy()
    sw_np = sw.numpy() #float
    sa1_np = sa1.numpy()
    
    gam_np = np.rint((sa_np * sw_np * bn_w_np / (np.sqrt(bn_var_np + eps) * sa1_np)) * pow(2, M))
    return gam_np.astype(np.int16)

def count_gam1(sa, sw, M):
    
    sa_np = sa.numpy()
    sw_np = sw.numpy() #float
    
    
    gam_np = np.rint((sa_np * sw_np * pow(2, M)))
    return gam_np.astype(np.int16)

def count_weight(sw, weight):
    sw_np = sw.numpy() #float
    weight_np = weight.numpy() # output_channel -- input_channel -- height -- width
    weight_quan_np = np.clip(np.rint(weight_np / sw_np), -3, 3)
    return weight_quan_np.astype(np.int16) # but need int 3bits

def print_param(model, bias_file, gam_file, weight_file):
    with open(bias_file, "w") as outfile1, open(gam_file, "w") as outfile2, open(weight_file, "w") as outfile3:
        key_num = ["0", "2", "4", "6", "8", "10", "13", "18", "21", "22"] # 10 conv
        weight_key = []
        sw_key = []
        sa_key = []
        bn_w_key = []
        bn_b_key = []
        bn_m_key = []
        bn_var_key = []
        for i in key_num :
            weight_key.append("module_list." + i + ".conv_" + i + ".weight")
            sw_key.append("module_list." + i + ".conv_" + i + ".quan_w_fn.s")
            sa_key.append("module_list." + i + ".conv_" + i + ".quan_a_fn.s")
            bn_w_key.append("module_list." + i + ".batch_norm_" + i + ".weight")
            bn_b_key.append("module_list." + i + ".batch_norm_" + i + ".bias")
            bn_m_key.append("module_list." + i + ".batch_norm_" + i + ".running_mean")
            bn_var_key.append("module_list." + i + ".batch_norm_" + i + ".running_var")
        
        for num in range(len(key_num)):
            outfile1.write("conv" + key_num[num] + ":\n")
            if num < len(key_num) - 1 :
                outfile1.write(str(count_bias(model[bn_m_key[num]], model[bn_w_key[num]], model[bn_var_key[num]], model[bn_w_key[num]], model[sa_key[num]], model[sw_key[num]])))
            outfile1.write("\n")

            outfile2.write("conv" + key_num[num] + ":\n")
            if num < len(key_num) - 1 :
                outfile2.write(str(count_gam(model[sa_key[num]], model[sw_key[num]], model[bn_w_key[num]], model[bn_var_key[num]], model[sa_key[num+1]],M)))
            outfile2.write("\n")

            outfile3.write("conv" + key_num[num] + ":\n")
            
            outfile3.write(str(count_weight(model[sw_key[num]], model[weight_key[num]])))
            outfile3.write("\n")

        outfile1.close()
        outfile2.close()
        outfile3.close()

bias_file = "./bias.txt"
gam_file = "./gam.txt"
weight_file = "./weight.txt"
'''
print_param(model, bias_file, gam_file, weight_file)
'''
def get_param(model): # only for our tiny_yolo3
    key_num = ["0", "2", "4", "6", "8", "10", "13", "18", "21", "22"] # 10 conv
    weight_key = []
    sw_key = []
    sa_key = []
    bn_w_key = []
    bn_b_key = []
    bn_m_key = []
    bn_var_key = []
    weight_quan = []
    bias_quan = []
    gam_quan = []
    
    for i in key_num :
        weight_key.append("module_list." + i + ".conv_" + i + ".weight")
        sw_key.append("module_list." + i + ".conv_" + i + ".quan_w_fn.s")
        sa_key.append("module_list." + i + ".conv_" + i + ".quan_a_fn.s")
        bn_w_key.append("module_list." + i + ".batch_norm_" + i + ".weight")
        bn_b_key.append("module_list." + i + ".batch_norm_" + i + ".bias")
        bn_m_key.append("module_list." + i + ".batch_norm_" + i + ".running_mean")
        bn_var_key.append("module_list." + i + ".batch_norm_" + i + ".running_var")
    for num in range(len(key_num)):
        if num < len(key_num) - 1 :
            bias_quan.append(count_bias(model[bn_m_key[num]], model[bn_w_key[num]], model[bn_var_key[num]], model[bn_w_key[num]], model[sa_key[num]], model[sw_key[num]]))
            weight_quan.append(count_weight(model[sw_key[num]], model[weight_key[num]]))
            gam_quan.append(count_gam(model[sa_key[num]], model[sw_key[num]], model[bn_w_key[num]], model[bn_var_key[num]], model[sa_key[num+1]],M))
    weight_quan.append(count_weight(model[sw_key[num]], model[weight_key[num]]))
    bias_quan.append(count_bias1(model["module_list.22.conv_22.bias"], model[sa_key[num]], model[sw_key[num]]))
    gam_quan.append(count_gam1(model[sa_key[num]], model[sw_key[num]],M))
    gam_shortcut = count_gam(model[sa_key[4]], model[sw_key[4]], model[bn_w_key[4]], model[bn_var_key[4]], model[sa_key[8]],M);
    return bias_quan, weight_quan, gam_quan, gam_shortcut
