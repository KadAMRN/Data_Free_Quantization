import torch
import torch.nn as nn
import copy
import numpy as np
from utils import visualize_per_layer
from utils.quantize import UniformQuantize
from ourplots import save_layer
import os


def _layer_equalization(W1, W2, B1, Batnorm_weight=None, Batnorm_bias=None, s_min_max=(1e-8, 1e8), signed=False, eps=0):
    num_channels = 1
    if W1.shape[0] != W2.shape[1]:
        # if input channels != output channels
        num_channels = W1.shape[0] // W2.shape[1]
	
    num_channel_in = W1.shape[0] // num_channels
    num_channel_o = W2.shape[0] // num_channels

    S = torch.zeros(W1.size(0))
    
    for channel in range(num_channels):

        in_channel_start = channel * num_channel_in
        in_channel_stop = (channel + 1) * num_channel_in
        W1_group = W1[in_channel_start:in_channel_stop] # shape [k, c, h, w]

        out_channel_start = channel * num_channel_o
        out_channel_stop = (channel + 1) * num_channel_o
        W2_group = W2[out_channel_start:out_channel_stop]

        for iter in range(W2_group.shape[1]):
            if signed:
                r1 = torch.max(torch.abs(W1_group[iter])) # signed
                r2 = torch.max(torch.abs(W2_group[:, iter])) # signed

            else:
                r1 = torch.max(W1_group[iter]) - torch.min(W1_group[iter]) # unsigned
                r2 = torch.max(W2_group[:, iter]) - torch.min(W2_group[:, iter]) # unsigned
            
            
            s = (1 / (r1 + eps)) * torch.sqrt(r1 * r2 + eps)
            s = max(s_min_max[0], min(s_min_max[1], s))
            S[in_channel_start + iter] = s

            W1[in_channel_start + iter].mul_(s)
            
            if B1 is not None:
                B1[in_channel_start + iter].mul_(s)

            if Batnorm_weight is not None:
                Batnorm_weight[in_channel_start + iter].mul_(s)

            if Batnorm_bias is not None:
                Batnorm_bias[in_channel_start + iter].mul_(s)

            W2[out_channel_start:out_channel_stop, iter].mul_(1/s)

    return W1, W2, B1, S



def cross_layer_equalization(graph, relations, Target_list, s_min_max=[1e-8, 1e8], Treshhold=2e-7, Count=20, signed=False, eps=0, Save_state=True):
    print("Cross layer equalization")
    with torch.no_grad():

        diff = 1e8
        iter_count = 0

        if Save_state:
            i=1
            dir='plots/exp'
            new_dir = dir + str(i)
            while os.path.exists(new_dir) == True:
                i+=1
                new_dir = dir + str(i)
            
            os.makedirs(new_dir)  


        while diff > Treshhold and iter_count < Count:

            old_graph = copy.deepcopy(graph)
            
            for rel in relations:
                layer_first, layer_second, bn_idx = rel.get_idxs()

                if Save_state:
                    
                    save_layer(graph[layer_first].weight.detach(), 'Before equalization',dir=new_dir)


                if graph[layer_first].bias is None: # add a fake bias term
                    graph[layer_first].bias = nn.Parameter(data=torch.zeros((graph[layer_first].weight.size(0)), dtype=torch.float32), requires_grad=False)
                
                # layer eualization
                graph[layer_first].weight, graph[layer_second].weight, graph[layer_first].bias, S =_layer_equalization(graph[layer_first].weight,graph[layer_second].weight, 
                                                                                                                       graph[layer_first].bias,graph[bn_idx].fake_weight,
                                                                                                                       graph[bn_idx].fake_bias, s_min_max=s_min_max, signed=signed, eps=eps)
                rel.set_scale_vec(S)
                
                if Save_state:
                    
                    save_layer(graph[layer_first].weight.detach(), 'After equalization',dir=new_dir)


            diff_list =[float(torch.mean(torch.abs(graph[layer_idx].weight - old_graph[layer_idx].weight))) for layer_idx in graph if type(graph[layer_idx]) in Target_list]
            diff_tmp = np.sum(diff_list)

            if abs(diff - diff_tmp) > 1e-9:
                iter_count = 0
                diff = diff_tmp
                
            else:
                iter_count += 1


