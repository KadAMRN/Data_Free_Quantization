import torch
import torch.nn as nn
import copy
import numpy as np
from utils import visualize_per_layer
from utils.quantize import UniformQuantize
from ourplots import display_layer
import os


def _layer_equalization(weight_first, weight_second, bias_first, bn_weight=None, bn_bias=None, s_range=(1e-8, 1e8), signed=False, eps=0):
    num_group = 1
    if weight_first.shape[0] != weight_second.shape[1]:
        # if input channels != output channels
        num_group = weight_first.shape[0] // weight_second.shape[1]
	
    group_channels_i = weight_first.shape[0] // num_group
    group_channels_o = weight_second.shape[0] // num_group

    S = torch.zeros(weight_first.size(0))
    
    for g in range(num_group):

        c_start_i = g * group_channels_i
        c_end_i = (g + 1) * group_channels_i
        weight_first_group = weight_first[c_start_i:c_end_i] # shape [k, c, h, w]

        c_start_o = g * group_channels_o
        c_end_o = (g + 1) * group_channels_o
        weight_second_group = weight_second[c_start_o:c_end_o]

        for ii in range(weight_second_group.shape[1]):
            if signed:
                range_1 = torch.max(torch.abs(weight_first_group[ii])) # signed
                range_2 = torch.max(torch.abs(weight_second_group[:, ii])) # signed

            else:
                range_1 = torch.max(weight_first_group[ii]) - torch.min(weight_first_group[ii]) # unsigned
                range_2 = torch.max(weight_second_group[:, ii]) - torch.min(weight_second_group[:, ii]) # unsigned
            
            
            s = (1 / (range_1 + eps)) * torch.sqrt(range_1 * range_2 + eps)
            s = max(s_range[0], min(s_range[1], s))
            S[c_start_i + ii] = s

            weight_first[c_start_i + ii].mul_(s)
            
            if bn_weight is not None:
                bn_weight[c_start_i + ii].mul_(s)

            if bn_bias is not None:
                bn_bias[c_start_i + ii].mul_(s)

            if bias_first is not None:
                bias_first[c_start_i + ii].mul_(s)

            weight_second[c_start_o:c_end_o, ii].mul_(1/s)

    return weight_first, weight_second, bias_first, S



def cross_layer_equalization(graph, relations, targ_type, s_range=[1e-8, 1e8], converge_thres=2e-7, converge_count=20, signed=False, eps=0, visualize_state=True):
    print("Start cross layer equalization")
    with torch.no_grad():

        diff = 1e8
        iter_count = 0

        if visualize_state:
            i=1
            dir='plots/exp'
            new_dir = dir + str(i)
            while os.path.exists(new_dir) == True:
                i+=1
                new_dir = dir + str(i)
            
            os.makedirs(new_dir)  


        while diff > converge_thres and iter_count < converge_count:

            state_prev = copy.deepcopy(graph)
            
            for rel in relations:
                layer_first, layer_second, bn_idx = rel.get_idxs()

                if visualize_state:
                    
                    display_layer(graph[layer_first].weight.detach(), 'Before equalization',dir=new_dir)


                if graph[layer_first].bias is None: # add a fake bias term
                    graph[layer_first].bias = nn.Parameter(data=torch.zeros((graph[layer_first].weight.size(0)), dtype=torch.float32), requires_grad=False)
                
                # layer eualization
                graph[layer_first].weight, graph[layer_second].weight, graph[layer_first].bias, S =_layer_equalization(graph[layer_first].weight,graph[layer_second].weight, 
                                                                                                                       graph[layer_first].bias,graph[bn_idx].fake_weight,
                                                                                                                       graph[bn_idx].fake_bias, s_range=s_range, signed=signed, eps=eps)
                rel.set_scale_vec(S)
                
                if visualize_state:
                    
                    display_layer(graph[layer_first].weight.detach(), 'After equalization',dir=new_dir)


            diff_list =[float(torch.mean(torch.abs(graph[layer_idx].weight - state_prev[layer_idx].weight))) for layer_idx in graph if type(graph[layer_idx]) in targ_type]
            diff_tmp = np.sum(diff_list)

            if abs(diff - diff_tmp) > 1e-9:
                iter_count = 0
                diff = diff_tmp
                
            else:
                iter_count += 1


