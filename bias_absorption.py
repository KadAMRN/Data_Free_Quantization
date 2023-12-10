import torch
import torch.nn as nn
import copy
import numpy as np



def bias_absorption(graph, relations, bottoms, N=3):
    def is_relu_activation(layer_second, layer_first, graph, bottoms):
        idx = layer_second
        while idx != layer_first:
            # assert len(bottoms[idx]) == 1, 'graph in equalization relations should be 1-to-1 input-output'
            if isinstance(graph[bottoms[idx][0]], torch.nn.ReLU):
                return True
            idx = bottoms[idx][0]
        return False
    print("Start bias absorption")
    for rel in relations :
        layer_first, layer_second, bn_idx = rel.get_idxs()
        if not is_relu_activation(layer_second, layer_first, graph, bottoms): # only absorb bias if there is relu activation accoding to the article calcs
            continue
        

        # get the weight and bias of the batchnorm layer
        # gamma std deviation, beta mean
        bn_gamma = getattr(graph[bn_idx], 'fake_weight').detach().clone() 
        bn_beta = getattr(graph[bn_idx], 'fake_bias').detach().clone() 

        # Bias coming from the layer equalization can be absorbed from layer 1 into layer 2, layer 1 output being the input of layer 2, weights of layer 1 will stay unchanged
        # get weights of layer 2
        layer_second_weight = graph[layer_second].weight.detach().clone()
        # layer_second_bias = graph[layer_second].bias.detach().clone()
        layer_second_shape = layer_second_weight.shape


        num_group = graph[layer_first].weight.size(0) // graph[layer_second].weight.size(1) 
        step_size_o = graph[layer_second].weight.size(0) // num_group 
        step_size_i = graph[layer_first].weight.size(0) // num_group 

        #assuming that pre-bias activations are distributed normally with the batch normalization shift and scale parameters
        c = (bn_beta - N * bn_gamma) # non negative vect to be absorbed  
        c.clamp_(0)

        # reshape layer second weights to be 3D since weights are 4D or 2D initially depending on the layer if conv or linear respectively and gotta check bn shape   

        layer_second_weight = layer_second_weight.view(layer_second_shape[0], layer_second_shape[1], -1)

        wc = torch.zeros(layer_second_weight.size(0))

        for i in range(num_group): # to check
            wc[i*step_size_o:(i+1)*step_size_o] = torch.matmul(torch.sum(layer_second_weight[i*step_size_o:(i+1)*step_size_o], -1), c[i*step_size_i:(i+1)*step_size_i])


        # apply absorbtion by updating W and b of layer 2, b of layer 1 and bn of layer 1

        # first check if bias in not none, we absorbb bias even if it is none on the layer
        # to check
        # for layer in [layer_first, layer_second]:
        #     if graph[layer].bias is None:
        #         graph[layer].bias = nn.Parameter(data=torch.zeros((graph[layer].weight.size(0)), dtype=torch.float32), requires_grad=False)

        graph[layer_first].bias.data.add_(-c) # b1_hat=b1-c
        graph[bn_idx].fake_bias.data.add_(-c) # h_hat=h-c
        graph[layer_second].bias.data.add_(wc) # b2_hat=b2+W2*c
    
    print("Bias absorption done")