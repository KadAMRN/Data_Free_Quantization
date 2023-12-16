import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt



def bias_absorption(graph, relations, bottoms, N=3, visualize=False):
    def is_relu_activation(layer_second, layer_first, graph, bottoms):
        idx = layer_second
        while idx != layer_first:
            # assert len(bottoms[idx]) == 1, 'graph in equalization relations should be 1-to-1 input-output'
            if isinstance(graph[bottoms[idx][0]], torch.nn.ReLU):
                return True
            idx = bottoms[idx][0]
        return False
    print("Start bias absorption")

    changed_layers_count = 0

    for rel in relations :
        layer_first, layer_second, bn_idx = rel.get_idxs()
        if not is_relu_activation(layer_second, layer_first, graph, bottoms): # only absorb bias if there is relu activation accoding to the article calcs
            continue
        

        # get the weight and bias of the batchnorm layer
        # gamma std deviation, beta mean
        bn_gamma = getattr(graph[bn_idx], 'fake_weight').detach().clone() 
        bn_beta = getattr(graph[bn_idx], 'fake_bias').detach().clone() 

   
        # Get the weights of the second layer and their shape
        second_layer_weights = graph[layer_second].weight.detach().clone()
        second_layer_shape = second_layer_weights.shape

        if visualize:
            layer_second_bias_tmp = graph[layer_second].bias.detach().clone()

        # Calculate the number of groups and the step sizes for the outer and inner loops
        num_groups = graph[layer_first].weight.size(0) // graph[layer_second].weight.size(1)
        step_size_outer = graph[layer_second].weight.size(0) // num_groups
        step_size_inner = graph[layer_first].weight.size(0) // num_groups

        # Calculate the vector to be absorbed, assuming that pre-bias activations are distributed normally
        # with the batch normalization shift and scale parameters
        c = (bn_beta - N * bn_gamma)
        c.clamp_(0)

        # Reshape the second layer weights to be 3D
        second_layer_weights = second_layer_weights.view(second_layer_shape[0], second_layer_shape[1], -1)

        # Initialize the result vector
        wc = torch.zeros(second_layer_weights.size(0))

        # Calculate the result vector by multiplying the sum of the second layer weights with the absorption vector
        for i in range(num_groups):
            start_outer = i * step_size_outer
            end_outer = (i + 1) * step_size_outer
            start_inner = i * step_size_inner
            end_inner = (i + 1) * step_size_inner

            wc[start_outer:end_outer] = torch.matmul(
                torch.sum(second_layer_weights[start_outer:end_outer], -1),
                c[start_inner:end_inner]
            )
        # apply absorbtion by updating W and b of layer 2, b of layer 1 and bn of layer 1

        # first check if bias in not none, we absorbb bias even if it is none on the layer
        # to check
        for layer in [layer_first, layer_second]:
            if graph[layer].bias is None:
                graph[layer].bias = nn.Parameter(data=torch.zeros((graph[layer].weight.size(0)), dtype=torch.float32), requires_grad=False)

        

        graph[layer_first].bias.data.add_(-c) # b1_hat=b1-c
        graph[bn_idx].fake_bias.data.add_(-c) # h_hat=h-c
        graph[layer_second].bias.data.add_(wc) # b2_hat=b2+W2*c


    if visualize:

        unequal_indices = np.where(graph[layer_second].bias.detach().clone() != layer_second_bias_tmp)
        unequal_biases = graph[layer_second].bias.detach().clone()[unequal_indices]
        unequal_biases_tmp = layer_second_bias_tmp[unequal_indices]

        if len(unequal_biases) > 0:
            bias_diff = unequal_biases - unequal_biases_tmp
            print(f"Difference of biases that have changed for layer {layer_second}: {bias_diff}")
            changed_layers_count += 1

        print(f"Bias absorption done. Number of layers with changed biases: {changed_layers_count}")



        # Create a figure with two subplots
        fig, axs = plt.subplots(2)

        # Plot the histogram of the original biases in the first subplot
        axs[0].hist(layer_second_bias_tmp.numpy().flatten(), bins=256, color='blue', label='Original Biases')
        axs[0].set_xlabel('Biases')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].set_title('Histogram of Original Biases of Layer ' + str(layer_second))

        # Plot the histogram of the updated biases in the second subplot
        axs[1].hist(graph[layer_second].bias.detach().clone().numpy().flatten(), bins=256, color='red', label='Updated Biases')
        axs[1].set_xlabel('Biases')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].set_title('Histogram of Updated Biases of Layer ' + str(layer_second))

        # Show the figure
        plt.show(block=False)
        plt.pause(3)  # display for 3 seconds
        plt.close()


    print("Bias absorption done")