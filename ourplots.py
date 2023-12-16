import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
from utils.quantize import QuantConv2d, QuantLinear

def boxplt_and_hist_graph_weights(graph, title='Weights Range'):
    plt_keys = list(graph.keys())
    fig, axs = plt.subplots(2, figsize=(25, 12))

    # Prepare a list to hold weights of all layers
    all_weights = []

    # Iterate over all keys in plt_keys
    for key in plt_keys:
        # Check if the layer has weights
        if hasattr(graph[key], 'weight'):
            weights = graph[key].weight.detach().cpu().numpy().flatten()
            all_weights.append(weights)

    # Create a box plot of the weights
    axs[0].boxplot(all_weights, showbox=True, showfliers=False, showmeans=True, meanline=True, meanprops={'color':'red','linewidth':2})

    # Set the labels for the x and y axes
    axs[0].set_ylabel('Weights Range')
    axs[0].set_title(title)




    # Check if the 18th layer exists and is a convolutional or linear layer
    if len(plt_keys) > 17 and isinstance(graph[plt_keys[17]], (nn.Conv2d, nn.Linear,QuantConv2d, QuantLinear)):
        weights_18th_layer = graph[plt_keys[17]].weight.detach().cpu().numpy().flatten()

        # Get the unique weight values
        unique_weights = np.unique(weights_18th_layer)
        min_weight = np.min(weights_18th_layer)
        max_weight = np.max(weights_18th_layer)
   
        print(f"Min and max weight values of the 18th layer ({type(graph[plt_keys[17]]).__name__}): {min_weight}, {max_weight}")
        print(f"Number of unique weight values of the 18th layer ({type(graph[plt_keys[17]]).__name__}): {len(unique_weights)}")
        # Create a histogram of the weights of the 18th layer
        axs[1].hist(weights_18th_layer, bins=256, color='blue', label='Weights')
        axs[1].set_xlabel('Weights')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].set_title('Histogram of Weights of 18th Layer')
    else:
        print("The 18th layer does not exist or is not a convolutional or linear layer.")

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.show(block=False)
    plt.pause(5)  # display for 5 seconds
    plt.close()



def save_layer(param, title='test',dir='plots/exp'):
    param_list = []
    #transform param from tensor to numpy 
    
    
    param_numpy = param.cpu().numpy()
    for idx in range(param.shape[0]): #Channels
        
        param_list.append(param_numpy[idx].reshape(-1))
    
    i=1
    filename = dir+"/"+ title +str(i)+ '.png'
    while os.path.exists(filename):
        i+=1
        filename =dir+"/"+ title +str(i)+ '.png'
        
    plt.figure()
    plt.boxplot(param_list, showfliers=False)
    plt.title(title)
    #plt.show()
    plt.savefig(filename)
    plt.close()


def plot_histograms(graph1, graph2):
    plt_keys1 = list(graph1.keys())
    plt_keys2 = list(graph2.keys())
    
    # Check if the 18th layer exists and is a convolutional or linear layer in both graphs
    if len(plt_keys1) > 17 and len(plt_keys2) > 17 and \
       isinstance(graph1[plt_keys1[17]], (nn.Conv2d, nn.Linear,QuantConv2d, QuantLinear)) and \
       isinstance(graph2[plt_keys2[17]], (nn.Conv2d, nn.Linear,QuantConv2d, QuantLinear)):

        # Get the weights of the 18th layer from both graphs
        weights1 = graph1[plt_keys1[17]].weight.detach().cpu().numpy().flatten()
        weights2 = graph2[plt_keys2[17]].weight.detach().cpu().numpy().flatten()

        # Create a histogram of the weights from both graphs
        plt.hist(weights1, bins=256, color='blue', label='Graph 1 Weights')
        plt.hist(weights2, bins=256, color='red', label='Graph 2 Weights', alpha=0.5)
        plt.xlabel('Weights')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Weights of 18th Layer Before/After')
        plt.show()
    else:
        print("The 18th layer does not exist or is not a convolutional or linear layer in one or both graphs.")





def plot_histograms_sub(weights1, weights2):
        # Create a histogram of the weights from both graphs
        plt.hist(weights1, bins=256, color='blue', label='Before DFQ')
        plt.hist(weights2, bins=256, color='red', label='After DFQ', alpha=0.5)
        plt.xlabel('Weights')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Weights of 18th Layer Before/After')
        plt.show()

def print_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            # print(param.element_size())
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))