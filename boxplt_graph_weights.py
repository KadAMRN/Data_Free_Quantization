# import matplotlib.pyplot as plt


# def boxplt_graph_weights(graph,title='Weights Range'):
#     plt_keys=list(graph.keys())
#     plt.figure(figsize=(25, 12))

#     # Prepare a list to hold weights of all layers
#     all_weights = []

#     # Iterate over all keys in plt_keys
#     for key in plt_keys:
#         # Check if the layer has weights
#         if hasattr(graph[key], 'weight'):
#             weights = graph[key].weight.detach().cpu().numpy().flatten()
#             all_weights.append(weights)

#     # Create a box plot of the weights
    
#     plt.boxplot(all_weights,showbox=True,showfliers=False,showmeans=True,meanline=True,meanprops={'color':'red','linewidth':2})



#     # Set the labels for the x and y axes
#     # plt.xlabel('Layers Index')
#     plt.ylabel('Weights Range')

#     plt.title(title)

#     # Show the plot
#     plt.tight_layout()
#     # plt.show(block=False)
#     # plt.pause(5)  # display for 5 seconds
#     # plt.close()
#     plt.show()
import torch.nn as nn


from utils.quantize import QuantConv2d, QuantLinear
import matplotlib.pyplot as plt
import numpy as np

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



# ...

    # Check if the 10th layer exists and is a convolutional or linear layer
    if len(plt_keys) > 9 and isinstance(graph[plt_keys[9]], (nn.Conv2d, nn.Linear,QuantConv2d, QuantLinear)):
        weights_10th_layer = graph[plt_keys[9]].weight.detach().cpu().numpy().flatten()

        # Get the unique weight values
        unique_weights = np.unique(weights_10th_layer)

        # Print the unique weight values
        print(f"Unique weight values of the 10th layer ({type(graph[plt_keys[9]]).__name__}): {unique_weights}")

        # Create a histogram of the weights of the 10th layer
        axs[1].hist(weights_10th_layer, bins=256, color='blue', label='Weights')
        axs[1].set_xlabel('Weights')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].set_title('Histogram of Weights of 10th Layer')
    else:
        print("The 10th layer does not exist or is not a convolutional or linear layer.")

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.show(block=False)
    plt.pause(7)  # display for 5 seconds
    plt.close()
        # plt.show()