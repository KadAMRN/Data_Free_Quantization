import matplotlib.pyplot as plt


def boxplt_graph_weights(graph,title='Weights Range'):
    plt_keys=list(graph.keys())
    plt.figure(figsize=(25, 12))

    # Prepare a list to hold weights of all layers
    all_weights = []

    # Iterate over all keys in plt_keys
    for key in plt_keys:
        # Check if the layer has weights
        if hasattr(graph[key], 'weight'):
            weights = graph[key].weight.detach().cpu().numpy().flatten()
            all_weights.append(weights)

    # Create a box plot of the weights
    
    plt.boxplot(all_weights,showbox=True,showfliers=False,showmeans=True,meanline=True,meanprops={'color':'red','linewidth':2})



    # Set the labels for the x and y axes
    # plt.xlabel('Layers Index')
    plt.ylabel('Weights Range')

    plt.title(title)

    # Show the plot
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)  # display for 5 seconds
    plt.close()