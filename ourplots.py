import matplotlib.pyplot as plt
import os

def boxplt_graph_weights(graph,title='Weights Range'):
    plt_keys=list(graph.keys())
    plt.figure(figsize=(10, 10))

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