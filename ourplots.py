import matplotlib.pyplot as plt
import os


def display_layer(param, title='test',dir='plots/exp'):
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