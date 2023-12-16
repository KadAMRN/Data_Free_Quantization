import os
import time
import matplotlib.pyplot as plt
from ourplots import boxplt_and_hist_graph_weights, save_layer, print_model_size, plot_histograms, plot_histograms_sub
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from modeling.segmentation.deeplab import DeepLab
from modeling.segmentation import resnet_v1

from modeling.classification.MobileNetV2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from utils.relation import create_relation
from dfq import bias_correction, _quantize_error, clip_weight #, bias_absorption, cross_layer_equalization
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer#, LayerTransform
from PyTransformer.transformers.torchTransformer import TorchTransformer

from bias_absorption  import bias_absorption
from Cross_layer_equal import cross_layer_equalization
# from bias_correction import bias_correction
# from clip_weight import clip_weight 

from utils.quantize import QuantConv2d, QuantLinear, QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits



# oonx # print len graph weights

def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", action='store_true')
    # parser.add_argument("--time_bench", action='store_true')
    parser.add_argument("--beforeafter", action='store_false')
    parser.add_argument("--sizedisp", action='store_false')
    parser.add_argument("--visualize", action='store_false')

    parser.add_argument("--quantize", action='store_false')
    parser.add_argument("--equalize", action='store_false')

    parser.add_argument("--correction", action='store_false')
    parser.add_argument("--absorption", action='store_false')
    parser.add_argument("--relu", action='store_false') # must replace relu6 to relu while equalization'
    parser.add_argument("--clip_weight", action='store_false')
 
    parser.add_argument("--task", default='cls', type=str, choices=['cls', 'seg'])
    parser.add_argument("--resnet", action='store_true')
    parser.add_argument("--log", action='store_true')

    # quantize params
    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    parser.add_argument("--bits_bias", type=int, default=8)

    return parser.parse_args()


def inference_all(model,args_gpu=True):
    print("Start inference")
    imagenet_dataset = datasets.ImageFolder('./val', transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ]))

    dataloader = DataLoader(imagenet_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for ii, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference Progress"):
            if args_gpu:
                image, label = sample[0].cuda(), sample[1].numpy()

            else:
                image, label = sample[0].cpu(), sample[1].numpy()

            logits = model(image)

            pred = torch.max(logits, 1)[1].cpu().numpy()
            
            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            
            # print accuracy for each batch
            print(f"Batch {ii+1}/{len(dataloader)}: Accuracy: {num_correct/num_total*100:.2f}%")

    acc = num_correct / num_total
    print(f"Final Accuracy: {acc*100:.2f}%")
    return acc


def main():
    args = get_argument()
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'

    

    if args.task == 'cls':

        data = torch.ones((4, 3, 224, 224))#.cuda()
        if args.resnet:
            model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
            # model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
            # model = models.resnet18(pretrained=True)

        else:
            model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')



    if args.task == 'seg':
        data = torch.ones((4, 3, 513, 513))
        model = DeepLab(sync_bn=False)
        state_dict = torch.load('modeling/segmentation/deeplab-mobilenet.pth.tar')['state_dict']
        model.load_state_dict(state_dict)




    if args.sizedisp:
        print_model_size(model)

    # data = torch.ones((4, 3, 513, 513))#.cuda()

    # model = resnet_v1(num_layers=152)
    # # state_dict = torch.load('modeling/segmentation/deeplab-mobilenet.pth.tar')['state_dict']
    # state_dict = torch.load('modeling/segmentation/res152_faster_rcnn_iter_1190000.pth')
    # model.load_state_dict(state_dict)

    
    
    # if args.beforeafter:

        # if args.gpu:
        #     model_tmp = model.cuda()
        # else:
        #     model_tmp = model.cpu()

        # model_tmp.eval()
        # start_time = time.time()
        # accc=inference_all(model_tmp,args.gpu)
        # end_time = time.time()
        # print(f"Inference time is {end_time - start_time} seconds")

    model=model.cpu()
    model.eval()

    transformer = TorchTransformer()
    module_dict = {}
    if args.quantize:

        module_dict[1] = [(nn.Conv2d, QuantConv2d), (nn.Linear, QuantLinear)]

    
    if args.relu:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'graph_cls', graph_size=120)

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)

    graph = transformer.log.getGraph()

    if args.visualize :
        graph_tmp = graph
        plt_keys = list(graph_tmp.keys())
        
        # Check if the 18th layer exists and is a convolutional or linear layer in both graphs
        if len(plt_keys) > 17 and isinstance(graph_tmp[plt_keys[17]], (nn.Conv2d, nn.Linear,QuantConv2d, QuantLinear)) :

            # Get the weights of the 18th layer from both graphs
            weights_tmp = graph_tmp[plt_keys[17]].weight.detach().cpu().numpy().flatten()

    bottoms = transformer.log.getBottoms()

    # print(graph)
    # print(len(graph))
    if args.visualize:
        boxplt_and_hist_graph_weights(graph, title='Before dfq')




    if args.quantize:

        targ_layer = [QuantConv2d, QuantLinear]
 
    else:
        targ_layer = [nn.Conv2d, nn.Linear]


    # if args.quantize:
    #     # set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

    model = merge_batchnorm(model, graph, bottoms, targ_layer)



   


    #create relations
    if args.equalize :
        res = create_relation(graph, bottoms, targ_layer, delete_single=False)

        cross_layer_equalization(graph, res, targ_layer, Save_state=False, Treshhold=2e-7,)
       
        if args.visualize:
            boxplt_and_hist_graph_weights(graph, title='After equalization')


    if args.absorption:
        bias_absorption(graph, res, bottoms, N=3, visualize=args.visualize)
        if args.visualize:

            boxplt_and_hist_graph_weights(graph, title='After absorption')




    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

        model = merge_batchnorm(model, graph, bottoms, targ_layer)





            # if args.quantize:
        # if not args.trainable :
        graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)

        # else:
        set_quant_minmax(graph, bottoms)

        torch.cuda.empty_cache()
        if args.visualize:
            boxplt_and_hist_graph_weights(graph, title='After quantization')
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)





    if args.correction:

        bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)

        if args.visualize:
            # boxplt_and_hist_graph_weights(graph, title='After bias correction/ final')
            plot_histograms_sub(graph_tmp[plt_keys[17]].weight.detach().cpu().numpy().flatten(),weights_tmp)

  



        # boxplt_and_hist_graph_weights(graph, title='final')

    # torch.save(model.state_dict(), 'model9.pth')





    if args.gpu :
        model = model.cuda()
    model.eval()

    if args.quantize:
        replace_op()

    if args.sizedisp:
        print_model_size(model)

    # x = torch.rand(1, 3, 224, 224)

    # # Export the onnx model
    # torch_out = torch.onnx._export(model, x, "modelxxxxx.onnx", export_params=True)


    start_time = time.time()
    acc=inference_all(model,args.gpu)
    end_time = time.time()
    print(f"Inference time is {end_time - start_time} seconds")
    # print("Acc: {}".format(acc))


    if args.quantize:
        restore_op()
    if args.log:
        with open("cls_result.txt", 'a+') as ww:
            ww.write("resnet: {}, quant: {}, relu: {}, equalize: {}, absorption: {}, correction: {}, clip: {}\n".format(
                args.resnet, args.quantize, args.relu, args.equalize, args.absorption, args.correction, args.clip_weight
            ))
            ww.write("Acc: {}\n\n".format(acc))


if __name__ == '__main__':
    main()