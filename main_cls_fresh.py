import os
import time
import matplotlib
import matplotlib.pyplot as plt
from boxplt_graph_weights import boxplt_and_hist_graph_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm


from utils.quantize import QuantConv2d, QuantLinear, QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits

from modeling.segmentation.deeplab import DeepLab
#from modeling.segmentation import resnet_v1

from modeling.classification.MobileNetV2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from utils.relation import create_relation
# from dfq import bias_correction, _quantize_error, clip_weight #, bias_absorption, cross_layer_equalization
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer#, LayerTransform
from PyTransformer.transformers.torchTransformer import TorchTransformer

from bias_absorption  import bias_absorption
from Cross_layer_equal import cross_layer_equalization
from bias_correction import bias_correction
from clip_weight import clip_weight 
#from utils import visualize_per_layer


def plot_bias_correction(bias_before, bias_after, layer_names=None):
    if layer_names is None:
        layer_names = bias_before.keys()

    for name in layer_names:
        if name in bias_before and name in bias_after:
            plt.figure(figsize=(10, 6))
            plt.plot(bias_before[name].cpu().numpy(), label='Before Correction')
            plt.plot(bias_after[name].cpu().numpy(), label='After Correction')
            plt.title(f'Bias Correction for {name}')
            plt.xlabel('Bias Units')
            plt.ylabel('Bias Value')
            plt.legend()
            plt.savefig(f'bias_correction_{name}.png')
            plt.show()
        else:
            print(f"No bias data available for {name}")


def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", action='store_true')

    parser.add_argument("--quantize", action='store_false')
    parser.add_argument("--equalize", action='store_true')

    parser.add_argument("--correction", action='store_false')
    parser.add_argument("--absorption", action='store_true')
    parser.add_argument("--relu", action='store_false') # must replace relu6 to relu while equalization'
    parser.add_argument("--clip_weight", action='store_false')
 
    parser.add_argument("--task", default='cls', type=str, choices=['cls', 'seg'])
    parser.add_argument("--resnet", action='store_false')
    parser.add_argument("--log", action='store_true')

    # quantize params
    parser.add_argument("--bits_weight", type=int, default=6)
    parser.add_argument("--bits_activation", type=int, default=6)
    parser.add_argument("--bits_bias", type=int, default=6)

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
            model = models.resnext101_32x8d(pretrained=True) #use pretrained model
            # model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
            # model = models.resnet18(pretrained=True)

        else:
            model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')


    if args.task == 'seg':
        data = torch.ones((4, 3, 513, 513))
        model = DeepLab(sync_bn=False)
        state_dict = torch.load('modeling/segmentation/deeplab-mobilenet.pth.tar')['state_dict']
        model.load_state_dict(state_dict)

    # data = torch.ones((4, 3, 513, 513))#.cuda()

    # model = resnet_v1(num_layers=152)
    # # state_dict = torch.load('modeling/segmentation/deeplab-mobilenet.pth.tar')['state_dict']
    # state_dict = torch.load('modeling/segmentation/res152_faster_rcnn_iter_1190000.pth')
    # model.load_state_dict(state_dict)

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
    bottoms = transformer.log.getBottoms()

    # print(graph)
    # print(len(graph))

    boxplt_and_hist_graph_weights(graph, title='Before dfq')


    #corrected

    if args.quantize:
        targ_layer = (QuantConv2d, QuantLinear)
    else:
        targ_layer = (nn.Conv2d, nn.Linear)



    # if args.quantize:
    #     # set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

    model = merge_batchnorm(model, graph, bottoms, targ_layer)



   


    #create relations
    if args.equalize :
        res = create_relation(graph, bottoms, targ_layer, delete_single=False)
        if args.equalize:

            cross_layer_equalization(graph, res, targ_layer, Save_state=False, Treshhold=2e-7)

            boxplt_and_hist_graph_weights(graph, title='After equalization')


    if args.absorption:
        bias_absorption(graph, res, bottoms, 3)

        boxplt_and_hist_graph_weights(graph, title='After absorption')




    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

        model = merge_batchnorm(model, graph, bottoms, targ_layer)

        boxplt_and_hist_graph_weights(graph, title='After quantization')
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)





    if args.correction:
        #corrected
        # Without unpacking, pass targ_layer directly
        # Print available keys
        

        #bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)
        bias_before_correction, bias_after_correction = bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)
        print("Layers with recorded bias (before correction):", bias_before_correction.keys())
        print("Layers with recorded bias (after correction):", bias_after_correction.keys())

        # Call the plotting function
        plot_bias_correction(bias_before_correction, bias_after_correction)


    if args.quantize:
        # if not args.trainable :
        # graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)

        # else:
        set_quant_minmax(graph, bottoms)

        torch.cuda.empty_cache()

        boxplt_and_hist_graph_weights(graph, title='final')


    if args.gpu :
        model = model.cuda()
    model.eval()

    if args.quantize:
        replace_op()
    acc = inference_all(model,args.gpu)
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