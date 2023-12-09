import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from modeling.classification.MobileNetV2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction, _quantize_error, clip_weight
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer#, LayerTransform
from PyTransformer.transformers.torchTransformer import TorchTransformer

from utils.quantize import QuantConv2d, QuantLinear, QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action='store_false')
    parser.add_argument("--equalize", action='store_false')

    parser.add_argument("--correction", action='store_false')
    parser.add_argument("--absorption", action='store_false')
    parser.add_argument("--relu", action='store_false') # must replace relu6 to relu while equalization'
    parser.add_argument("--clip_weight", action='store_true')
    parser.add_argument("--trainable", action='store_true')

    parser.add_argument("--resnet", action='store_true')
    parser.add_argument("--log", action='store_false')

    # quantize params
    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    parser.add_argument("--bits_bias", type=int, default=8)

    return parser.parse_args()


def inference_all(model):
    print("Start inference")
    imagenet_dataset = datasets.ImageFolder('/home/jakc4103/WDesktop/dataset/ILSVRC/Data/CLS-LOC/val', transforms.Compose([
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
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].cuda(), sample[1].numpy()
            logits = model(image)

            pred = torch.max(logits, 1)[1].cpu().numpy()
            
            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            # print(num_correct, num_total, num_correct/num_total)
    acc = num_correct / num_total
    return acc


def main():
    args = get_argument()
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'

    data = torch.ones((4, 3, 224, 224))#.cuda()

    if args.resnet:
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
    else:
        model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')
    model.eval()
    


    transformer = TorchTransformer()
    module_dict = {}
    if args.quantize:

        if args.trainable:
            module_dict[1] = [(nn.Conv2d, QuantConv2d), (nn.Linear, QuantLinear)]
        else:
            module_dict[1] = [(nn.Conv2d, QuantNConv2d), (nn.Linear, QuantNLinear)]
    
    if args.relu:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'graph_cls', graph_size=120)

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    if args.quantize:

        if args.trainable:
            targ_layer = [QuantConv2d, QuantLinear]
        else:
            targ_layer = [QuantNConv2d, QuantNLinear]
    else:
        targ_layer = [nn.Conv2d, nn.Linear]

    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

    model = merge_batchnorm(model, graph, bottoms, targ_layer)

    #create relations
    if args.equalize :
        res = create_relation(graph, bottoms, targ_layer, delete_single=False)
        if args.equalize:
            cross_layer_equalization(graph, res, targ_layer, visualize_state=False, converge_thres=2e-7)


    
    if args.absorption:
        bias_absorption(graph, res, bottoms, 3)
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:

        bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)

    if args.quantize:
        if not args.trainable :
            graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)

        else:
            set_quant_minmax(graph, bottoms)

        torch.cuda.empty_cache()



    model = model.cuda()
    model.eval()

    if args.quantize:
        replace_op()
    acc = inference_all(model)
    print("Acc: {}".format(acc))
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