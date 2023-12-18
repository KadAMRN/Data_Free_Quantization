# Import necessary libraries and modules
import os
import time
from ourplots import boxplt_and_hist_graph_weights, save_layer, print_model_size, plot_histograms, plot_histograms_sub
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from modeling.segmentation.deeplab import DeepLab
from torch.utils.data import DataLoader
from dataset.segmentation.pascal import VOCSegmentation
from utils.segmentation.utils import forward_all

from modeling.classification.MobileNetV2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from utils.relation import create_relation

from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer
from PyTransformer.transformers.torchTransformer import TorchTransformer

from bias_absorption import bias_absorption
from Cross_layer_equal import cross_layer_equalization
from bias_correction import bias_correction
from clip_weight import clip_weight

from utils.quantize import QuantConv2d, QuantLinear, QuantMeasure, set_layer_bits


# Function to parse command line arguments
def get_argument():
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--sizedisp", action='store_true')
    parser.add_argument("--visualize", action='store_true')

    # Transformation options
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--equalize", action='store_true')
    parser.add_argument("--correction", action='store_true')
    parser.add_argument("--absorption", action='store_true')
    parser.add_argument("--relu", action='store_true')  # Must replace relu6 to relu while equalization
    parser.add_argument("--clip_weight", action='store_true')

    # Task-specific options
    parser.add_argument("--task", default='cls', type=str, choices=['cls', 'seg'])
    parser.add_argument("--resnet", action='store_true')
    parser.add_argument("--log", action='store_true')

    # Quantization parameters
    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    parser.add_argument("--bits_bias", type=int, default=8)

    return parser.parse_args()


# Function to perform inference on the model for classification or segmentation
def inference_all(model, task, args_gpu=True):
    print("Start inference")

    if task == 'cls':
        # ImageNet dataset for classification
        imagenet_dataset = datasets.ImageFolder('./val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
        dataloader = DataLoader(imagenet_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    elif task == 'seg':
        # PASCAL VOC dataset for segmentation
        base_size = 513
        crop_size = 513
        voc_val = VOCSegmentation(base_size, crop_size, base_dir="./VOCdevkit/VOC2012/", split='val')
        dataloader = DataLoader(voc_val, batch_size=32, shuffle=False, num_workers=2)

        # Perform segmentation inference
        acc = forward_all(model, dataloader, visualize=False, opt=None)
        return acc

    if task == 'cls':
        # Classification inference
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

                # Print accuracy for each batch
                print(f"Batch {ii+1}/{len(dataloader)}: Accuracy: {num_correct/num_total*100:.2f}%")

        acc = num_correct / num_total
        print(f"Final Accuracy: {acc*100:.2f}%")
        return acc


# Main function
def main():
    # Parse command line arguments
    args = get_argument()
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'

    if args.task == 'cls':
        # Create input data for classification
        data = torch.ones((4, 3, 224, 224))  # .cuda()
        if args.resnet:
            # Use ResNet model if specified
            model = models.resnet18(pretrained=True)
        else:
            # Use MobileNetV2 model otherwise
            model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')

    elif args.task == 'seg':
        # Create input data for segmentation
        data = torch.ones((4, 3, 513, 513))
        model = DeepLab(sync_bn=False)
        state_dict = torch.load('modeling/segmentation/deeplab-mobilenet.pth.tar')['state_dict']
        model.load_state_dict(state_dict)

    if args.sizedisp:
        # Display model size information
        print_model_size(model)

    # Set model to evaluation mode and move it to CPU
    model = model.cpu()
    model.eval()

    # Create a transformer for PyTorch
    transformer = TorchTransformer()
    module_dict = {}

    if args.quantize:
        # Specify layers for quantization
        module_dict[1] = [(nn.Conv2d, QuantConv2d), (nn.Linear, QuantLinear)]

    if args.relu:
        # Specify layers for ReLU replacement
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # Switch layers in the model according to specified transformations
    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)

    # Get the computation graph from the transformer
    graph = transformer.log.getGraph()

    if args.visualize:
        # Visualize graph weights before transformations
        graph_tmp = graph
        plt_keys = list(graph_tmp.keys())

        if len(plt_keys) > 15 and isinstance(graph_tmp[plt_keys[15]], (nn.Conv2d, nn.Linear, QuantConv2d, QuantLinear)):
            weights_tmp = graph_tmp[plt_keys[15]].weight.detach().cpu().numpy().flatten()

    # Get bottom nodes of the graph
    bottoms = transformer.log.getBottoms()

    if args.visualize:
        # Visualize boxplot and histogram of graph weights before transformations
        boxplt_and_hist_graph_weights(graph, title='Before dfq')

    if args.quantize:
        # Specify target layers for merging batch normalization
        targ_layer = (QuantConv2d, QuantLinear)
    else:
        targ_layer = (nn.Conv2d, nn.Linear)

    # Merge batch normalization layers
    model = merge_batchnorm(model, graph, bottoms, targ_layer)

    if args.equalize:
        # Create relation matrix and perform cross-layer equalization
        res = create_relation(graph, bottoms, targ_layer, delete_single=False)
        cross_layer_equalization(graph, res, targ_layer, Save_state=False, Treshhold=2e-7,)

        if args.visualize:
            # Visualize graph weights after equalization
            boxplt_and_hist_graph_weights(graph, title='After equalization')
            plot_histograms_sub(weights_tmp, graph[plt_keys[15]].weight.detach().cpu().numpy().flatten())

    if args.absorption:
        # Perform bias absorption
        bias_absorption(graph, res, bottoms, N=3, visualize=args.visualize)
        if args.visualize:
            # Visualize graph weights after absorption
            boxplt_and_hist_graph_weights(graph, title='After absorption')

    if args.quantize:
        # Set bitwidths for quantization
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

        model = merge_batchnorm(model, graph, bottoms, targ_layer)

        # Quantize target layers
        graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)

        # Set quantization minmax values
        set_quant_minmax(graph, bottoms)

        # Clear GPU memory
        torch.cuda.empty_cache()
        if args.visualize:
            # Visualize graph weights after quantization
            boxplt_and_hist_graph_weights(graph, title='After quantization')

    if args.clip_weight:
        # Clip weights within a specified range
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:
        # Perform bias correction
        bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight, visualize=args.visualize)

    if args.visualize:
        # Visualize graph weights after correction
        plot_histograms_sub(weights_tmp, graph[plt_keys[15]].weight.detach().cpu().numpy().flatten())

    if args.gpu:
        # Move the model to GPU
        model = model.cuda()
    model.eval()

    if args.quantize:
        # Replace original operations after quantization
        replace_op()

    if args.sizedisp:
        # Display model size information after transformations
        print_model_size(model)

    # Measure inference time
    start_time = time.time()
    accuracy = inference_all(model, args.task, args.gpu)
    end_time = time.time()
    print(f"Inference time is {end_time - start_time} seconds")

    if args.quantize:
        # Restore original operations after inference
        restore_op()

    if args.log:
        # Log results to a file
        with open("dfq_result.txt", 'a+') as ww:
            ww.write("task: {}, resnet: {}, relu: {}, equalize: {}, absorption: {}, quantize: {}, correction: {}, clip: {}, bits_weight: {}, bits_activation: {}, bits_bias: {}\n".format(
                args.task, args.resnet, args.relu, args.equalize, args.absorption, args.quantize, args.correction,
                args.clip_weight, args.bits_weight, args.bits_activation, args.bits_bias
            ))
            ww.write("Accuracy: {} %\n\n".format(accuracy*100))


if __name__ == '__main__':
    main()
