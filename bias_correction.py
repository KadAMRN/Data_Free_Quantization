import logging
from quantize_error import quantize_error
from utils.layer_transform import find_prev_bn
from scipy.stats import norm
import torch
import numpy as np
from utils.quantize import UniformQuantize

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _calculate_bias_correction_for_branches(bn_branch, calculate_mean):
    """
    Calculate bias corrections for each branch in the network.

    Args:
        bn_branch (dict): A dictionary of branches in the network.
        calculate_mean (function): A function to calculate the mean for bias correction.

    Returns:
        dict: A dictionary containing the bias corrections for each branch.
    """

    bn_res = {}
    for key, branch in bn_branch.items():
        # Initialize a variable to store the cumulative expectation for the current branch
        cumulative_expect = None

        # Process each layer in the branch
        for layer, relu_attached, connect_type in branch:
            bn_bias = layer.fake_bias.detach().clone()
            bn_weight = layer.fake_weight.detach().clone()

            if relu_attached:
                expect = calculate_mean(bn_weight, bn_bias)
                expect[expect < 0] = 0
            else:
                expect = bn_bias

            # Handle the accumulation of expectations based on the connect type
            if connect_type == 'cat':
                if cumulative_expect is None:
                    cumulative_expect = expect
                else:
                    cumulative_expect = torch.cat([cumulative_expect, expect])
            else:  # Assuming 'add' or similar operation
                if cumulative_expect is None:
                    cumulative_expect = expect
                else:
                    cumulative_expect += expect

        # Store the final cumulative expectation and connection type for the branch
        bn_res[key] = (connect_type, cumulative_expect)

    return bn_res


def _compute_final_bias_correction(eps, bn_values):
    """
    Compute the final bias correction for a layer.

    Args:
        eps (torch.Tensor): The quantization error.
        bn_values (tuple): A tuple containing the connection type and expected values for bias correction.

    Returns:
        torch.Tensor: The computed bias correction.
    """
    connect_type, expect = bn_values

    if connect_type == 'cat':
        bias_correction = torch.cat([eps, expect])
    else:
        bias_correction = eps + expect

    return bias_correction

def _apply_bias_correction(layer, bias):
    """
    Apply the bias correction to the given layer.

    Args:
        layer (torch.nn.Module): The layer to apply the bias correction.
        bias (torch.Tensor): The bias correction to apply.
    """
    if layer.bias is None:
        layer.bias = nn.Parameter(torch.zeros_like(layer.weight, requires_grad=True))

    # Check if the bias correction can be directly applied
    if bias.size() == layer.bias.size():
        layer.bias.data.add_(bias)
    else:
        # Reshape or adjust the bias tensor
        # Example: average the bias correction if it's larger than the layer's bias
        if bias.numel() > layer.bias.data.numel():
            # Assuming bias correction for each output channel is independent
            reshaped_bias = bias.view(layer.bias.size(0), -1).mean(dim=1)
            # Before the line causing the error


            layer.bias.data.add_(reshaped_bias)
        else:
            raise ValueError("Bias correction shape mismatch that cannot be handled automatically.")


#Corrected Version

def _quantize_error(param, num_bits=8, reduction='none', signed=False):
    """
    Compute the quantization error for a parameter tensor.

    Args:
        param (Tensor): The tensor representing the parameter to be quantized.
        num_bits (int): The number of bits to use for quantization.
        reduction (str): The method for aggregating the quantization error tensor. 
                         Choices include 'sum', 'mean', 'channel', 'spatial', or 'none'.
        signed (bool): Indicates whether the quantization is signed.

    Returns:
        Tensor: The quantization error.
    """
    param = param.detach().clone()
    with torch.no_grad():
        # Apply uniform quantization
        param_quant = UniformQuantize().apply(param, num_bits, float(param.min()), float(param.max()), False, signed)
        # Calculate the quantization error
        quantization_error = param_quant - param

        # Reduce the quantization error tensor according to the specified method
        if reduction == 'sum':
            return torch.sum(torch.abs(quantization_error))
        elif reduction == 'mean':
            return torch.mean(torch.abs(quantization_error))
        elif reduction == 'channel':
            return torch.sum(torch.abs(quantization_error.view(quantization_error.size(0), -1)), dim=-1)
        elif reduction == 'spatial':
            return torch.sum(torch.abs(quantization_error.view(quantization_error.size(0), quantization_error.size(1), -1)), dim=-1)
        elif reduction == 'none' or reduction is None:
            return quantization_error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        

def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d, signed=False):
    """
    Perform bias correction on a neural network graph. This function adjusts the biases in the layers of the network
    based on statistical properties of inputs and weights. It's specifically designed for batch normalization layers
    and certain target layer types.

    Args:
        graph (dict): The neural network graph, mapping layer indices to layer objects.
        bottoms (dict): Dictionary mapping layer indices to their 'bottom' layers.
        targ_type (type or tuple of types): Layer types to target for bias correction.
        bits_weight (int, optional): Bit precision for quantization error calculation. Defaults to 8.
        bn_type (torch.nn.Module, optional): Type of batch normalization layer. Defaults to torch.nn.BatchNorm2d.
        signed (bool, optional): Indicates whether weights are signed. Defaults to False.
    """

    # Basic parameter validations
    assert isinstance(graph, dict), "Expected 'graph' to be a dictionary."
    assert isinstance(bottoms, dict), "Expected 'bottoms' to be a dictionary."
    assert isinstance(targ_type, (type, tuple)), "Expected 'targ_type' to be a type or tuple of types."

    logger.info("Starting bias correction...")

    # Helper lambda functions for statistical calculations
    standard_normal = lambda x: torch.from_numpy(norm(0, 1).pdf(x)).float()
    standard_cdf = lambda x: torch.from_numpy(norm.cdf(x)).float()
    calculate_mean = lambda weight, bias: weight * standard_normal(-bias/weight) + bias * (1 - standard_cdf(-bias/weight))

    # Initialization
    bn_module = {}
    relu_attached = {}
    bias_prev = None

    with torch.no_grad():
        for idx_layer in graph:
            bot = bottoms[idx_layer]

            if bot is None or bot[0] == 'Data':
                continue

            if isinstance(graph[idx_layer], bn_type):
                bn_module[idx_layer] = graph[idx_layer]
                relu_attached[idx_layer] = False

                if bias_prev is not None:
                    fake_bias_size = graph[idx_layer].fake_bias.size(0)
                    if bias_prev.numel() != fake_bias_size:
                        # Average bias_prev when its size is larger than fake_bias
                        bias_prev = bias_prev.view(-1, fake_bias_size).mean(dim=0)

                    graph[idx_layer].fake_bias.add_(bias_prev)
                    bias_prev = None

                continue


            # Check for ReLU layers
            if isinstance(graph[idx_layer], torch.nn.ReLU):
                if bot[0] in bn_module:
                    relu_attached[bot[0]] = True

            # Process target layer types
            if isinstance(graph[idx_layer], targ_type):
                # Find previous batch normalization layers
                bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot[:])

                # Weight processing for quantization error
                weight = graph[idx_layer].weight.detach().clone()
                eps = _quantize_error(weight, bits_weight, reduction=None, signed=signed)
                eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

                # Handling branches in the network
                bn_branch = {}
                for idx, (layer_cur, bid) in enumerate(bn_list):
                    bn_branch.setdefault(bid[0], []).append((layer_cur, relu_attach_list[idx], connect_type_list[idx]))

                # Calculate bias corrections for each branch
                bn_res = _calculate_bias_correction_for_branches(bn_branch, calculate_mean)

                # Apply bias corrections for each branch
                for connect_type, expect in bn_res.values():
                    bias = _compute_final_bias_correction(eps, (connect_type, expect))
                    try:
                        _apply_bias_correction(graph[idx_layer], bias)
                    except ValueError as e:
                        logger.error(f"Error in applying bias correction: {e}")
                        raise

                # Prepare for next iteration
                bias_prev = -bias
                
    logger.info("Bias correction completed.")

'''
Here are the roles of these functions:

_calculate_bias_correction_for_branches: This function calculates bias corrections for each branch in the network, aggregating expectations based on the connection type.

_compute_final_bias_correction: This function computes the final bias correction for a layer by combining the quantization error (eps) with the expected bias correction calculated for the branch.

_apply_bias_correction: This function applies the calculated bias correction to a given layer. This is where the shape of the bias tensor is ensured to match the shape of the layer's bias.

_quantize_error: This function computes the quantization error for a parameter tensor, which is used in calculating the bias correction.

'''