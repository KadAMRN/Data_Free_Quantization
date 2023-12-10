import torch
import torch.nn as nn
import copy
import numpy as np
from utils.quantize import UniformQuantize

def _quantize_error(param, num_bits=8, reduction='sum', signed=False):
    """
    Calculate the quantization error of the given parameter.

    Args:
        param (Tensor): The parameter tensor to be quantized.
        num_bits (int): The number of bits to use for quantization.
        reduction (str): The method to reduce the quantization error tensor. 
                         Options are 'sum', 'mean', 'channel', 'spatial', or 'none'.
        signed (bool): Whether the quantization is signed.

    Returns:
        Tensor: The quantization error, reduced according to the specified method.
    """
    param = param.detach().clone()
    with torch.no_grad():
        # Apply uniform quantization
        param_quant = UniformQuantize().apply(param, num_bits, float(param.min()), float(param.max()), False, signed)
        # Calculate the error between quantized and original parameter
        eps = param_quant - param

        # Reduce the error tensor according to the specified method
        if reduction == 'sum':
            eps = torch.sum(torch.abs(eps))
        elif reduction == 'mean':
            eps = torch.mean(torch.abs(eps))
        elif reduction == 'channel':
            eps = torch.sum(torch.abs(eps.view(eps.size(0), -1)), dim=-1)
        elif reduction == 'spatial':
            eps = torch.sum(torch.abs(eps.view(eps.size(0), eps.size(1), -1)), dim=-1)
        # If reduction is 'none' or an unrecognized option, return the error as is
        elif reduction != 'none':
            print(f"Warning: Reduction method '{reduction}' is not recognized. Returning un-reduced error.")

        return eps
