import torch
import torch.nn as nn
import copy
import numpy as np
from utils.quantize import UniformQuantize
#Corrected Version
def quantize_error(param, original_activations, num_bits=8, reduction='sum', signed=False):
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
        # Calculate the quantization error
        quantization_error = param_quant - param
        # Calculate the expected error on the output E[ǫx]
        expected_error_output = torch.mean(quantization_error * original_activations)
        # Calculate the correction term E[~y] - E[ǫx]
        correction_term = torch.mean(original_activations) - expected_error_output
        # Subtract the correction term from the biased output ey
        corrected_output = original_activations - correction_term
        # Calculate the new quantization error after correction
        corrected_quantization_error = param_quant - corrected_output


        # Reduce the corrected quantization error tensor according to the specified method
        if reduction == 'sum':
            corrected_error = torch.sum(torch.abs(corrected_quantization_error))
        elif reduction == 'mean':
            corrected_error = torch.mean(torch.abs(corrected_quantization_error))
        elif reduction == 'channel':
            corrected_error = torch.sum(torch.abs(corrected_quantization_error.view(corrected_quantization_error.size(0), -1)), dim=-1)
        elif reduction == 'spatial':
            corrected_error = torch.sum(torch.abs(corrected_quantization_error.view(corrected_quantization_error.size(0), corrected_quantization_error.size(1), -1)), dim=-1)
        # If reduction is 'none' or an unrecognized option, return the corrected error as is
        elif reduction != 'none':
            print(f"Warning: La méthode de réduction '{reduction}' n'est pas reconnue. Retour de l'erreur corrigée non réduite.")
            corrected_error = corrected_quantization_error

        return corrected_error
        
        
