from utils.layer_transform import find_prev_bn
from scipy.stats import norm

def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d, signed=False):
    """
    Perform bias correction on the given graph.
    ...
    """
    print("Start bias correction")

    def standard_normal(x):
        return torch.from_numpy(norm(0, 1).pdf(x)).float()

    def standard_cdf(x):
        return torch.from_numpy(norm.cdf(x)).float()

    def calculate_mean(weight, bias):
        return weight * standard_normal(-bias / weight) + bias * (1 - standard_cdf(-bias / weight))

    bn_module = {}
    relu_attached = {}
    bias_prev = None

    with torch.no_grad():
        for idx_layer, layer in graph.items():
            bot = bottoms.get(idx_layer, None)
            if bot is None or bot[0] == 'Data':
                continue

            if isinstance(layer, bn_type):
                bn_module[idx_layer] = layer
                relu_attached[idx_layer] = False
                if bias_prev is not None:
                    layer.fake_bias.add_(bias_prev)
                    bias_prev = None
                continue

            if isinstance(layer, torch.nn.ReLU) and bot[0] in bn_module:
                relu_attached[bot[0]] = True

            if isinstance(layer, targ_type):
                process_target_layer(layer, graph, bn_module, relu_attached, bottoms, bot, bits_weight, signed)

def process_target_layer(layer, graph, bn_module, relu_attached, bottoms, bot, bits_weight, signed):
    """
    Process a target layer for bias correction.
    """
    bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot)
    weight = layer.weight.detach().clone()
    eps = _quantize_error(weight, bits_weight, reduction=None, signed=signed)
    eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

    bn_res = process_bn_branches(bn_list, relu_attach_list, connect_type_list)

