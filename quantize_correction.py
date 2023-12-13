from quantize_error import quantize_error
from utils.layer_transform import find_prev_bn
from scipy.stats import norm
import torch
import torch.nn as nn
from scipy.stats import norm
from utils.layer_transform import find_prev_bn

def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d, signed=False):
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
                # Processus de correction du biais
                bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot)
                weight = layer.weight.detach().clone()
                eps = quantize_error(weight, bits_weight, reduction=None, signed=signed)
                eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

                bn_res = process_bn_branches(bn_list, relu_attach_list, connect_type_list)

                for key, (connect_type, expect) in bn_res.items():
                    # Ajouter la logique pour la correction du biais ici...
                    num_group = expect.size(0) // eps.size(1)
                    step_size_o = eps.size(0) // num_group
                    step_size_i = expect.size(0) // num_group

                    bias = torch.zeros(eps.size(0))
                    for g in range(num_group):
                        bias[g * step_size_o:(g + 1) * step_size_o] = torch.matmul(eps[g * step_size_o:(g + 1) * step_size_o],
                                                                                   expect[g * step_size_i:(g + 1) * step_size_i])

                    if layer.bias is None:
                        layer.bias = nn.Parameter(data=torch.zeros((layer.weight.size(0)), dtype=torch.float32),
                                                 requires_grad=False)
                    layer.bias.add_(-bias)
                    bias_prev = -bias

def process_bn_branches(bn_list, relu_attach_list, connect_type_list):
    bn_branch = {}
    for idx, tmp in enumerate(bn_list):
        _, bid = tmp
        if bid[0] in bn_branch:
            bn_branch[bid[0]].append((tmp, relu_attach_list[idx], connect_type_list[idx]))
        else:
            bn_branch[bid[0]] = [(tmp, relu_attach_list[idx], connect_type_list[idx])]
    bn_res = {}
    for key in bn_branch:
        tmp_list = sorted(bn_branch[key], key=lambda x: len(x[0][1]), reverse=True)
        node_cur, use_relu, connect_type = tmp_list[0]
        layer_cur, bid = node_cur
        depth = len(bid)
        tmp_list.pop(0)
        bn_bias = layer_cur.fake_bias.detach().clone()
        bn_weight = layer_cur.fake_weight.detach().clone()

        if use_relu:
            expect = calculate_mean(bn_weight, bn_bias)
            expect[expect < 0] = 0
        else:
            expect = bn_bias

        while len(tmp_list) > 0:
            idx_bound = 0

            while idx_bound < len(tmp_list) and len(tmp_list[idx_bound][0][1]) == depth:
                idx_bound += 1

            if idx_bound == 0 and len(tmp_list) > 0:
                # cut depth, add node_cur back
                depth = len(tmp_list[idx_bound][0][1])

            else:
                for idx in range(idx_bound):
                    node_tmp, use_relu_tmp, connect_type = tmp_list[idx]
                    bn_bias = node_tmp[0].fake_bias.detach().clone()
                    bn_weight = node_tmp[0].fake_weight.detach().clone()

                    if use_relu_tmp:
                        expect_tmp = calculate_mean(bn_weight, bn_bias)
                        expect_tmp[expect_tmp < 0] = 0
                    else:
                        expect_tmp = bn_bias

                    if 'cat' == connect_type:
                        expect = torch.cat([expect, expect_tmp], 0)

                    else:
                        expect += expect_tmp

                tmp_list = tmp_list[idx_bound:]

        bn_res[key] = (connect_type, expect)

    return bn_res

# ...
import torch
import torch.nn as nn
from scipy.stats import norm
from utils.layer_transform import find_prev_bn

def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d, signed=False):
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
                # Processus de correction du biais
                bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot)
                weight = layer.weight.detach().clone()
                eps = quantize_error(weight, bits_weight, reduction=None, signed=signed)
                eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

                bn_res = process_bn_branches(bn_list, relu_attach_list, connect_type_list)

                for key, (connect_type, expect) in bn_res.items():
                    # Ajouter la logique pour la correction du biais ici...
                    num_group = expect.size(0) // eps.size(1)
                    step_size_o = eps.size(0) // num_group
                    step_size_i = expect.size(0) // num_group

                    bias = torch.zeros(eps.size(0))
                    for g in range(num_group):
                        bias[g * step_size_o:(g + 1) * step_size_o] = torch.matmul(eps[g * step_size_o:(g + 1) * step_size_o],
                                                                                   expect[g * step_size_i:(g + 1) * step_size_i])

                    if layer.bias is None:
                        layer.bias = nn.Parameter(data=torch.zeros((layer.weight.size(0)), dtype=torch.float32),
                                                 requires_grad=False)
                    layer.bias.add_(-bias)
                    bias_prev = -bias

def process_bn_branches(bn_list, relu_attach_list, connect_type_list):
    bn_branch = {}
    for idx, tmp in enumerate(bn_list):
        _, bid = tmp
        if bid[0] in bn_branch:
            bn_branch[bid[0]].append((tmp, relu_attach_list[idx], connect_type_list[idx]))
        else:
            bn_branch[bid[0]] = [(tmp, relu_attach_list[idx], connect_type_list[idx])]
    bn_res = {}
    for key in bn_branch:
        tmp_list = sorted(bn_branch[key], key=lambda x: len(x[0][1]), reverse=True)
        node_cur, use_relu, connect_type = tmp_list[0]
        layer_cur, bid = node_cur
        depth = len(bid)
        tmp_list.pop(0)
        bn_bias = layer_cur.fake_bias.detach().clone()
        bn_weight = layer_cur.fake_weight.detach().clone()

        if use_relu:
            expect = calculate_mean(bn_weight, bn_bias)
            expect[expect < 0] = 0
        else:
            expect = bn_bias

        while len(tmp_list) > 0:
            idx_bound = 0

            while idx_bound < len(tmp_list) and len(tmp_list[idx_bound][0][1]) == depth:
                idx_bound += 1

            if idx_bound == 0 and len(tmp_list) > 0:
                # cut depth, add node_cur back
                depth = len(tmp_list[idx_bound][0][1])

            else:
                for idx in range(idx_bound):
                    node_tmp, use_relu_tmp, connect_type = tmp_list[idx]
                    bn_bias = node_tmp[0].fake_bias.detach().clone()
                    bn_weight = node_tmp[0].fake_weight.detach().clone()

                    if use_relu_tmp:
                        expect_tmp = calculate_mean(bn_weight, bn_bias)
                        expect_tmp[expect_tmp < 0] = 0
                    else:
                        expect_tmp = bn_bias

                    if 'cat' == connect_type:
                        expect = torch.cat([expect, expect_tmp], 0)

                    else:
                        expect += expect_tmp

                tmp_list = tmp_list[idx_bound:]

        bn_res[key] = (connect_type, expect)

    return bn_res

# ...
import torch
import torch.nn as nn
from scipy.stats import norm
from utils.layer_transform import find_prev_bn

def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d, signed=False):
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
                # Processus de correction du biais
                bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot)
                weight = layer.weight.detach().clone()
                eps = quantize_error(weight, bits_weight, reduction=None, signed=signed)
                eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

                bn_res = process_bn_branches(bn_list, relu_attach_list, connect_type_list)

                for key, (connect_type, expect) in bn_res.items():
                    # Ajouter la logique pour la correction du biais ici...
                    num_group = expect.size(0) // eps.size(1)
                    step_size_o = eps.size(0) // num_group
                    step_size_i = expect.size(0) // num_group

                    bias = torch.zeros(eps.size(0))
                    for g in range(num_group):
                        bias[g * step_size_o:(g + 1) * step_size_o] = torch.matmul(eps[g * step_size_o:(g + 1) * step_size_o],
                                                                                   expect[g * step_size_i:(g + 1) * step_size_i])

                    if layer.bias is None:
                        layer.bias = nn.Parameter(data=torch.zeros((layer.weight.size(0)), dtype=torch.float32),
                                                 requires_grad=False)
                    layer.bias.add_(-bias)
                    bias_prev = -bias

def process_bn_branches(bn_list, relu_attach_list, connect_type_list):
    bn_branch = {}
    for idx, tmp in enumerate(bn_list):
        _, bid = tmp
        if bid[0] in bn_branch:
            bn_branch[bid[0]].append((tmp, relu_attach_list[idx], connect_type_list[idx]))
        else:
            bn_branch[bid[0]] = [(tmp, relu_attach_list[idx], connect_type_list[idx])]
    bn_res = {}
    for key in bn_branch:
        tmp_list = sorted(bn_branch[key], key=lambda x: len(x[0][1]), reverse=True)
        node_cur, use_relu, connect_type = tmp_list[0]
        layer_cur, bid = node_cur
        depth = len(bid)
        tmp_list.pop(0)
        bn_bias = layer_cur.fake_bias.detach().clone()
        bn_weight = layer_cur.fake_weight.detach().clone()

        if use_relu:
            expect = calculate_mean(bn_weight, bn_bias)
            expect[expect < 0] = 0
        else:
            expect = bn_bias

        while len(tmp_list) > 0:
            idx_bound = 0

            while idx_bound < len(tmp_list) and len(tmp_list[idx_bound][0][1]) == depth:
                idx_bound += 1

            if idx_bound == 0 and len(tmp_list) > 0:
                # cut depth, add node_cur back
                depth = len(tmp_list[idx_bound][0][1])

            else:
                for idx in range(idx_bound):
                    node_tmp, use_relu_tmp, connect_type = tmp_list[idx]
                    bn_bias = node_tmp[0].fake_bias.detach().clone()
                    bn_weight = node_tmp[0].fake_weight.detach().clone()

                    if use_relu_tmp:
                        expect_tmp = calculate_mean(bn_weight, bn_bias)
                        expect_tmp[expect_tmp < 0] = 0
                    else:
                        expect_tmp = bn_bias

                    if 'cat' == connect_type:
                        expect = torch.cat([expect, expect_tmp], 0)

                    else:
                        expect += expect_tmp

                tmp_list = tmp_list[idx_bound:]

        bn_res[key] = (connect_type, expect)

    return bn_res

# ...
import torch
import torch.nn as nn
from scipy.stats import norm
from utils.layer_transform import find_prev_bn

def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d, signed=False):
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
                # Processus de correction du biais
                bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot)
                weight = layer.weight.detach().clone()
                eps = quantize_error(weight, bits_weight, reduction=None, signed=signed)
                eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

                bn_res = process_bn_branches(bn_list, relu_attach_list, connect_type_list)

                for key, (connect_type, expect) in bn_res.items():
                    # Ajouter la logique pour la correction du biais ici...
                    num_group = expect.size(0) // eps.size(1)
                    step_size_o = eps.size(0) // num_group
                    step_size_i = expect.size(0) // num_group

                    bias = torch.zeros(eps.size(0))
                    for g in range(num_group):
                        bias[g * step_size_o:(g + 1) * step_size_o] = torch.matmul(eps[g * step_size_o:(g + 1) * step_size_o],
                                                                                   expect[g * step_size_i:(g + 1) * step_size_i])

                    if layer.bias is None:
                        layer.bias = nn.Parameter(data=torch.zeros((layer.weight.size(0)), dtype=torch.float32),
                                                 requires_grad=False)
                    layer.bias.add_(-bias)
                    bias_prev = -bias

def process_bn_branches(bn_list, relu_attach_list, connect_type_list):
    bn_branch = {}
    for idx, tmp in enumerate(bn_list):
        _, bid = tmp
        if bid[0] in bn_branch:
            bn_branch[bid[0]].append((tmp, relu_attach_list[idx], connect_type_list[idx]))
        else:
            bn_branch[bid[0]] = [(tmp, relu_attach_list[idx], connect_type_list[idx])]
    bn_res = {}
    for key in bn_branch:
        tmp_list = sorted(bn_branch[key], key=lambda x: len(x[0][1]), reverse=True)
        node_cur, use_relu, connect_type = tmp_list[0]
        layer_cur, bid = node_cur
        depth = len(bid)
        tmp_list.pop(0)
        bn_bias = layer_cur.fake_bias.detach().clone()
        bn_weight = layer_cur.fake_weight.detach().clone()

        if use_relu:
            expect = calculate_mean(bn_weight, bn_bias)
            expect[expect < 0] = 0
        else:
            expect = bn_bias

        while len(tmp_list) > 0:
            idx_bound = 0

            while idx_bound < len(tmp_list) and len(tmp_list[idx_bound][0][1]) == depth:
                idx_bound += 1

            if idx_bound == 0 and len(tmp_list) > 0:
                # cut depth, add node_cur back
                depth = len(tmp_list[idx_bound][0][1])

            else:
                for idx in range(idx_bound):
                    node_tmp, use_relu_tmp, connect_type = tmp_list[idx]
                    bn_bias = node_tmp[0].fake_bias.detach().clone()
                    bn_weight = node_tmp[0].fake_weight.detach().clone()

                    if use_relu_tmp:
                        expect_tmp = calculate_mean(bn_weight, bn_bias)
                        expect_tmp[expect_tmp < 0] = 0
                    else:
                        expect_tmp = bn_bias

                    if 'cat' == connect_type:
                        expect = torch.cat([expect, expect_tmp], 0)

                    else:
                        expect += expect_tmp

                tmp_list = tmp_list[idx_bound:]

        bn_res[key] = (connect_type, expect)

    return bn_res

# ...
