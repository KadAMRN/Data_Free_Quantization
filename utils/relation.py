from collections import OrderedDict
from torch.nn import BatchNorm2d, ReLU, Dropout, AvgPool2d
from utils.quantize import QConv2d, QuantMeasure

# Relation: A class to represent the relationship between layers in a neural network.
class Relation():
    def __init__(self, layer_idx_1, layer_idx_2, bn_idx_1):
        self.layer_first = layer_idx_1
        self.layer_second = layer_idx_2
        self.bn_idx = bn_idx_1
        self.S = None

        # layer_idx_1: Index of the first layer in the relation
        # layer_idx_2: Index of the second layer in the relation
        # bn_idx_1: Index of the BatchNorm layer associated with the first layer
        # S: Scale vector (initially None)

    # String representation for debugging purposes.
    def __repr__(self):
        return '({}, {})'.format(self.layer_first, self.layer_second)

    # Getter for layer indices and BatchNorm index.
    def get_idxs(self):
        return self.layer_first, self.layer_second, self.bn_idx
    # Setter for the scale vector.
    def set_scale_vec(self, S):
        if self.S is None:
            self.S = S
        else:
            self.S *= S
    # Getter for the scale vector.
    def get_scale_vec(self):
        return self.S

# Function to create relations between layers in a neural network graph.
def create_relation(graph, bottoms, targ_type=[QConv2d], delete_single=False):
    relation_dict = OrderedDict()
    # graph: Neural network graph as a dictionary.
    # bottoms: Dictionary mapping each layer to its "bottom" layers (inputs).
    # targ_type: Target layer types for forming relations.
    # delete_single: Flag to indicate whether to delete single relations.

    # Helper function to find the previous target layer in the graph.
    def _find_prev(graph, bottoms, layer_idx, targ_type, top_counter): # find previous target layer to form relations
        bot = bottoms[layer_idx]
        last_bn = None
        while len(bot) == 1 and "Data" != bot[0] and top_counter[bot[0]] == 1:
            if type(graph[bot[0]]) == BatchNorm2d:
                last_bn = bot[0]
            if type(graph[bot[0]]) in targ_type:
                return bot[0], last_bn

            elif not(type(graph[bot[0]]) in [BatchNorm2d, ReLU, QuantMeasure, AvgPool2d] or
                (type(graph[bot[0]]) == str and ("F.pad" in bot[0] or "torch.mean" in bot[0]))):
                return None, None

            bot = bottoms[bot[0]]

        return None, None

    top_counter = {} #count the number of output branches of each layer
    # Populate top_counter with the number of outputs for each layer.
    for layer_idx in graph:
        if layer_idx == "Data":
            continue
        for bot in bottoms[layer_idx]:
            if bot in top_counter:
                top_counter[bot] += 1
            else:
                top_counter[bot] = 1

    # Find relation pairs for each layer in the graph.
    for layer_idx in graph:
        if type(graph[layer_idx]) in targ_type:
            prev, bn = _find_prev(graph, bottoms, layer_idx, targ_type, top_counter)
            if prev in relation_dict:
                relation_dict.pop(prev)
            elif prev is not None:
                rel = Relation(prev, layer_idx, bn)
                relation_dict[prev] = rel

    if delete_single:
        # only take the relations with more than 3 targ_layers, ex: Conv2d->Conv2d->Conv2d,, ignore Conv2d->Conv2d (in detection task)
        tmp = list(relation_dict.values())
        res_group = []
        for rr in tmp:
            group_idx = -1
            for idx, group in enumerate(res_group):
                for rr_prev in group:
                    if rr.get_idxs()[0] == rr_prev.get_idxs()[1]:
                        group_idx = idx
                        break
            if group_idx != -1:
                res_group[group_idx].append(rr)
            else:
                res_group.append([rr])
        res = []
        for group in res_group:
            if len(group) > 1:
                res.extend(group)

        # print(len(res), len(list(relation_dict.values())))
    else:
        res = list(relation_dict.values())

    return res #list(relation_dict.values())
