def clip_weight(graph, range_clip=None, targ_type=[nn.Conv2d, nn.Linear]):
    """
    Clips the weights of specified layers in the graph to a given range.

    Args:
        graph (dict): The neural network represented as a dictionary of layers.
        range_clip (list, optional): A two-element list specifying the min and max values for clipping.
                                     Defaults to [-15, 15] if not provided.
        targ_type (list, optional): List of layer types to apply weight clipping. 
                                    Defaults to [nn.Conv2d, nn.Linear].

    Note: The function modifies the layer weights in-place.
    """
    # Default values if not provided
    if range_clip is None:
        range_clip = [-15, 15]

    # Assert that range_clip is a list or tuple of two elements
    assert isinstance(range_clip, (list, tuple)) and len(range_clip) == 2, \
        "range_clip should be a list or tuple of two elements"

    # Iterate through the graph and clip the weights
    for idx, layer in graph.items():
        if isinstance(layer, tuple(targ_type)):
            if hasattr(layer, 'weight'):
                layer.weight.data.clamp_(range_clip[0], range_clip[1])
            else:
                print(f"Warning: Layer at index {idx} does not have 'weight' attribute")
        else:
            print(f"Warning: Layer at index {idx} is not in the target type list for clipping")

