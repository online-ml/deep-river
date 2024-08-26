import re
from collections.abc import Callable
from typing import Any, Dict, Union

import torch
from torch import nn


def get_lstm_param_shapes(lstm: nn.LSTM):
    param_shapes = {
        "hidden_size": "O",
        "input_size": "I",
    }
    if lstm.bidirectional:
        num_directions = 2
        direction_suffixes = ["", "_reverse"]
    else:
        num_directions = 1
        direction_suffixes = [""]

    for d in direction_suffixes:
        for k in range(lstm.num_layers):
            param_shapes[f"weight_ih_l{k}{d}"] = (
                f"(4o,{num_directions}{'o' if lstm.proj_size == 0 else 'e'})"
            )
            param_shapes[f"weight_hh_l{k}{d}"] = (
                f"(4o,{'o' if lstm.proj_size == 0 else 'e'})"
            )
            param_shapes[f"bias_ih_l{k}{d}"] = "(4o)"
            param_shapes[f"bias_hh_l{k}{d}"] = "(4o)"
            if lstm.proj_size > 0:
                param_shapes[f"weight_hr_l{k}{d}"] = "(e,o)"
        param_shapes[f"weight_ih_l0{d}"] = "(4o,i)"
    return param_shapes


PARAM_SHAPES: Dict[type, Union[Dict[Any, Any], Callable[..., Any]]] = {
    nn.Linear: {
        "weight": "(o,i)",
        "bias": "(o)",
        "in_features": "I",
        "out_features": "O",
    },
    nn.LSTM: get_lstm_param_shapes,
}

SUPPORTED_LAYERS = tuple(PARAM_SHAPES.keys())


def check_shape_str(shape_str: str):
    blueprint = ""  # TODO: Write blueprint as regex
    if not re.match(blueprint, shape_str):
        raise ValueError("Invalid shape string for parameter.")


def get_in_out_axes(shape_str: str):
    """
    Returns a dictionary containing information on how a
    specific parameter's axis sizes correspond to the input
    and output dimensionality given its shape string.

    Parameters
    ----------
    shape_str
        String specifying the shape of a parameter.

    Returns
    -------
    axes
        Dictionary specifying which axes have to be
        altered to modify the input- or output
        dimensionality as well as the number of
        sub-parameters contained in the axes.

    """
    check_shape_str(shape_str)
    shape_str = shape_str.strip("()")
    axis_strs = shape_str.split(",")
    axes: Dict = {"input": [], "output": []}
    for idx, axis_str in enumerate(axis_strs):
        input_output = re.findall(r"o|i", axis_str)

        if input_output:
            numbers = re.findall(r"\d+", axis_str)
            n_subparams = int(numbers[0]) if numbers else 1
            target = "output" if input_output[0] == "o" else "input"
            axes[target].append({"axis": idx, "n_subparams": n_subparams})

    return axes


def get_expansion_instructions(
    param_shapes: Dict,
) -> Dict:
    """
    Returns a dictionary containing information on how
    each parameter of a layer contained in param_shapes
    corresponds to the input and output dimensionality
    given its shape string.

    Parameters
    ----------
    param_shapes
        Dictionary containing all parameters of a layer
        as keys and their corresponding shape strings as values.

    Returns
    -------
    instructions
        Dictionary specifying which axes of each parameter
        have to be altered to modify the input- or output
        dimensionality as well as the number of
        sub-parameters contained in the axes.

    """

    instructions = {}
    for key, shape_str in param_shapes.items():
        if shape_str == "I":
            instruction = "input_attribute"
        elif shape_str == "O":
            instruction = "output_attribute"
        else:
            instruction = get_in_out_axes(shape_str)
        instructions[key] = instruction
    return instructions


def expand_weights(
    weights: torch.Tensor,
    axis: int,
    n_dims_to_add: int,
    init_fn: Callable,
    n_subparams: int = 1,
):
    """
    Expands `weights` along the given axis by `n_dims_to_add`.
    The expanded weights are created by evenly splitting the
    original weights into its subparams and appending new weights
    to them.

    Parameters
    ----------
    weights
        Parameter to be expanded.
    axis
        Axis along which to expand the parameter.
    n_dims_to_add
        Number of dims to add to each sub-parameter within the parameter.
    init_fn
        Function to initiate the new weights with.
    n_subparams
        Number of sub-parameters contained in the parameter.

    Returns
    -------
    weights_expanded
        The expanded weights as a pytorch parameter.

    """
    shape_new_weights = list(weights.shape)
    shape_new_weights[axis] = n_dims_to_add
    all_weights = []
    for chunk in torch.chunk(weights, chunks=n_subparams, dim=axis):
        new_weights = torch.empty(
            *shape_new_weights, dtype=weights.dtype, device=weights.device
        )
        init_fn(new_weights)
        all_weights.extend([chunk, new_weights])
    weights_expanded = torch.cat(all_weights, dim=axis)
    return nn.Parameter(weights_expanded)


def expand_layer(
    layer: nn.Module,
    instructions: dict,
    target_size: int = 1,
    output: bool = True,
    init_fn: Callable = nn.init.normal_,
):

    target_str = "output" if output else "input"
    param_names = list(instructions.keys())
    param_instructions = list(instructions.values())
    pos = param_instructions.index(f"{target_str}_attribute")
    current_size = getattr(layer, param_names[pos])
    dims_to_add = target_size - current_size

    for param_name, instruction in instructions.items():
        param = getattr(layer, param_name)
        if instruction == f"{target_str}_attribute":
            setattr(layer, param_name, param + dims_to_add)
        elif isinstance(instruction, dict):
            axes = instruction[target_str]

            for axis in axes:
                param = expand_weights(
                    param,
                    axis["axis"],
                    dims_to_add,
                    init_fn,
                    axis["n_subparams"],
                )
            setattr(layer, param_name, param)


def load_instructions(
    layer: nn.Module,
    param_shapes: Dict[Any, Any] | Callable[..., Any] | None = None,
) -> Dict:

    if param_shapes is None:
        param_shapes = PARAM_SHAPES[type(layer)]
    if callable(param_shapes):
        param_shapes = param_shapes(layer)
    if isinstance(param_shapes, dict):
        return get_expansion_instructions(param_shapes=param_shapes)
    else:
        raise TypeError("param_shapes must be a dictionary")


def get_input_dim(
    layer: nn.Module, instructions: Dict, input_dim_key: str | None = None
):
    if input_dim_key is None:
        keys = list(instructions.keys())
        values = list(instructions.values())
        input_dim_key = keys[values.index("input_attribute")]
    return getattr(layer, input_dim_key)


def get_output_dim(
    layer: nn.Module, instructions: Dict, output_dim_key: str | None = None
):
    if output_dim_key is None:
        keys = list(instructions.keys())
        values = list(instructions.values())
        output_dim_key = keys[values.index("O")]
    return getattr(layer, output_dim_key)
