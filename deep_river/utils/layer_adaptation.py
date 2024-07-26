import re
from typing import Callable

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


PARAM_SHAPES = {
    nn.Linear: {
        "weight": "(o,i)",
        "bias": "(o)",
        "in_features": "I",
        "out_features": "O",
    },
    nn.LSTM: get_lstm_param_shapes,
}

SUPPORTED_LAYERS = tuple(PARAM_SHAPES.keys())


def check_shape_str(shape_str):
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
    axes = {"input": [], "output": []}
    for idx, axis_str in enumerate(axis_strs):
        input_output = re.findall(r"o|i", axis_str)

        if input_output:
            numbers = re.findall(r"\d+", axis_str)
            n_subparams = int(numbers[0]) if numbers else 1
            target = "output" if input_output[0] == "o" else "input"
            axes[target].append({"axis": idx, "n_subparams": n_subparams})

    return axes


def get_expansion_instructions(param_shapes: dict):
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
    output: bool,
    size: int,
    instructions: dict,
    init_fn: Callable,
):
    target_str = "output" if output else "input"
    for param_name, instruction in instructions.items():
        param = getattr(layer, param_name)
        if instruction == f"{target_str}_attribute":
            setattr(layer, param_name, param + size)
        elif isinstance(instruction, dict):
            axes = instruction[target_str]

            for axis in axes:
                param = expand_weights(
                    param, axis["axis"], size, init_fn, axis["n_subparams"]
                )
            setattr(layer, param_name, param)


class LayerExpander:
    """Utility class for expanding the input or output dimensionality of a layer. Currently, nn.Linear and nn.LSTM layers are explicitly supported. For any other layers, a dictionary that contains the names of all parameters as well as their respective shapes, expressed as functions of the input- and output dimensions, must be provided.
    These shapes are expected to resemble tuple, where each entry can contain an "o" referring to the output shape of the layer, an "i" referring to the input shape, or an "e" for axes that depend on neither in- or output. For the hidden-hidden weights of an LSTM, whose shape is given by (4*hidden_size, hidden_size), the correct string would be "(4o, o)", while for the hidden-hidden bias the correct string would be "(4o)". The dict must also contain the names of any integer attributes specifying input and output sizes as keys, and values of "O" for output related attributes or "I" for input related attributes.
    As an example, complete shape specification for a basic, single layer LSTM, would be
    {
        "hidden_size": "O",
        "input_size": "I",
        "weight_ih_l0": "(4o,i)",
        "weight_hh_l0": "(4o,o)",
        "bias_ih_l0": "(4o)",
        "bias_hh_l0": "(4o)"
    }

    Parameters
    ----------
    layer
        The layer to be expanded.
    param_shapes
        The shapes of all parameters of the layer, specified in the form described above.
    init_fn
        Function that will be used to initialize the new weights. The function must take a tensor as an input and modify it in-place.
    """

    def __init__(
        self,
        layer: nn.Module,
        param_shapes: dict = None,
        init_fn: Callable = nn.init.normal_,
    ) -> None:
        self.layer = layer
        self.instructions = None
        self.input_dim_key = None
        self.output_dim_key = None
        self.param_shapes = param_shapes
        self.init_fn = init_fn

    def load_instructions(self):
        if not self.param_shapes:
            self.param_shapes = PARAM_SHAPES[type(self.layer)]
        if isinstance(self.param_shapes, Callable):
            self.param_shapes = self.param_shapes(self.layer)
        self.instructions = get_expansion_instructions(self.param_shapes)

    def get_input_dim(self):
        if self.instructions is None:
            self.load_instructions()
        if self.input_dim_key is None:
            keys = list(self.instructions.keys())
            values = list(self.instructions.values())
            self.input_dim_key = keys[values.index("input_attribute")]

        return getattr(self.layer, self.input_dim_key)

    def get_output_dim(self):
        if self.output_dim_key is None:
            keys = list(self.instructions.keys())
            values = list(self.instructions.values())
            self.output_dim_key = keys[values.index("O")]

        return getattr(self.layer, self.output_dim_key)

    def expand_input(self, size: int):
        if self.instructions is None:
            self.load_instructions()
        expand_layer(
            self.layer,
            output=False,
            size=size,
            instructions=self.instructions,
            init_fn=self.init_fn,
        )

    def expand_output(self, size: int):
        if self.instructions is None:
            self.load_instructions()
        expand_layer(
            self.layer,
            output=True,
            size=size,
            instructions=self.instructions,
            init_fn=self.init_fn,
        )
