from torch import nn


def find_output_layer(net: nn.Sequential) -> nn.Linear:
    """Return the output layer of a network.

    Parameters
    ----------
    net
        The network to find the output layer of.

    Returns
    -------
    nn.Linear
        The output layer of the network.
    """

    for layer in list(net.children())[::-1]:
        if isinstance(layer, nn.Linear):
            return layer
    raise ValueError("No dense layer found.")
