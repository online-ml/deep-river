from typing import List

from torch import nn


class ForwardOrderTracker:
    def __init__(self) -> None:
        self.ordered_modules: List[nn.Module] = []

    def __call__(self, module, input, output):
        if list(module.parameters()) and not list(module.children()):
            self.ordered_modules.append(module)


def apply_hooks(module, hook, handles=[]):
    for child in module.children():
        apply_hooks(child, hook, handles)
    handle = module.register_forward_hook(hook)
    handles.append(handle)
    return handles
