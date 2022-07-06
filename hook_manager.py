"""
Helper for CKA in PyTorch.
Adds hooks to modules of a given model.

Repo: https://github.com/numpee/CKA.pytorch
Author: Dongwan Kim (Github: Numpee)
Year: 2022
"""

from typing import Optional, Union, Callable, Tuple, Type, List

import torch
from torch import nn as nn
from torchvision.models.resnet import Bottleneck, BasicBlock

_HOOK_LAYER_TYPES = (
    Bottleneck, BasicBlock, nn.Conv2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.modules.batchnorm._BatchNorm)


class HookManager:
    def __init__(self, model: nn.Module, hook_fn: Optional[Union[str, Callable]] = None,
                 hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES,
                 calculate_gram: bool = True) -> None:
        """
        Add hooks to models.
        Mainly supports ResNets.
        :param model: model to attach hooks to
        :param hook_fn: the hook function or string. Options: ("avgpool", "flatten"). Default: flatten
        :param hook_layer_types: layer types to register hooks. Should be nn.Module
        """
        self.model = model
        self.hook_fn = hook_fn
        self.hook_layer_types = hook_layer_types
        self.calculate_gram = calculate_gram
        for layer in self.hook_layer_types:
            if not issubclass(layer, nn.Module):
                raise TypeError(f"Class {layer} is not an nn.Module.")

        if self.hook_fn is None:
            self.hook_fn = self.flatten_hook_fn
            print("No hook function provided. Using flatten_hook_fn.")
        elif type(self.hook_fn) == str:
            hook_fn_dict = {'flatten': self.flatten_hook_fn, 'avgpool': self.avgpool_hook_fn}
            if self.hook_fn in hook_fn_dict:
                self.hook_fn = hook_fn_dict[self.hook_fn]
            else:
                raise ValueError(f"No hook function named {self.hook_fn}. Options: {list(hook_fn_dict.keys())}")

        # Not using dictionary because a single module may be used multiple times in forward
        self.features = []
        self.module_names = []
        self.handles = []

        self.register_hooks(self.hook_fn)

    def get_features(self) -> List[torch.Tensor]:
        return self.features

    def get_module_names(self) -> List[str]:
        return self.module_names

    def clear_features(self) -> None:
        self.features = []
        self.module_names = []

    def clear_all(self) -> None:
        self.clear_hooks()
        self.clear_features()

    def clear_hooks(self) -> None:
        num_handles = len(self.handles)
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for m in self.model.modules():
            if hasattr(m, 'module_name'):
                delattr(m, 'module_name')
        print(f"{num_handles} handles removed.")

    def register_hooks(self, hook_fn: Callable) -> None:
        prev_num_handles = len(self.handles)
        self._register_hook_recursive(self.model, hook_fn, prev_name="")
        new_num_handles = len(self.handles)
        print(f"{new_num_handles - prev_num_handles} Hooks registered. Total hooks: {new_num_handles}")

    def _register_hook_recursive(self, module: nn.Module, hook_fn: Callable, prev_name: str = "") -> None:
        for name, child in module.named_children():
            curr_name = f"{prev_name}.{name}" if prev_name else name
            curr_name = curr_name.replace("_model.", "")
            num_grandchildren = len(list(child.children()))
            if num_grandchildren > 0:
                self._register_hook_recursive(child, hook_fn, prev_name=curr_name)
            if isinstance(child, self.hook_layer_types):
                handle = child.register_forward_hook(hook_fn)
                self.handles.append(handle)
                setattr(child, 'module_name', curr_name)

    def flatten_hook_fn(self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        batch_size = out.size(0)
        feature = out.reshape(batch_size, -1)
        if self.calculate_gram:
            feature = gram(feature)
        module_name = getattr(module, 'module_name')
        self.features.append(feature)
        self.module_names.append(module_name)

    def avgpool_hook_fn(self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        if out.dim() == 4:
            feature = out.mean(dim=(-1, -2))
        elif out.dim() == 3:
            feature = out.mean(dim=-1)
        else:
            feature = out
        if self.calculate_gram:
            feature = gram(feature)
        module_name = getattr(module, 'module_name')
        self.features.append(feature)
        self.module_names.append(module_name)


def gram(x: torch.Tensor) -> torch.Tensor:
    return x.matmul(x.t())
