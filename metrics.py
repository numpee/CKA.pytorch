import torch
from torchmetrics import Metric


class AccumTensor(Metric):
    def __init__(self, default_value: torch.Tensor):
        super().__init__()

        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor):
        self.val += input_tensor

    def compute(self):
        return self.val
