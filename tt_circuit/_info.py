import torch


class Info(object):
    def __init__(self, data_type=torch.complex128, device="cpu"):
        self.data_type = data_type
        self.device = device
