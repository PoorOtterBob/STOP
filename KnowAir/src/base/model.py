import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=24, horizon=24):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon


    @abstractmethod
    def forward(self): # (x, label=None, adj=None)
        raise NotImplementedError


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])