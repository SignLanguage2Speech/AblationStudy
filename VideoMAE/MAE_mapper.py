from torch import nn
import torch

def get_MAE_mapper(CFG):
    return  nn.Sequential(
                nn.Linear(1305, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            ).to(CFG.device)
