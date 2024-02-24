import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel


class ModelBERT(nn.Module):
    def __init__(self):
        super(ModelBERT, self).__init__()
