import os
import sys
import time
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from src.pkg.CSVLogger import CSVLogger
from src.model.ModelBERT import ModelBERT
from src.MusicGenreDataset import MusicGenreDatasetWithPreprocess, _batch_to_tensor


class ModelTrainer:
    def __init__(self):
        super(ModelTrainer, self).__init__()
