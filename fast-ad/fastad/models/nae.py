"""
NAE (Normalized Autoencoder) implementation for fastad package.

Based on the external reference implementation but simplified to integrate 
with the fastad architecture and training patterns.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from .modules import IsotropicGaussian
from ..utils import get_roc_auc_from_scores


