import os
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
import sklearn
import time
from tqdm import tqdm
from torch.optim import Adam
from typing import Union, Type, Tuple
import urllib
import zipfile
import h5py
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VisionDataset, MNIST, FashionMNIST, CIFAR10

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
