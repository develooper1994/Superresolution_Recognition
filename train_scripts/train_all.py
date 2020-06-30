import argparse
import os
from path import Path

import numpy as np
from PIL import Image

# torch modules
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

## my modules
try:
    from dataset.superresolution_dataset.superresolution_dataset import *
    from models.esrgan_models import *
    from models import fully_conv_model
    from . import dataset, UFPR_ALPR_dataset
    from .crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
except:
    from models import fully_conv_model
    from dataset import UFPR_ALPR_dataset
    from train_scripts.crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter

torch.manual_seed(0)
plt.style.use('seaborn')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
