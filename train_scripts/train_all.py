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

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="dataset", help="name of the dataset")  # img_align_celeba
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")  # 4
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="batch interval between model checkpoints")  # 5000
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)