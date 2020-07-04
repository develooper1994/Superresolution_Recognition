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
    from train_scripts.esrgan_train import crnn, esrgan
    from dataset.superresolution_dataset.superresolution_dataset import *
    from models.esrgan_models import *
    from models import fully_conv_model
    from . import dataset, UFPR_ALPR_dataset
    from .crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
except:
    from .esrgan_train import crnn, esrgan
    from .superresolution_dataset.superresolution_dataset import *
    from .esrgan_models import *
    from .models import fully_conv_model
    from dataset import UFPR_ALPR_dataset
    from .crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter

torch.manual_seed(0)
plt.style.use('seaborn')


class esrgan_crnn(esrgan, crnn):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reload(self, b1, b2, batch_size, channels, checkpoint_interval, dataset_name, decay_epoch, epoch, hr_height,
               hr_width, lambda_adv, lambda_pixel, lr, n_cpu, n_epochs, residual_blocks, sample_interval,
               warmup_batches):
       raise NotImplementedError

    def network_initializers(self):
        raise NotImplementedError

    def losses(self):
        raise NotImplementedError

    def optimizers(self):
        raise NotImplementedError

    def __commandline_interface(self, epoch=0, n_epochs=200, dataset_name="dataset", batch_size=4, lr=0.0002, b1=0.9,
                                b2=0.999, decay_epoch=100, n_cpu=8, hr_height=256, hr_width=256,
                                channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23,
                                warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        raise NotImplementedError

    def esrgan_crnn_train(self):
        if self.opt.epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % self.opt.epoch))
            self.discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % self.opt.epoch))
            self.ocr_model.load_state_dict(torch.load("ocr_saved/models/ocr_model_%d.pth" % self.opt.epoch))
        self.__train()

    def __train(self):
        ## Train
        n_iter = 0
        npa = self.opt.npa
        max_elem, max_preds, max_target = 0, 0, 0
        input_lengths, loss, log_probs = 0, 0, 0
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            print("Epoch:", epoch, "started")
            for i, ge in enumerate(self.dataloader):
                pass

    def __log_progress(self, i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                       loss_pixel):
        raise NotImplementedError


if __name__ == "__main__":
    combined_model = esrgan_crnn()

