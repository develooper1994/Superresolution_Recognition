"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan_train.py'
"""

import os
import argparse

# torch modules
import torch
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from dataset.superresolution_dataset.superresolution_dataset import *
    from models import *
except:
    try:
        from train_scripts.esrgan_train import *
        from train_scripts.esrgan_train import *
    except:
        from implementations.esrgan.datasets import *
        from implementations.esrgan.models import *


class esrgan:
    def __init__(self, epoch=0, n_epochs=200, dataset_name="../dataset/TurkishPlates", batch_size=4, lr=0.0002, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=8, hr_height=256, hr_width=256,
                 channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23, warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        os.makedirs("images/training", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)

        # self.epoch = epoch
        # self.n_epochs = n_epochs
        # self.dataset_name = dataset_name
        # self.batch_size = batch_size
        # self.lr = lr
        # self.b1 = b1
        # self.b2 = b2
        # self.decay_epoch = decay_epoch
        # self.n_cpu = n_cpu
        # self.hr_height = hr_height
        # self.hr_width = hr_width
        # self.channels = channels
        # self.sample_interval = sample_interval
        # self.checkpoint_interval = checkpoint_interval
        # self.residual_blocks = residual_blocks
        # self.warmup_batches = warmup_batches
        # self.lambda_adv = lambda_adv
        # self.lambda_pixel = lambda_pixel

        self.opt = self.__commandline_interface(epoch=epoch, n_epochs=n_epochs, dataset_name=dataset_name,
                                                batch_size=batch_size, lr=lr, b1=b1, b2=b2, decay_epoch=decay_epoch,
                                                n_cpu=n_cpu, hr_height=hr_height, hr_width=hr_width, channels=channels,
                                                sample_interval=sample_interval, checkpoint_interval=checkpoint_interval,
                                                residual_blocks=residual_blocks, warmup_batches=warmup_batches,
                                                lambda_adv=lambda_adv, lambda_pixel=lambda_pixel)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize generator and discriminator
        hr_shape = (self.opt.hr_height, self.opt.hr_width)
        self.discriminator, self.feature_extractor, self.generator = self.esrgan_network_initializer(hr_shape)
        # Losses
        self.criterion_GAN, self.criterion_content, self.criterion_pixel = self.esrgan_losses()
        # Optimizers
        self.optimizer_D, self.optimizer_G = self.esrga_optimizers()
        # Data
        self.superres_dataset = ImageDataset_superresolution(self.opt.dataset_name, hr_shape=hr_shape)
        self.dataloader = DataLoader(
            self.superres_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
        )

    def __call__(self, *args, **kwargs):
        return self.esrgan_train()

    def __commandline_interface(self, epoch=0, n_epochs=200, dataset_name="dataset", batch_size=4, lr=0.0002, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=8, hr_height=256, hr_width=256,
                                channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23, warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch", type=int, default=epoch, help="epoch to start training from")
        parser.add_argument("--n_epochs", type=int, default=n_epochs, help="number of epochs of training")
        parser.add_argument("--dataset_name", type=str, default=dataset_name, help="name of the dataset")  # img_align_celeba
        parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")  # 4
        parser.add_argument("--lr", type=float, default=lr, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=b1, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=b2, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=decay_epoch, help="epoch from which to start lr decay")
        parser.add_argument("--n_cpu", type=int, default=n_cpu, help="number of cpu threads to use during batch generation")
        parser.add_argument("--hr_height", type=int, default=hr_height, help="high res. image height")
        parser.add_argument("--hr_width", type=int, default=hr_width, help="high res. image width")
        parser.add_argument("--channels", type=int, default=channels, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=sample_interval, help="interval between saving image samples")
        parser.add_argument("--checkpoint_interval", type=int, default=checkpoint_interval,
                            help="batch interval between model checkpoints")  # 5000
        parser.add_argument("--residual_blocks", type=int, default=residual_blocks, help="number of residual blocks in the generator")
        parser.add_argument("--warmup_batches", type=int, default=warmup_batches, help="number of batches with pixel-wise loss only")
        parser.add_argument("--lambda_adv", type=float, default=lambda_adv, help="adversarial loss weight")
        parser.add_argument("--lambda_pixel", type=float, default=lambda_pixel, help="pixel-wise loss weight")
        opt = parser.parse_args()
        print(opt)
        return opt

    def esrgan_network_initializer(self, hr_shape):
        generator = GeneratorRRDB(self.opt.channels, filters=64, num_res_blocks=self.opt.residual_blocks).to(self.device)
        discriminator = Discriminator(input_shape=(self.opt.channels, *hr_shape)).to(self.device)
        feature_extractor = FeatureExtractor().to(self.device)
        # Set feature extractor to inference mode
        feature_extractor.eval()
        return discriminator, feature_extractor, generator

    def esrgan_losses(self):
        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(self.device)
        criterion_content = torch.nn.L1Loss().to(self.device)
        criterion_pixel = torch.nn.L1Loss().to(self.device)
        return criterion_GAN, criterion_content, criterion_pixel

    def esrga_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        return optimizer_D, optimizer_G

    def esrgan_train(self):
        if self.opt.epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % self.opt.epoch))
            self.discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % self.opt.epoch))
        self.__train()

    def __train(self):
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            for i, imgs in enumerate(self.dataloader):

                batches_done = epoch * len(self.dataloader) + i

                # Configure model input
                imgs_lr = imgs["lr"].to(self.device)
                imgs_hr = imgs["hr"].to(self.device)

                # Adversarial ground truths
                valid = torch.ones((imgs_lr.size(0), *self.discriminator.output_shape), requires_grad=False)
                fake = torch.ones((imgs_lr.size(0), *self.discriminator.output_shape), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = self.generator(imgs_lr)

                # Measure pixel-wise loss against ground truth
                loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

                if batches_done < self.opt.warmup_batches:
                    # Warm-up (pixel-wise loss only)
                    loss_pixel.backward()
                    self.optimizer_G.step()
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                        % (epoch, self.opt.n_epochs, i, len(self.dataloader), loss_pixel.item())
                    )
                    continue

                # Extract validity predictions from discriminator
                pred_real = self.discriminator(imgs_hr).detach()
                pred_fake = self.discriminator(gen_hr)

                # Adversarial loss (relativistic average GAN)
                loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr).detach()
                loss_content = self.criterion_content(gen_features, real_features)  # after many epochs throws an exception. some data has to be be inside of cpu cache. I don't know what? and why?.

                # Total generator loss
                loss_G = loss_content + self.opt.lambda_adv * loss_GAN + self.opt.lambda_pixel * loss_pixel

                loss_G.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                loss_D = self.train_discriminator(fake, gen_hr, imgs_hr, valid)

                # --------------
                #  Log Progress
                # --------------

                self.__log_progress(batches_done, epoch, gen_hr, i, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                                    loss_pixel)

    def train_discriminator(self, fake, gen_hr, imgs_hr, valid):
        self.optimizer_D.zero_grad()
        pred_real = self.discriminator(imgs_hr)
        pred_fake = self.discriminator(gen_hr.detach())
        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D

    def __log_progress(self, batches_done, epoch, gen_hr, i, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                       loss_pixel):
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                self.opt.n_epochs,
                i,
                len(self.dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            )
        )
        if batches_done % self.opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)
        if batches_done % self.opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(self.generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(self.discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)


if __name__ == "__main__":
    superres = esrgan()
    superres()
    # try:
    #     esrgan()
    # except:
    #     error("training stoped. Adjust batch_size carefully")
    #     torch.cuda.empty_cache()
