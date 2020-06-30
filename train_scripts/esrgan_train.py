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
    def __init__(self):
        os.makedirs("images/training", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)

        self.opt = self.commandline_interface()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize generator and discriminator
        hr_shape = (self.opt.hr_height, self.opt.hr_width)
        self.discriminator, self.feature_extractor, self.generator = self.esrgan_network_initializer(self.device, hr_shape)
        # Losses
        self.criterion_GAN, self.criterion_content, self.criterion_pixel = self.esrgan_losses(self.device)
        # Optimizers
        self.optimizer_D, self.optimizer_G = self.esrga_optimizers(self.discriminator, self.generator)
        # Data
        self.superres_dataset = ImageDataset_superresolution("../../data/%s" % self.opt.dataset_name, hr_shape=hr_shape)
        self.dataloader = DataLoader(
            self.superres_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
        )

    def __call__(self, *args, **kwargs):
        return self.esrgan_train()

    def commandline_interface(self):
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
        parser.add_argument("--checkpoint_interval", type=int, default=100,
                            help="batch interval between model checkpoints")  # 5000
        parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
        parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
        parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
        parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
        opt = parser.parse_args()
        print(opt)
        return opt

    def esrgan_network_initializer(self, device, hr_shape):
        generator = GeneratorRRDB(self.opt.channels, filters=64, num_res_blocks=self.opt.residual_blocks).to(device)
        discriminator = Discriminator(input_shape=(self.opt.channels, *hr_shape)).to(device)
        feature_extractor = FeatureExtractor().to(device)
        # Set feature extractor to inference mode
        feature_extractor.eval()
        return discriminator, feature_extractor, generator

    def esrgan_losses(self, device):
        criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        criterion_content = torch.nn.L1Loss().to(device)
        criterion_pixel = torch.nn.L1Loss().to(device)
        return criterion_GAN, criterion_content, criterion_pixel

    def esrga_optimizers(self, discriminator, generator):
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        return optimizer_D, optimizer_G

    def esrgan_train(self, hr_height=None, hr_width=None, device=None):
        if hr_height is None:
            hr_height = self.opt.hr_height
        if hr_width is None:
            hr_width = self.opt.hr_width
        if device is None:
            device = self.device
        hr_shape = (hr_height, hr_width)
        # Initialize generator and discriminator
        # discriminator, feature_extractor, generator = self.esrgan_network_initializer(device, hr_shape)
        discriminator, feature_extractor, generator = self.discriminator, self.feature_extractor, self.generator
        # Losses
        # criterion_GAN, criterion_content, criterion_pixel = self.esrgan_losses(device)
        criterion_GAN, criterion_content, criterion_pixel = self.criterion_GAN, self.criterion_content, self.criterion_pixel
        if self.opt.epoch != 0:
            # Load pretrained models
            generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % self.opt.epoch))
            discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % self.opt.epoch))
        # Optimizers
        # optimizer_D, optimizer_G = self.esrga_optimizers(discriminator, generator)
        optimizer_D, optimizer_G = self.optimizer_D, self.optimizer_G
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        # if dataset_name is None:
        #     dataset_name = self.opt.dataset_name
        # dataloader = DataLoader(
        #     ImageDataset_superresolution("../../data/%s" % dataset_name, hr_shape=hr_shape),
        #     batch_size=self.opt.batch_size,
        #     shuffle=True,
        #     num_workers=self.opt.n_cpu,
        # )
        dataloader = self.dataloader
        # ----------
        #  Training
        # ----------
        self.__train(Tensor, criterion_GAN, criterion_content, criterion_pixel, dataloader, discriminator, feature_extractor,
                     generator, optimizer_D, optimizer_G)

    def __train(self, Tensor, criterion_GAN, criterion_content, criterion_pixel, dataloader, discriminator, feature_extractor,
                generator, optimizer_D, optimizer_G):
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            for i, imgs in enumerate(dataloader):

                batches_done = epoch * len(dataloader) + i

                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)

                # Measure pixel-wise loss against ground truth
                loss_pixel = criterion_pixel(gen_hr, imgs_hr)

                if batches_done < self.opt.warmup_batches:
                    # Warm-up (pixel-wise loss only)
                    loss_pixel.backward()
                    optimizer_G.step()
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                        % (epoch, self.opt.n_epochs, i, len(dataloader), loss_pixel.item())
                    )
                    continue

                # Extract validity predictions from discriminator
                pred_real = discriminator(imgs_hr).detach()
                pred_fake = discriminator(gen_hr)

                # Adversarial loss (relativistic average GAN)
                loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

                # Content loss
                gen_features = feature_extractor(gen_hr)
                real_features = feature_extractor(imgs_hr).detach()
                loss_content = criterion_content(gen_features, real_features)

                # Total generator loss
                loss_G = loss_content + self.opt.lambda_adv * loss_GAN + self.opt.lambda_pixel * loss_pixel

                loss_G.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                loss_D = self.train_discriminator(criterion_GAN, discriminator, fake, gen_hr, imgs_hr, optimizer_D, valid)

                # --------------
                #  Log Progress
                # --------------

                self.__log_progress(batches_done, dataloader, discriminator, epoch, gen_hr, generator, i, imgs_lr, loss_D, loss_G,
                                    loss_GAN, loss_content, loss_pixel)

    def train_discriminator(self, criterion_GAN, discriminator, fake, gen_hr, imgs_hr, optimizer_D, valid):
        optimizer_D.zero_grad()
        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())
        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        return loss_D

    def __log_progress(self, batches_done, dataloader, discriminator, epoch, gen_hr, generator, i, imgs_lr, loss_D, loss_G,
                       loss_GAN, loss_content, loss_pixel):
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                self.opt.n_epochs,
                i,
                len(dataloader),
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
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)


if __name__ == "__main__":
    superres = esrgan()
    superres()
    # try:
    #     esrgan()
    # except:
    #     error("training stoped. Adjust batch_size carefully")
    #     torch.cuda.empty_cache()
