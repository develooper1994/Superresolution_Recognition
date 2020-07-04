"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan_train.py'
"""

# torch modules
import torch
from torch import nn
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

from train_scripts.train_loop import train_loop
from dataset.superresolution_dataset.superresolution_dataset import denormalize


class esrgan(train_loop):

    def __call__(self, *args, **kwargs):
        return self.esrgan_train()

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
                imgs_lr = imgs["lr"].to(self.device, non_blocking=True)
                imgs_hr = imgs["hr"].to(self.device, non_blocking=True)

                # Adversarial ground truths
                valid = torch.ones((imgs_lr.size(0), *self.discriminator.output_shape), requires_grad=False).to(
                    self.device, non_blocking=True)
                fake = torch.zeros((imgs_lr.size(0), *self.discriminator.output_shape), requires_grad=False).to(
                    self.device, non_blocking=True)

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = self.generator(imgs_lr)

                # Measure pixel-wise loss against ground truth
                loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)  # L1Loss

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
                real_features = self.feature_extractor(imgs_hr).detach()
                gen_features = self.feature_extractor(gen_hr)
                loss_content = self.criterion_content(gen_features, real_features)  # L1Loss

                # Total generator loss
                loss_G = loss_content + self.opt.lambda_adv * loss_GAN + self.opt.lambda_pixel * loss_pixel

                loss_G.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                loss_D = self.train_discriminator(gen_hr, imgs_hr, fake, valid)

                # --------------
                #  Log Progress
                # --------------

                self.__log_progress(i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                                    loss_pixel)

    def train_discriminator(self, gen_hr, imgs_hr, fake, valid):
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

    def __log_progress(self, i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
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
