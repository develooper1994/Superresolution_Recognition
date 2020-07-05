import argparse
import os

import albumentations
import cv2
from path import Path

import numpy as np
from PIL import Image

# torch modules
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchfun import count_parameters
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

## my modules
from train_scripts.crnn_train import crnn
from train_scripts.esrgan_train import esrgan
from dataset.superresolution_dataset.superresolution_dataset import ImageDataset_superresolution, denormalize
from models.esrgan_models import FeatureExtractor, DenseResidualBlock, ResidualInResidualDenseBlock, GeneratorRRDB, \
    Discriminator
from models import fully_conv_model
from dataset import UFPR_ALPR_dataset
from train_scripts.crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter


# try:
#     from train_scripts.esrgan_train import crnn, esrgan
#     from dataset.superresolution_dataset.superresolution_dataset import ImageDataset_superresolution, denormalize
#     from models.esrgan_models import FeatureExtractor, DenseResidualBlock, ResidualInResidualDenseBlock, GeneratorRRDB, \
#         Discriminator
#     from models import fully_conv_model
#     from . import dataset, UFPR_ALPR_dataset
#     from .crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
# except:
#     from .esrgan_train import crnn, esrgan
#     from .superresolution_dataset.superresolution_dataset import ImageDataset_superresolution, denormalize
#     from .esrgan_models import FeatureExtractor, DenseResidualBlock, ResidualInResidualDenseBlock, GeneratorRRDB, \
#         Discriminator
#     from .models import fully_conv_model
#     from dataset import UFPR_ALPR_dataset
#     from .crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter


class esrgan_crnn:
    def __init__(self, epoch=0, n_epochs=200, dataset_name=r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset",
                 batch_size=4, npa=1, lr=0.0002, eta_min=1e-6, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=8, hr_height=32,
                 hr_width=80, channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23,
                 warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        os.makedirs("ocr_images/training", exist_ok=True)
        os.makedirs("ocr_saved/models", exist_ok=True)

        self.reload(epoch=epoch, n_epochs=n_epochs, dataset_name=dataset_name, batch_size=batch_size, npa=npa, lr=lr,
                    eta_min=eta_min, b1=b1, b2=b2, decay_epoch=decay_epoch, n_cpu=n_cpu, hr_height=hr_height,
                    hr_width=hr_width, channels=channels, sample_interval=sample_interval,
                    checkpoint_interval=checkpoint_interval, residual_blocks=residual_blocks,
                    warmup_batches=warmup_batches, lambda_adv=lambda_adv, lambda_pixel=lambda_pixel)

    def __call__(self, *args, **kwargs):
        self.esrgan_crnn_train()

    def reload(self, epoch=0, n_epochs=200, dataset_name=r"dataset",
               batch_size=4, npa=1, lr=0.0002, eta_min=1e-6, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=4, hr_height=256,
               hr_width=256, channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23,
               warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        ## ocr
        self.parser = argparse.ArgumentParser()
        self.opt = self.__commandline_interface(epoch=epoch, n_epochs=n_epochs, dataset_name=dataset_name,
                                                batch_size=batch_size, npa=npa, lr=lr, eta_min=eta_min,
                                                b1=b1, b2=b2, decay_epoch=decay_epoch,
                                                n_cpu=n_cpu, hr_height=hr_height,
                                                hr_width=hr_width, channels=channels, sample_interval=sample_interval,
                                                checkpoint_interval=checkpoint_interval,
                                                residual_blocks=residual_blocks, warmup_batches=warmup_batches,
                                                lambda_adv=lambda_adv, lambda_pixel=lambda_pixel)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## Set up Tensorboard writer for current test
        self.writer = SummaryWriter(log_dir="../logs")  # /home/leander/AI/repos/OCR-CNN/logs2/correct_cosine_2
        ## Model
        hr_shape = (self.opt.hr_height, self.opt.hr_width)
        self.ocr_model, self.discriminator, self.feature_extractor, self.generator = self.network_initializers(hr_shape)
        ## Loss
        self.ctc_loss, self.criterion_GAN, self.criterion_content, self.criterion_pixel = self.losses()
        ## Optimizer: Good initial is 5e5
        self.ocr_optimizer, self.ocr_cosine_learning_rate_scheduler, self.optimizer_D, self.optimizer_G = self.optimizers()
        ## We keep track of the Average loss and CER
        self.ave_total_loss = AverageMeter()
        self.CER_total = AverageMeter()
        ## Dataset
        ## esrgan
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        self.lr_crnn_transforms = albumentations.Compose([
            albumentations.Resize(self.opt.hr_height // 4, self.opt.hr_width // 4),
            albumentations.Normalize(mean, std),
            ToTensor(),
        ])
        self.hr_crnn_transforms = albumentations.Compose([
            albumentations.Resize(self.opt.hr_height, self.opt.hr_width, interpolation=cv2.INTER_CUBIC),
            albumentations.Normalize(mean, std),
            ToTensor(),
        ])
        root = Path(self.opt.dataset_name)
        self.lr_crnn_dataset = UFPR_ALPR_dataset(data_path=root, dataset_type="__train",
                                                 transform=self.lr_crnn_transforms)
        self.lr_crnn_dataloader = DataLoader(
            self.lr_crnn_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
            pin_memory=True,
        )
        self.hr_crnn_dataset = UFPR_ALPR_dataset(data_path=root, dataset_type="__train",
                                                 transform=self.hr_crnn_transforms)
        self.hr_crnn_dataloader = DataLoader(
            self.hr_crnn_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
            pin_memory=True,
        )
        # testset = DataLoader(test_dataset, batch_size=batch_size)
        print("Number of parameters of OCR model", count_parameters(self.ocr_model))

        ## esrgan
        # # Data
        # # The most problematic part. If doesn't work use lr and hr transforms inside of this class
        # self.superres_dataset = ImageDataset_superresolution(device=self.device, root=self.opt.dataset_name, hr_shape=hr_shape)  # self.crnn_dataset into this dataset.
        # self.superres_dataloader = DataLoader(
        #     self.superres_dataset,
        #     batch_size=self.opt.batch_size,
        #     num_workers=self.opt.n_cpu,
        #     shuffle=True,
        #     pin_memory=True,
        # )

    def network_initializers(self, hr_shape, use_LeakyReLU_Mish=False):
        ## ocr
        ocr_model = fully_conv_model.cnn_attention_ocr(n_layers=8, nclasses=93, model_dim=64, input_dim=3)
        ocr_model = ocr_model.to(self.device, non_blocking=True).train()

        ## esrgan
        generator = GeneratorRRDB(self.opt.channels, filters=64, num_res_blocks=self.opt.residual_blocks,
                                  use_LeakyReLU_Mish=use_LeakyReLU_Mish).to(self.device, non_blocking=True)
        discriminator = Discriminator(input_shape=(self.opt.channels, *hr_shape),
                                      use_LeakyReLU_Mish=use_LeakyReLU_Mish).to(self.device, non_blocking=True)
        feature_extractor = FeatureExtractor().to(self.device, non_blocking=True)
        # Set feature extractor to inference mode
        feature_extractor.eval()

        return ocr_model, discriminator, feature_extractor, generator

    def losses(self):
        ctc_loss = nn.CTCLoss(blank=0, reduction="mean").to(self.device)
        criterion_GAN = torch.nn.BCEWithLogitsLoss()
        criterion_content = torch.nn.L1Loss()
        criterion_pixel = torch.nn.L1Loss()
        return ctc_loss, criterion_GAN, criterion_content, criterion_pixel

    def optimizers(self):
        ocr_optimizer = optim.Adam(self.ocr_model.parameters(), lr=self.opt.lr)
        ocr_cosine_learning_rate_scheduler = CosineAnnealingLR(optimizer=ocr_optimizer, T_max=250000,
                                                               eta_min=self.opt.eta_min)
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr,
                                       betas=(self.opt.b1, self.opt.b2))
        return ocr_optimizer, ocr_cosine_learning_rate_scheduler, optimizer_D, optimizer_G

    def __commandline_interface(self, epoch=0, n_epochs=200, dataset_name="dataset", batch_size=4, npa=1,
                                lr=0.0002, eta_min=1e-6, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=8, hr_height=32,
                                hr_width=80, channels=3, sample_interval=100, checkpoint_interval=100,
                                residual_blocks=23, warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        self.parser.add_argument("--epoch", type=int, default=epoch, help="epoch to start training from")
        self.parser.add_argument("--n_epochs", type=int, default=n_epochs, help="number of epochs of training")
        self.parser.add_argument("--dataset_name", type=str, default=dataset_name,
                                 help="name of the dataset")  # img_align_celeba
        self.parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")  # 4
        self.parser.add_argument("--npa", type=int, default=npa,
                                 help="maximum avereage CR(character error rate)")  # !!! explain !!!
        self.parser.add_argument("--lr", type=float, default=lr, help="adam: learning rate")
        self.parser.add_argument("--eta_min", type=float, default=eta_min, help="cosine lr scheduler rate.")
        self.parser.add_argument("--b1", type=float, default=b1, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=b2, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--decay_epoch", type=int, default=decay_epoch,
                                 help="epoch from which to start lr decay")
        self.parser.add_argument("--n_cpu", type=int, default=n_cpu,
                                 help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--hr_height", type=int, default=hr_height, help="high res. image height")
        self.parser.add_argument("--hr_width", type=int, default=hr_width, help="high res. image width")
        self.parser.add_argument("--channels", type=int, default=channels, help="number of image channels")
        self.parser.add_argument("--sample_interval", type=int, default=sample_interval,
                                 help="interval between saving image samples")
        self.parser.add_argument("--checkpoint_interval", type=int, default=checkpoint_interval,
                                 help="batch interval between model checkpoints")  # 5000
        self.parser.add_argument("--residual_blocks", type=int, default=residual_blocks,
                                 help="number of residual blocks in the generator")
        self.parser.add_argument("--warmup_batches", type=int, default=warmup_batches,
                                 help="number of batches with pixel-wise loss only")
        self.parser.add_argument("--lambda_adv", type=float, default=lambda_adv, help="adversarial loss weight")
        self.parser.add_argument("--lambda_pixel", type=float, default=lambda_pixel, help="pixel-wise loss weight")
        opt = self.parser.parse_args()
        print(opt)
        return opt

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
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            print("Epoch:", epoch, "started")
            for i, ge_lr, ge_hr in enumerate(zip(self.lr_crnn_dataloader, self.hr_crnn_dataloader)):
                batches_done = epoch * len(self.lr_crnn_dataloader) + i
                images_lr, plate_encoded, images_len, plate_encoded_len = ge_lr
                images_hr, plate_encoded, images_len, plate_encoded_len = ge_hr
                # Configure model input
                imgs_lr = images_lr.to(self.device, non_blocking=True)
                imgs_hr = images_hr.to(self.device, non_blocking=True)
                plate_encoded = plate_encoded.to(self.device, non_blocking=True)
                if imgs_lr.shape[3] <= 800 and imgs_hr.shape[3] <= 800:
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

                    # optimize ocr_model
                    n_iter, input_lengths, ocr_loss, log_probs, targets, wer_list, max_elem, max_preds, max_target = \
                        self.ocr_one_batch(batches_done, epoch, gen_hr, images_len, max_elem, max_preds,
                                           max_target, n_iter, npa, plate_encoded, plate_encoded_len)

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
                    loss_G = loss_content + self.opt.lambda_adv * loss_GAN + self.opt.lambda_pixel * loss_pixel \
                             + ocr_loss

                    loss_G.backward()
                    self.optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    loss_D = self.train_discriminator(gen_hr, imgs_hr, fake, valid)

                    # --------------
                    #  Log Progress
                    # --------------

                    # self.__esrgan_log_progress(i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN,
                    #                            loss_content,
                    #                            loss_pixel)
                    self.__log_progress(self, i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN,
                                        loss_content, loss_pixel, ocr_loss, max_target, max_preds, max_elem, n_iter,
                                        npa, wer_list=wer_list)

    def __ocr_log_progress2(self, batches_done, epoch, images, ocr_loss, max_elem, max_preds, max_target, n_iter, npa,
                            wer_list):
        self.ave_total_loss.update(ocr_loss.data.item())
        self.writer.add_scalar("total_loss", self.ave_total_loss.average(), n_iter)
        if np.average(wer_list) > 0.1:
            # Save Loss in averagemeter and write to tensorboard
            self.writer.add_text("label", max_target, n_iter)
            self.writer.add_text("pred", max_preds, n_iter)
            self.writer.add_image("img", images[max_elem].detach().cpu().numpy(), n_iter)

            # gen.close()
            # break
        # Might become infinite
        if np.average(wer_list) < 10:
            self.CER_total.update(np.average(wer_list))
            self.writer.add_scalar("CER", self.CER_total.average(), n_iter)
        # We save when the new avereage CR is beloew the NPA
        # npa>CER_total.average() and CER_total.average()>0 and CER_total.average()<1
        if npa > self.CER_total.average() > 0 and self.CER_total.average() < 1:
            torch.save(self.ocr_model.state_dict(), "ocr_autosave.pt")
            npa = self.CER_total.average()
        n_iter = n_iter + 1
        self.ocr_cosine_learning_rate_scheduler.step()
        lr = self.ocr_optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("lr", lr, n_iter)
        # Save result checkpoints
        if batches_done % self.opt.sample_interval == 0:
            save_image(images[max_elem], "ocr_images/training/%d.png" % batches_done, nrow=1, normalize=False)
        # Save model checkpoints
        if batches_done % self.opt.checkpoint_interval == 0:
            torch.save(self.ocr_model.state_dict(), "ocr_saved/models/ocr_model_%d.pth" % epoch)
        return npa, n_iter

    def __log_progress(self, i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                       loss_pixel, ocr_loss, max_target, max_preds, max_elem, n_iter, npa, wer_list):

        npa, n_iter = self.__ocr_log_progress2(batches_done, epoch, gen_hr, ocr_loss, max_elem, max_preds, max_target,
                                               n_iter, npa, wer_list)
        self.summary_string = f"[Epoch {epoch}/{self.opt.n_epochs}] " \
                              f"[Batch {i}/{len(self.dataloader)}] " \
                              f"[D loss: {loss_D.item()}] " \
                              f"[G loss: {loss_G.item()}, content: {loss_content.item()}, adv: {loss_GAN.item()}, " f"pixel: {loss_pixel.item()}] " \
                              f"[OCR loss: {ocr_loss.data.item()}]" \
                              f"[max_target: {max_target}] " \
                              f"[max_preds: {max_preds}] " \
                              f"[Average CER total: {self.CER_total.average()}] " \
                              f"[Average ave_total_loss: {self.ave_total_loss.average()}] "
        print(self.summary_string)

        self.__esrgan_log_progress2(i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                                    loss_pixel)
        return npa, n_iter

    def __esrgan_log_progress(self, i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                              loss_pixel):
        self.summary_string = f"[Epoch {epoch}/{self.opt.n_epochs}] " \
                              f"[Batch {i}/{len(self.dataloader)}] " \
                              f"[D loss: {loss_D.item()}] " \
                              f"[G loss: {loss_G.item()}, content: {loss_content.item()}, adv: {loss_GAN.item()}, " \
                              f"pixel: {loss_pixel.item()}]"

        print(self.summary_string)
        if batches_done % self.opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)
        if batches_done % self.opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(self.generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(self.discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)

    def __esrgan_log_progress2(self, i, batches_done, epoch, gen_hr, imgs_lr, loss_D, loss_G, loss_GAN, loss_content,
                               loss_pixel):
        if batches_done % self.opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)
        if batches_done % self.opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(self.generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(self.discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)

    def __ocr_log_progress(self, batches_done, epoch, images, ocr_loss, max_elem, max_preds, max_target, n_iter, npa,
                           wer_list):
        self.ave_total_loss.update(ocr_loss.data.item())
        self.writer.add_scalar("total_loss", self.ave_total_loss.average(), n_iter)
        if np.average(wer_list) > 0.1:
            # Save Loss in averagemeter and write to tensorboard
            self.writer.add_text("label", max_target, n_iter)
            self.writer.add_text("pred", max_preds, n_iter)
            self.writer.add_image("img", images[max_elem].detach().cpu().numpy(), n_iter)

            # gen.close()
            # break
        # Might become infinite
        if np.average(wer_list) < 10:
            self.CER_total.update(np.average(wer_list))
            self.writer.add_scalar("CER", self.CER_total.average(), n_iter)
        # We save when the new avereage CR is beloew the NPA
        # npa>CER_total.average() and CER_total.average()>0 and CER_total.average()<1
        if npa > self.CER_total.average() > 0 and self.CER_total.average() < 1:
            torch.save(self.ocr_model.state_dict(), "ocr_autosave.pt")
            npa = self.CER_total.average()
        n_iter = n_iter + 1
        self.ocr_cosine_learning_rate_scheduler.step()
        lr = self.ocr_optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("lr", lr, n_iter)
        # Save result checkpoints
        if batches_done % self.opt.sample_interval == 0:
            save_image(images[max_elem], "ocr_images/training/%d.png" % batches_done, nrow=1, normalize=False)
        # Save model checkpoints
        if batches_done % self.opt.checkpoint_interval == 0:
            torch.save(self.ocr_model.state_dict(), "ocr_saved/models/ocr_model_%d.pth" % epoch)

        self.summary_string = \
            f"||epoch: {epoch}|> " \
            f"||n_iter: {n_iter}|> " \
            f"||Loss: {ocr_loss.data.item()}|> " \
            f"||max_target: {max_target}|> " \
            f"||max_preds: {max_preds}|> " \
            f"||Average CER total: {self.CER_total.average()}|> " \
            f"||Average ave_total_loss: {self.ave_total_loss.average()}|> "
        print(self.summary_string)
        return npa, n_iter

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

    ## ocr
    def ocr_one_batch(self, batches_done, epoch, images, images_len, max_elem, max_preds, max_target, n_iter, npa,
                      plate_encoded, plate_encoded_len):
        # one_batch_train
        input_lengths, ocr_loss, log_probs = self.ocr_one_batch_train(images, images_len, plate_encoded,
                                                                      plate_encoded_len)
        # Here we Calculate the Character error rate
        targets, wer_list = self.calculate_character_error_rate(input_lengths, log_probs, plate_encoded)
        # Here we save an example together with its decoding and truth
        # Only if it is positive
        if np.average(wer_list) > 0.1:
            max_elem, max_preds, max_target = self.one_batch_eval(log_probs, targets, wer_list)
        # npa, n_iter = self.__ocr_log_progress(batches_done, epoch, images, ocr_loss, max_elem, max_preds, max_target,
        #                                  n_iter, npa, wer_list)
        return n_iter, input_lengths, ocr_loss, log_probs, targets, wer_list, max_elem, max_preds, max_target

    def ocr_one_batch_train(self, images, images_len, plate_encoded, plate_encoded_len):
        # DONT FORGET THE ZERO GRAD!!!!
        self.ocr_optimizer.zero_grad()
        # Get Predictions, permuted for CTC loss
        # log_probs = self.ocr_model(images.to(self.device)).permute((2, 0, 1))
        log_probs = self.ocr_model(images).permute((2, 0, 1))
        # Targets have to be CPU for baidu loss
        targets = plate_encoded.to(self.device)  # .cpu()
        # Get the Lengths/2 becase this is how much we downsample the width
        input_lengths = images_len / 2
        target_lengths = plate_encoded_len
        # Get the CTC Loss
        input_len, batch_size, vocab_size = log_probs.size()
        log_probs_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        loss = self.ctc_loss(log_probs, targets, log_probs_lens, target_lengths)
        # Then backward and step
        loss.backward()
        self.ocr_optimizer.step()
        # input_lengths, log_probs, loss = lengths, probs, loss1
        return input_lengths, loss, log_probs

    def ocr_one_batch_eval(self, log_probs, targets, wer_list):
        # max_value = np.max(wer_list)
        max_elem = np.argmax(wer_list)
        # max_image = images[max_elem].cpu()
        max_target = targets[max_elem]
        max_target = [self.dataset.decode_dict[x] for x in max_target.tolist()]
        max_target = "".join(max_target)
        ou = preds_to_integer(log_probs[:, max_elem, :])
        max_preds = [self.dataset.decode_dict[x] for x in ou]
        max_preds = "".join(max_preds)
        return max_elem, max_preds, max_target

    def ocr_calculate_character_error_rate(self, input_lengths, log_probs, plate_encoded):
        # cum_len = torch.cumsum(target_lengths, axis=0)
        # targets = np.split(plate_encoded.cpu(), cum_len[:-1])
        targets = plate_encoded.cpu()
        wer_list = []
        for j in range(log_probs.shape[1]):
            temp = log_probs[:, j, :][0:input_lengths[j], :]
            wer = wer_eval(temp, targets[j])
            wer_list.append(wer)
        return targets, wer_list


if __name__ == "__main__":
    data_path = Path(r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset")
    combined_model = esrgan_crnn()
    combined_model()
