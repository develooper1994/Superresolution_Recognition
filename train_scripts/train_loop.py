import argparse
import os
from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import DataLoader

try:
    from dataset.superresolution_dataset.superresolution_dataset import ImageDataset_superresolution
    from models import GeneratorRRDB, Discriminator, FeatureExtractor
except:
    from train_scripts.esrgan_train import ImageDataset_superresolution
    from train_scripts.esrgan_train import GeneratorRRDB, Discriminator, FeatureExtractor


class train_loop(metaclass=ABCMeta):
    def __init__(self, epoch=0, n_epochs=200, dataset_name="../dataset/TurkishPlates", batch_size=3, lr=0.0002, b1=0.9,
                 b2=0.999, decay_epoch=100, n_cpu=8, hr_height=256, hr_width=256,
                 channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23, warmup_batches=500,
                 lambda_adv=5e-3, lambda_pixel=1e-2):
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

        self.reload(b1, b2, batch_size, channels, checkpoint_interval, dataset_name, decay_epoch, epoch, hr_height,
                    hr_width, lambda_adv, lambda_pixel, lr, n_cpu, n_epochs, residual_blocks, sample_interval,
                    warmup_batches)

    def reload(self, b1, b2, batch_size, channels, checkpoint_interval, dataset_name, decay_epoch, epoch, hr_height,
               hr_width, lambda_adv, lambda_pixel, lr, n_cpu, n_epochs, residual_blocks, sample_interval,
               warmup_batches):
        self.parser = argparse.ArgumentParser()
        self.opt = self.__commandline_interface(epoch=epoch, n_epochs=n_epochs, dataset_name=dataset_name,
                                                batch_size=batch_size, lr=lr, b1=b1, b2=b2, decay_epoch=decay_epoch,
                                                n_cpu=n_cpu, hr_height=hr_height, hr_width=hr_width, channels=channels,
                                                sample_interval=sample_interval,
                                                checkpoint_interval=checkpoint_interval,
                                                residual_blocks=residual_blocks, warmup_batches=warmup_batches,
                                                lambda_adv=lambda_adv, lambda_pixel=lambda_pixel)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize generator and discriminator
        hr_shape = (self.opt.hr_height, self.opt.hr_width)
        self.discriminator, self.feature_extractor, self.generator = self.network_initializers(hr_shape)
        # Losses
        self.criterion_GAN, self.criterion_content, self.criterion_pixel = self.losses()
        # Optimizers
        self.optimizer_D, self.optimizer_G = self.optimizers()
        # Data
        self.dataset = ImageDataset_superresolution(device=self.device, root=self.opt.dataset_name, hr_shape=hr_shape)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
            pin_memory=True,
        )

    @abstractmethod
    def __call__(self, args, kwargs):
        pass

    def __commandline_interface(self, epoch=0, n_epochs=200, dataset_name="dataset", batch_size=4, lr=0.0002, b1=0.9,
                                b2=0.999, decay_epoch=100, n_cpu=8, hr_height=256, hr_width=256,
                                channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23,
                                warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        self.parser.add_argument("--epoch", type=int, default=epoch, help="epoch to start training from")
        self.parser.add_argument("--n_epochs", type=int, default=n_epochs, help="number of epochs of training")
        self.parser.add_argument("--dataset_name", type=str, default=dataset_name,
                            help="name of the dataset")  # img_align_celeba
        self.parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")  # 4
        self.parser.add_argument("--lr", type=float, default=lr, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=b1, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=b2, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--decay_epoch", type=int, default=decay_epoch, help="epoch from which to start lr decay")
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

    def network_initializers(self, hr_shape):
        use_LeakyReLU_Mish = False
        generator = GeneratorRRDB(self.opt.channels, filters=64, num_res_blocks=self.opt.residual_blocks,
                                  use_LeakyReLU_Mish=use_LeakyReLU_Mish).to(self.device, non_blocking=True)
        discriminator = Discriminator(input_shape=(self.opt.channels, *hr_shape),
                                      use_LeakyReLU_Mish=use_LeakyReLU_Mish).to(self.device, non_blocking=True)
        feature_extractor = FeatureExtractor().to(self.device, non_blocking=True)
        # Set feature extractor to inference mode
        feature_extractor.eval()
        return discriminator, feature_extractor, generator

    def losses(self):
        criterion_GAN = torch.nn.BCEWithLogitsLoss()
        criterion_content = torch.nn.L1Loss()
        criterion_pixel = torch.nn.L1Loss()
        return criterion_GAN, criterion_content, criterion_pixel

    def optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr,
                                       betas=(self.opt.b1, self.opt.b2))
        return optimizer_D, optimizer_G

    def __train(self):
        pass
