# standart modules
import argparse
import os

import numpy as np
# torch modules
import torch
from PIL import Image
from albumentations.pytorch import ToTensor
from path import Path
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, ToPILImage, Resize
from torchvision.utils import save_image

# import matplotlib.pyplot as plt

## my modules
try:
    from models import fully_conv_model
    from . import dataset, UFPR_ALPR_dataset
    from .crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
    from utils import count_parameters
    from .train_loop import train_loop
except:
    from models import fully_conv_model
    from dataset import UFPR_ALPR_dataset
    from train_scripts.crnn_evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
    from utils import count_parameters
    from train_scripts.train_loop import train_loop

# torch.manual_seed(0)
# plt.style.use('seaborn')


# def train(epochs=5, batch_size=4, npa=1, lr=5e-4, eta_min=1e-6):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ## Set up Tensorboard writer for current test
#     writer = SummaryWriter(log_dir="../logs")  # /home/leander/AI/repos/OCR-CNN/logs2/correct_cosine_2
#     ## Model
#     ocr_model = fully_conv_model.cnn_attention_ocr(n_layers=8, nclasses=93, model_dim=64, input_dim=3)
#     ocr_model = ocr_model.to(device).train()
#     ctc_loss = nn.CTCLoss(blank=0, reduction="mean")
#     ## Optimizer: Good initial is 5e5
#     ocr_optimizer = optim.Adam(ocr_model.parameters(), lr=lr)
#     ocr_cosine_learning_rate_scheduler = CosineAnnealingLR(optimizer=ocr_optimizer, T_max=250000, eta_min=eta_min)
#     ## We keep track of the Average loss and CER
#     ave_total_loss = AverageMeter()
#     CER_total = AverageMeter()
#     ## Dataset
#     transforms = Compose([
#         ToPILImage(),
#         Resize((29, 73), Image.BICUBIC),
#         ToTensor()
#     ])
#     root = Path(r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset")
#     ds = UFPR_ALPR_dataset(root, dataset_type="__train", transform=transforms)
#     trainset = DataLoader(ds, batch_size=batch_size)
#     # testset = DataLoader(test_dataset, batch_size=batch_size)
#
#     print(count_parameters(ocr_model))
#
#     ## Train
#     n_iter = 0
#     for epochs in range(epochs):
#
#         print("Epoch:", epochs, "started")
#         for i, ge in enumerate(trainset):
#
#             # to avoid OOM
#             if ge[0].shape[3] <= 800:
#
#                 # DONT FORGET THE ZERO GRAD!!!!
#                 ocr_optimizer.zero_grad()
#
#                 # Get Predictions, permuted for CTC loss
#                 log_probs = ocr_model(ge[0].to(device)).permute((2, 0, 1))
#
#                 # Targets have to be CPU for baidu loss
#                 targets = ge[1].to(device)  # .cpu()
#
#                 # Get the Lengths/2 becase this is how much we downsample the width
#                 input_lengths = ge[2] / 2
#                 target_lengths = ge[3]
#
#                 # Get the CTC Loss
#                 input_len, batch_size, vocab_size = log_probs.size()
#                 log_probs_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
#                 loss = ctc_loss(log_probs, targets, log_probs_lens, target_lengths)
#
#                 # Then backward and step
#                 loss.backward()
#                 ocr_optimizer.step()
#
#                 # Save Loss in averagemeter and write to tensorboard
#                 ave_total_loss.update(loss.data.item())
#                 writer.add_scalar("total_loss", ave_total_loss.average(), n_iter)
#
#                 # Here we Calculate the Character error rate
#                 # cum_len = torch.cumsum(target_lengths, axis=0)
#                 # targets = np.split(ge[1].cpu(), cum_len[:-1])
#                 targets = ge[1].cpu()
#                 wer_list = []
#                 for j in range(log_probs.shape[1]):
#                     temp = log_probs[:, j, :][0:input_lengths[j], :]
#                     wer = wer_eval(temp, targets[j])
#                     wer_list.append(wer)
#
#                 # Here we save an example together with its decoding and truth
#                 # Only if it is positive
#
#                 if np.average(wer_list) > 0.1:
#                     # max_value = np.max(wer_list)
#                     max_elem = np.argmax(wer_list)
#                     # max_image = ge[0][max_elem].cpu()
#                     max_target = targets[max_elem]
#
#                     max_target = [ds.decode_dict[x] for x in max_target.tolist()]
#                     max_target = "".join(max_target)
#
#                     ou = preds_to_integer(log_probs[:, max_elem, :])
#                     max_preds = [ds.decode_dict[x] for x in ou]
#                     max_preds = "".join(max_preds)
#
#                     writer.add_text("label", max_target, n_iter)
#                     writer.add_text("pred", max_preds, n_iter)
#                     writer.add_image("img", ge[0][max_elem].detach().cpu().numpy(), n_iter)
#
#                     # gen.close()
#                     # break
#
#                 # Might become infinite
#                 if np.average(wer_list) < 10:
#                     CER_total.update(np.average(wer_list))
#                     writer.add_scalar("CER", CER_total.average(), n_iter)
#
#                 # We save when the new average CR is below the NPA
#                 # npa>CER_total.average() and CER_total.average()>0 and CER_total.average()<1
#                 if npa > CER_total.average() > 0 and CER_total.average() < 1:
#                     torch.save(ocr_model.state_dict(), "autosave.pt")
#                     npa = CER_total.average()
#
#                 n_iter = n_iter + 1
#                 ocr_cosine_learning_rate_scheduler.step()
#                 lr = ocr_optimizer.param_groups[0]["lr"]
#                 writer.add_scalar("lr", lr, n_iter)
#
#         summary_string = f"||epochs: {epochs}|> " \
#                          f"||n_iter: {n_iter}|> " \
#                          f"||Loss: {loss.data.item()}|> " \
#                          f"||max_target: {max_target}|> " \
#                          f"||max_preds: {max_preds}|> " \
#                          f"||Average CER total: {CER_total.average()}|> " \
#                          f"||Average ave_total_loss: {ave_total_loss.average()}|> "
#         print(summary_string)
#
#     print(CER_total.average())


# Helper to count params


class crnn(train_loop):
    def __init__(self, epoch=0, n_epochs=200, dataset_name=r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset",
                 batch_size=4, npa=1, lr=5e-4, eta_min=1e-6, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=4, hr_height=256,
                 hr_width=256, channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23,
                 warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        # super().__init__(epoch=epoch, n_epochs=n_epochs, dataset_name=dataset_name, batch_size=batch_size, lr=lr, b1=b1,
        #                  b2=b2, decay_epoch=decay_epoch, n_cpu=n_cpu, hr_height=hr_height, hr_width=hr_width,
        #                  channels=channels, sample_interval=sample_interval, checkpoint_interval=checkpoint_interval,
        #                  residual_blocks=residual_blocks, warmup_batches=warmup_batches, lambda_adv=lambda_adv,
        #                  lambda_pixel=lambda_pixel)
        os.makedirs("ocr_images/training", exist_ok=True)
        os.makedirs("ocr_saved/models", exist_ok=True)

        self.reload(epoch=epoch, n_epochs=n_epochs, dataset_name=dataset_name, batch_size=batch_size, npa=npa, lr=lr,
                    eta_min=eta_min, b1=b1, b2=b2, decay_epoch=decay_epoch, n_cpu=n_cpu, hr_height=hr_height,
                    hr_width=hr_width, channels=channels, sample_interval=sample_interval,
                    checkpoint_interval=checkpoint_interval, residual_blocks=residual_blocks,
                    warmup_batches=warmup_batches, lambda_adv=lambda_adv, lambda_pixel=lambda_pixel)

    def __call__(self, *args, **kwargs):
        self.crnn_train()

    def reload(self, epoch=0, n_epochs=200, dataset_name=r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset",
               batch_size=4, npa=1, lr=5e-4, eta_min=1e-6, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=4, hr_height=256,
               hr_width=256, channels=3, sample_interval=100, checkpoint_interval=100, residual_blocks=23,
               warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
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
        self.ocr_model = self.network_initializers()
        ## Loss
        self.ctc_loss = self.losses()
        ## Optimizer: Good initial is 5e5
        self.ocr_optimizer, self.ocr_cosine_learning_rate_scheduler = self.optimizers(eta_min)
        ## We keep track of the Average loss and CER
        self.ave_total_loss = AverageMeter()
        self.CER_total = AverageMeter()
        ## Dataset
        # self.transforms = albumentations.Compose([
        #     albumentations.Resize(29, 73, interpolation=cv2.INTER_CUBIC),
        #     ToTensor(),
        # ])
        self.transforms = Compose([
            ToPILImage(),
            Resize((29, 73), Image.BICUBIC),
            ToTensor()
        ])
        root = Path(self.opt.dataset_name)
        self.dataset = UFPR_ALPR_dataset(root, dataset_type="__train", transform=self.transforms)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
            pin_memory=True,
        )
        # testset = DataLoader(test_dataset, batch_size=batch_size)
        print("Number of parameters of OCR model", count_parameters(self.ocr_model))

    def network_initializers(self):
        ocr_model = fully_conv_model.cnn_attention_ocr(n_layers=8, nclasses=93, model_dim=64, input_dim=3)
        ocr_model = ocr_model.to(self.device, non_blocking=True).train()
        return ocr_model

    def losses(self):
        ctc_loss = nn.CTCLoss(blank=0, reduction="mean").to(self.device)
        return ctc_loss

    def optimizers(self):
        ocr_optimizer = optim.Adam(self.ocr_model.parameters(), lr=self.opt.lr)
        ocr_cosine_learning_rate_scheduler = CosineAnnealingLR(optimizer=ocr_optimizer, T_max=250000, eta_min=self.opt.eta_min)
        return ocr_optimizer, ocr_cosine_learning_rate_scheduler

    def __commandline_interface(self, epoch=0, n_epochs=200, dataset_name="dataset", batch_size=4, npa=1,
                                lr=0.0002, eta_min=1e-6, b1=0.9, b2=0.999, decay_epoch=100, n_cpu=8, hr_height=256,
                                hr_width=256, channels=3, sample_interval=100, checkpoint_interval=100,
                                residual_blocks=23, warmup_batches=500, lambda_adv=5e-3, lambda_pixel=1e-2):
        self.parser.add_argument("--epoch", type=int, default=epoch, help="epoch to start training from")
        self.parser.add_argument("--n_epochs", type=int, default=n_epochs, help="number of epochs of training")
        self.parser.add_argument("--dataset_name", type=str, default=dataset_name,
                            help="name of the dataset")  # img_align_celeba
        self.parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")  # 4
        self.parser.add_argument("--npa", type=int, default=npa, help="maximum avereage CR(character error rate)")  # !!! explain !!!
        self.parser.add_argument("--lr", type=float, default=lr, help="adam: learning rate")
        self.parser.add_argument("--eta_min", type=float, default=eta_min, help="cosine lr scheduler rate.")
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

    def crnn_train(self):
        if self.opt.epoch != 0:
            self.ocr_model.load_state_dict(torch.load("ocr_saved/models/ocr_model_%d.pth" % self.opt.epoch))
        self.__train()

    def __train(self):
        ## Train
        n_iter = 0
        npa = self.opt.npa
        max_elem, max_preds, max_target = 0, 0, 0
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            print("Epoch:", epoch, "started")
            for i, ge in enumerate(self.dataloader):

                batches_done = epoch * len(self.dataloader) + i

                # Configure model input
                images, plate_encoded, images_len, plate_encoded_len = ge
                images = images.to(self.device, non_blocking=True)
                plate_encoded = plate_encoded.to(self.device, non_blocking=True)

                # to avoid OOM
                if images.shape[3] <= 800:

                    n_iter, input_lengths, loss, log_probs, max_elem, max_preds, max_target = \
                        self.ocr_one_batch(batches_done, epoch, images, images_len, max_elem, max_preds, max_target,
                                           n_iter, npa, plate_encoded, plate_encoded_len)
        print(self.CER_total.average())

    def ocr_one_batch(self, batches_done, epoch, images, images_len, max_elem, max_preds, max_target, n_iter, npa,
                      plate_encoded, plate_encoded_len):
        # one_batch_train
        input_lengths, loss, log_probs = self.one_batch_train(images, images_len, plate_encoded, plate_encoded_len)
        # Here we Calculate the Character error rate
        targets, wer_list = self.calculate_character_error_rate(input_lengths, log_probs, plate_encoded)
        # Here we save an example together with its decoding and truth
        # Only if it is positive
        if np.average(wer_list) > 0.1:
            max_elem, max_preds, max_target = self.one_batch_eval(log_probs, targets, wer_list)
        npa, n_iter = self.__log_progress(batches_done, epoch, images, loss, max_elem, max_preds, max_target,
                                     n_iter, npa, wer_list)
        return n_iter, input_lengths, loss, log_probs, max_elem, max_preds, max_target

    def __log_progress(self, batches_done, epoch, images, loss, max_elem, max_preds, max_target, n_iter, npa, wer_list):
        self.ave_total_loss.update(loss.data.item())
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
            f"||Loss: {loss.data.item()}|> " \
            f"||max_target: {max_target}|> " \
            f"||max_preds: {max_preds}|> " \
            f"||Average CER total: {self.CER_total.average()}|> " \
            f"||Average ave_total_loss: {self.ave_total_loss.average()}|> "
        print(self.summary_string)

        return npa, n_iter

    def one_batch_train(self, images, images_len, plate_encoded, plate_encoded_len):
        # DONT FORGET THE ZERO GRAD!!!!
        self.ocr_optimizer.zero_grad()
        # Get Predictions, permuted for CTC loss
        # log_probs = self.ocr_model(images.to(self.device)).permute((2, 0, 1))
        log_probs = self.ocr_model(images).permute((2, 0, 1))
        # Targets have to be CPU for baidu loss
        targets = plate_encoded.to(self.device)  # .cpu()
        # Get the Lengths/2 becase this is how much we downsample the width
        input_lengths = images_len / 2
        # Get the CTC Loss
        input_len, batch_size, vocab_size = log_probs.size()
        log_probs_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        loss = self.ctc_loss(log_probs, targets, log_probs_lens, plate_encoded_len)
        # Then backward and step
        loss.backward()
        self.ocr_optimizer.step()
        # input_lengths, log_probs, loss = lengths, probs, loss1
        return input_lengths, loss, log_probs

    @torch.no_grad()
    def one_batch_eval(self, log_probs, targets, wer_list):
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

    def calculate_character_error_rate(self, input_lengths, log_probs, plate_encoded):
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
    # epochs = 100000
    # batch_size = 50
    # train(epochs=epochs, batch_size=batch_size, npa=1, lr=5e-4, eta_min=1e-6)
    recog = crnn()
    recog()

