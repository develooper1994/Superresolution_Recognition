import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import albumentations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, RandomCrop, Flip, OneOf, Compose, Resize, Normalize
)
from albumentations.pytorch import ToTensor

import kornia

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset_superresolution(Dataset):
    def __init__(self, device, root=None, hr_shape=None, interpolation='bicubic', load_all=True):
        hr_height, hr_width = hr_shape
        self.hr_height = hr_height
        self.hr_width = hr_width
        self.device = device
        self.load_all = load_all

        # ## kornia
        # Transforms for low resolution images and high resolution images
        # self.lr_transform = kornia.nn.Sequential(
        #     kornia.geometry.Resize(size=(hr_height // 4, hr_height // 4), interpolation=interpolation),
        #     kornia.color.Normalize(torch.from_numpy(mean), torch.from_numpy(std)),
        # ).to(device=device)
        # self.hr_transform = kornia.nn.Sequential(
        #     kornia.geometry.Resize(size=(hr_height, hr_height), interpolation=interpolation),
        #     kornia.color.Normalize(torch.from_numpy(mean), torch.from_numpy(std)),
        # )

        # ## torchvision
        # self.lr_transform = transforms.Compose(
        #     [
        # #       transforms.ToPILImage(),
        #         transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean, std),
        #     ]
        # )
        # self.hr_transform = transforms.Compose(
        #     [
        # #       transforms.ToPILImage(),
        #         transforms.Resize((hr_height, hr_height), Image.BICUBIC),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean, std),
        #     ]
        # )

        ## Albumentation
        self.lr_transform = albumentations.Compose([
            Resize(hr_height // 4, hr_height // 4),
            Normalize(mean, std),
            ToTensor(),
        ])

        self.hr_transform = albumentations.Compose([
            Resize(hr_height, hr_height),
            Normalize(mean, std),
            ToTensor(),
        ])

        if isinstance(root, str):
            self.files = sorted(glob.glob(root + "/*.*"))
        else:
            self.files = root

        if load_all:
            self.images_lr, self.images_hr = self.__load_all_images()

    def __load_all_images(self):
        files_len = len(self.files)

        index = 0
        img_pivot = cv2.imread(self.files[index % files_len])
        img_lr = self.lr_transform(image=img_pivot)['image']
        img_hr = self.hr_transform(image=img_pivot)['image']
        images_lr = torch.zeros((files_len, img_pivot.shape[-1], self.hr_height//4, self.hr_width//4))
        images_hr = torch.zeros((files_len, img_pivot.shape[-1], self.hr_height, self.hr_width))

        images_lr[index, ...] = img_lr
        images_hr[index, ...] = img_hr
        for index in range(1, files_len):
            img = cv2.imread(self.files[index % files_len])
            img_lr = self.lr_transform(image=img)['image']
            img_hr = self.hr_transform(image=img)['image']
            images_lr[index, ...] = img_lr
            images_hr[index, ...] = img_hr
        return images_lr, images_hr

    def load_one_image(self, index):
        if isinstance(self.files[index], str):
            img = cv2.imread(self.files[index % len(self.files)])
        else:
            img = self.files
        img_lr = self.lr_transform(image=img)['image']
        img_hr = self.hr_transform(image=img)['image']
        return {"lr": img_lr, "hr": img_hr}

    def __getitem__(self, index):
        if self.load_all:
            return {"lr": self.images_lr[index], "hr": self.images_hr[index]}
        else:
            return self.load_one_image(index)

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    dataset_name = "../TurkishPlates"
    hr_shape = (256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset_superresolution(device=device, root=dataset_name, hr_shape=hr_shape)
    print(dataset[0])
