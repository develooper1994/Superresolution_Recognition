import torch
import torch.nn as nn
from timm.models.layers import Mish
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2, use_LeakyReLU_Mish=True):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.MISH = Mish()

        def block(in_features, non_linearity=True, use_LeakyReLU_Mish=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                if use_LeakyReLU_Mish:
                    layers += [nn.LeakyReLU()]
                else:
                    layers += [Mish()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters, use_LeakyReLU_Mish=use_LeakyReLU_Mish)
        self.b2 = block(in_features=2 * filters, use_LeakyReLU_Mish=use_LeakyReLU_Mish)
        self.b3 = block(in_features=3 * filters, use_LeakyReLU_Mish=use_LeakyReLU_Mish)
        self.b4 = block(in_features=4 * filters, use_LeakyReLU_Mish=use_LeakyReLU_Mish)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2, use_LeakyReLU_Mish=True):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters, use_LeakyReLU_Mish),
            DenseResidualBlock(filters, use_LeakyReLU_Mish),
            DenseResidualBlock(filters, use_LeakyReLU_Mish)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2, use_LeakyReLU_Mish=True):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters, use_LeakyReLU_Mish) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape, use_LeakyReLU_Mish=True):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        self.MISH = Mish()

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)]
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            if use_LeakyReLU_Mish:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(self.MISH)
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            if use_LeakyReLU_Mish:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(self.MISH)
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
