import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.up(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class BothNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BothNet, self).__init__()

        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 64)

        self.top_up_1 = DoubleConv(64, 32)
        self.top_up_2 = DoubleConv(32, 16)
        self.top_up_3 = DoubleConv(16, 8)
        self.top_up_4 = DoubleConv(8, 8)

        self.top_down_1 = DoubleConv(16, 16)
        self.top_down_2 = DoubleConv(32, 32)
        self.top_down_3 = DoubleConv(64, 64)

        self.bot_up_1 = DoubleConv(1024, 256)
        self.bot_up_2 = DoubleConv(512, 128)
        self.bot_up_3 = DoubleConv(256, 64)

        self.bot_down_1 = DoubleConv(64, 128)
        self.bot_down_2 = DoubleConv(128, 256)
        self.bot_down_3 = DoubleConv(256, 512)
        self.bot_down_4 = DoubleConv(512, 512)

        self.out_mid = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        self.sigmoid_activation = nn.Sigmoid()

#         self.dropout = nn.Dropout(dropout_rate)

        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)

        xu1 = self.top_up_1(x1)
        xu2 = self.top_up_2(xu1)
        xu3 = self.top_up_3(xu2)
        xud1 = self.top_up_4(xu3)

        xud2 = self.top_down_1(torch.cat([xud1, xu3], dim=1))
        xud3 = self.top_down_2(torch.cat([xud2, xu2], dim=1))
        xud4 = self.top_down_3(torch.cat([xud3, xu1], dim=1))

        xd1 = self.bot_down_1(x1)
        xd2 = self.bot_down_2(xd1)
        xd3 = self.bot_down_3(xd2)
        xdu1 = self.bot_down_4(xd3)

        xdu2 = self.bot_up_1(torch.cat([xdu1, xd3], dim=1))
        xdu3 = self.bot_up_2(torch.cat([xdu2, xd2], dim=1))
        xdu4 = self.bot_up_3(torch.cat([xdu3, xd1], dim=1))

        middle_out = self.out_mid(torch.cat([xdu4, xud4], dim=1))

        logits = self.outc(middle_out)
    

        return logits



class UNet_1(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5):
        super(UNet_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = DoubleConv(1024, 256)
        self.up2 = DoubleConv(512, 128)
        self.up3 = DoubleConv(256, 64)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid_activation = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)
        # logits = self.sigmoid_activation(x)

        return logits
    


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

        # Original encoder: Use ResNet as the encoder with pretrained weights
        original_encoder = models.resnet18(pretrained=True)

        # Modify the first convolution layer to accept 3-channel input
        self.first_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy weights from the original first layer (for the first 3 channels)
        self.first_conv.weight.data[:, :3] = original_encoder.conv1.weight.data

        # Use the rest of the layers from the original encoder
        self.encoder_layers = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *list(original_encoder.children())[4:-2]  # Exclude the original first conv layer and the fully connected layers
        )
        
        self.encoder = nn.Sequential(
            self.first_conv,
            self.encoder_layers
        )

        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder: A simple decoder with transpose convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)  # Output depth map
        )

    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)

        # Forward pass through the decoder
        x = self.decoder(x)
        
        return x
