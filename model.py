import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

from cbam import CBAM
import torchvision

pretrained_model1 = list(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2]
pretrained_model2 = list(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2]

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResNet(nn.Module):
    def __init__(self, pretrained_model=None):
        super().__init__()

        self.conv1 = nn.Sequential(*pretrained_model[:4])
        # C2
        self.layers1 = pretrained_model[4]
        # C3
        self.layers2 = pretrained_model[5]
        # C4
        self.layers3 = pretrained_model[6]
        # C5
        self.layers4 = pretrained_model[7]

    def forward(self, x):
        x = self.conv1(x)

        c2 = self.layers1(x)
        c3 = self.layers2(c2)
        c4 = self.layers3(c3)
        c5 = self.layers4(c4)

        return c2, c3, c4, c5

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, is_highest=False, is_lowest=False):
        super().__init__()

        # 1x1 convolution to unify the number of feature map channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # 3x3 convolution for feature output
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.is_highest = is_highest
        self.is_lowest = is_lowest

    def forward(self, x, y):
        x = self.conv1(x)
        if not self.is_highest:
            target_height = x.shape[2]
            target_width = x.shape[3]
            x += F.interpolate(y, size=(target_height, target_width), mode="bilinear", align_corners=True)
        if self.is_lowest:
            x = self.conv2(x)
        return x

class FPN(nn.Module):
    def __init__(self, expansion=1, in_channels_list=[64, 128, 256, 512], out_channels=256):
        """
        Args:
            expansion: expansion rate of ResBlock (1 for BasicBlock or 4 for BottleNeck)
            in_channels_list: list of the output channel numbers for conv2_x - conv5_x
            out_channels: target number of channels (256 by default)
        """
        super().__init__()

        # Create layers to generate P2-P5
        self.P2 = FPNBlock(in_channels_list[0]*expansion, out_channels=out_channels, is_lowest=True)
        self.P3 = FPNBlock(in_channels_list[1]*expansion, out_channels=out_channels)
        self.P4 = FPNBlock(in_channels_list[2]*expansion, out_channels=out_channels)
        self.P5 = FPNBlock(in_channels_list[3]*expansion, out_channels=out_channels, is_highest=True)

    def forward(self, C2, C3, C4, C5):
        """
        Args:
            C2-C5: feature maps output by ResNet
        Returns:
            P2-P5: enchanced features by FPN
        """

        x = self.P5(C5, None)
        x = self.P4(C4, x)
        x = self.P3(C3, x)
        P2 = self.P2(C2, x)

        return P2


class ResNetFPN(nn.Module):
    def __init__(self):
        """
        Args:
            resnet_arch: the type of ResNet architecture
        """
        super().__init__()

        # Create ResNet
        self.resnet1 = ResNet(pretrained_model=pretrained_model1)
        self.resnet2 = ResNet(pretrained_model=pretrained_model2)

        # Create FPN
        self.FPN1 = FPN(expansion=4)
        self.FPN2 = FPN(expansion=4)

    def forward(self, x, y):
        """
        Args:
            x: input tensor
        Returns:
            P2-P6: enhanced features
        """
        C2, C3, C4, C5 = self.resnet1(x)
        c2, c3, c4, c5 = self.resnet2(y)

        P2 = self.FPN1(C2, C3, C4, C5)
        p2 = self.FPN2(c2, c3, c4, c5)

        return P2, p2

class SAM_FNet(nn.Module):
    def __init__(self, num_classes=3, num_features=2):
        super().__init__()
        self.resnet_fpn = ResNetFPN()

        self.cbam5_x = CBAM(256)
        self.cbam5_x1 = CBAM(256)

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.drop_fusion = nn.Dropout(p=0.5, inplace=True)

        self.fc1_dense = nn.Linear(256, num_classes)
        self.fc2_dense = nn.Linear(256, num_classes)
        self.fc_dense = nn.Linear(256 * 2, num_classes)

        self.modal_dense = nn.Linear(256, num_features)
        

    def forward(self, img1, img2, target = None):
        feats = {}

        # ResNet and FPN -- 1
        x, y = self.resnet_fpn(img1, img2)

        # attention block -- 2
        output1 = self.cbam5_x(x) # (1, 256, 56, 56)
        output2 = self.cbam5_x1(y)

        # global
        c1 = self.avg_pool1(output1)
        c1 = c1.view(c1.size(0), -1)
        # local
        c2 = self.avg_pool2(output2)
        c2 = c2.view(c2.size(0), -1)

        feats['global'] = F.normalize(c1, dim=-1, p=2)
        feats['local'] = F.normalize(c2, dim=-1, p=2)

        c1_cls = self.fc1_dense(c1)
        c2_cls = self.fc2_dense(c2)
        c1_mdl = self.modal_dense(c1)
        c2_mdl = self.modal_dense(c2)

        # fusion
        output = torch.cat((c1, c2), dim = 1)
        output = self.drop_fusion(output)
        output = self.fc_dense(output)

        return c1_cls, c2_cls, output, c1_mdl, c2_mdl, feats

def SAM_FNet50(num_classes=3, num_features=2):
    return SAM_FNet(num_classes=num_classes, num_features=num_features)


if __name__ == "__main__":
    net = SAM_FNet50(3, 2)
    x = torch.randn((1, 3, 256, 256), dtype=torch.float32)
    y = torch.randn((1, 3, 256, 256), dtype=torch.float32)
    ### Inference
    _, _, output1, _, _, _ = net.forward(x, y)
    ### Output
    print(output1)

