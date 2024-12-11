import torch.nn as nn


class StarBlock(nn.Module):
    expansion = 4  # 与ResNet的Bottleneck保持一致

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample

        # 第一个1x1卷积，降维
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 深度可分离卷积替代3x3卷积
        self.dwconv = nn.Conv2d(
            planes,
            planes,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Star Operation部分
        self.f1 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.f2 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)

        # 1x1卷积，升维
        self.conv3 = nn.Conv2d(
            planes * 2, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x

        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 深度可分离卷积
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Star Operation
        x1 = self.f1(out)
        x2 = self.f2(out)
        out = self.relu(x1) * x2

        # 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StarNet(nn.Module):
    def __init__(self, block=StarBlock, layers=[3, 4, 6, 3]):
        super().__init__()
        self.inplanes = 64

        # 保持与ResNet相同的stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个stage，与ResNet保持一致
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def StarNet50(pretrained=False):
    """
    构建StarNet-50模型，保持与ResNet50相同的结构
    """
    model = StarNet(StarBlock, [3, 4, 6, 3])
    return model


# 使用方式与原ResNet完全一致
def Backbone_StarNet50(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = StarNet50(pretrained=pretrained)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32
