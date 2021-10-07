from torch import nn


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, alpha=1):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha

        self.model = nn.Sequential(
            self.conv_bn(3, int(32 * alpha), 2),
            self.deep_wise_conv_bn(int(32 * alpha), int(64 * alpha), 1),
            self.deep_wise_conv_bn(int(64 * alpha), int(128 * alpha), 2),
            self.deep_wise_conv_bn(int(128 * alpha), int(128 * alpha), 1),
            self.deep_wise_conv_bn(int(128 * alpha), int(256 * alpha), 2),
            self.deep_wise_conv_bn(int(256 * alpha), int(256 * alpha), 1),
            self.deep_wise_conv_bn(int(256 * alpha), int(512 * alpha), 2),
            self.deep_wise_conv_bn(int(512 * alpha), int(512 * alpha), 1),
            self.deep_wise_conv_bn(int(512 * alpha), int(512 * alpha), 1),
            self.deep_wise_conv_bn(int(512 * alpha), int(512 * alpha), 1),
            self.deep_wise_conv_bn(int(512 * alpha), int(512 * alpha), 1),
            self.deep_wise_conv_bn(int(512 * alpha), int(512 * alpha), 1),
            self.deep_wise_conv_bn(int(512 * alpha), int(1024 * alpha), 2),
            self.deep_wise_conv_bn(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(int(1024 * alpha), num_classes)
        self.parameter_init()

    def parameter_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024 * self.alpha))
        x = self.classifier(x)
        return x

    @staticmethod
    def deep_wise_conv_bn(in_channels, out_channels, stride):
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def conv_bn(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):

        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(input_channel * width_mult)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult))

        features = [ConvBNActivation(3, input_channel, stride=2)]
        # 翻转残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNActivation(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self.parameter_init()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        n = x.shape[0]
        x = x.view(n, -1)
        x = self.classifier(x)
        return x

    def parameter_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def _make_divisible(v, divisor=8):
    """
    确保所有层的通道数能够被8整除
    原始实现如下:https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    new_v = max(divisor, int(round(v / divisor) * divisor))
    # 确保四舍五入后不会低于原来的90%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        self.out_channels = out_planes
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # 逐点卷积,升维
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # 深度卷积
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 线性变换
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
