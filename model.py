from torch import nn


class MobileNet(nn.Module):
    def __init__(self, class_num=1000, alpha=1):
        if alpha not in [0.25, 0.5, 0.75, 1.0]:
            raise ValueError("参数alpha取值需要在[0.24,0.5.0.75,1.0]中!")

        super(MobileNet, self).__init__()
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
        self.classifier = nn.Linear(int(1024 * alpha), class_num)

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
