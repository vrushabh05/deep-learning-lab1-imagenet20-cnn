"""ImageNet-20 CNN trained from scratch (no pretrained weights)."""
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_convs - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class ImageNet20Net(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, num_convs=1),      # 240 -> 120
            ConvBlock(32, 64, num_convs=1),     # 120 -> 60
            ConvBlock(64, 128, num_convs=2),    # 60 -> 30
            ConvBlock(128, 256, num_convs=2),   # 30 -> 15
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


if __name__ == "__main__":
    import torch
    net = ImageNet20Net()
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")
    print("Output shape:", tuple(net(torch.randn(1, 3, 240, 240)).shape))
