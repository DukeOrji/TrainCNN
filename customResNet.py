import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        #if shpaes dont match, transform identity
        if stride !=1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = x #save previous state

        x = torch.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.bn2(
            self.conv2(x)
        )

        #make identity match x
        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity #enables residual learning
        x = torch.relu(x)

        return x
    
class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=2),
            ResidualBlock(64, 64)
        )

    

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )

        

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(
            self.conv(x)
        )

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        

        x = self.pool(x)

        x = x.view(x.size(0), -1) #flattened to be compatible with fc layers 
        x = self.fc(x)

        return x

    
    

