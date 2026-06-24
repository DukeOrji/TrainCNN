import torch
import torch.nn as nn



class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels//2, out_channels//2, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)

        self.dwconv2 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1, groups=out_channels//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//2)

        self.conv3 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels//2)


    def channel_shuffle(self, x):

        #channel grouping and shuffling
        N, C, H, W = x.size()
        x = x.view(N, 2, C//2, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(N, C, H, W)
        return x

    def forward(self, x):
        
        x1, x2 = torch.chunk(x, 2, dim=1) # splits the channels into identical dimensions

        x2 = torch.relu(
            self.bn1(
                self.conv1(x2)
            )
        )

        x2 = torch.relu(
            self.bn2(
                self.dwconv2(x2)
            )
        )

        x2 = self.bn3(
            self.conv3(x2)
        )
        x = torch.cat([x1, x2], dim=1) #residual connection through concencation
        x = self.channel_shuffle(x)

        x = torch.relu(x)
        return x

class ShuffleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(24)
        self.mpool = nn.MaxPool2d(2)
        self.transition1 = nn.Conv2d(24, 240, kernel_size=1, bias=False)

        self.stage2 = nn.Sequential(
            ShuffleUnit(240, 240),  
            ShuffleUnit(240, 240),
            ShuffleUnit(240, 240),
            ShuffleUnit(240, 240)
        )

        self.transition2 = nn.Conv2d(240, 480, kernel_size=1, bias=False)
        self.stage3 = nn.Sequential(
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480),
            ShuffleUnit(480, 480)
        )

        self.transition3 = nn.Conv2d(480, 960, kernel_size=1, bias=False)
        self.stage4 = nn.Sequential(
            ShuffleUnit(960, 960),
            ShuffleUnit(960, 960),
            ShuffleUnit(960, 960),
            ShuffleUnit(960, 960)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, 10)
        
    
    
    def forward(self, x):
        x = torch.relu(
            self.bn(self.conv(x))
        )
        x = self.mpool(x)

        x = self.transition1(x)
        x = self.stage2(x)

        x = self.transition2(x)
        x = self.stage3(x)

        x = self.transition3(x)
        x = self.stage4(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
