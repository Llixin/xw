from torch import nn
import torch
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
class Res2NetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

        self.shortcut = nn.Sequential()
        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes)

            )


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out
class Res2Net(nn.Module):
    def __init__(self, num_classes=19):
        super(Res2Net, self).__init__()

        # input: 1, num, features_num
        base_channel=64
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(1, 64, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 2
            Res2NetBottleneck(64, 128, norm_layer=nn.BatchNorm2d, se=True, scales=4, groups=2),
            nn.MaxPool2d(2,2),  #(bs, 128, 30, 4)
            # 3
            Res2NetBottleneck(128, 128, norm_layer=nn.BatchNorm2d, se=True, scales=4, groups=2),
            nn.MaxPool2d((1, 2)),   #(bs, 128, 30, 2)
            # 4
            Res2NetBottleneck(128, 256, norm_layer=nn.BatchNorm2d, se=True, scales=4, groups=2),
            # 5
            Res2NetBottleneck(256, 512, norm_layer=nn.BatchNorm2d, se=True, scales=4, groups=2),


        )
        self.classier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)    #(bs, 512, 30, 2)
        bs, chan, frames, feat = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(bs, chan * feat, frames) #(bs, 30, 1024)
        # print(x.size())
        x = nn.AdaptiveAvgPool1d(1)(x)
        feature = x.reshape(bs, 1024)   #(bs, 1024)
        #x = x.view(x.shape[0], -1)
        x = self.classier(feature)  #(bs, 19)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(CELoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss=  nn.NLLLoss(reduction=reduction)
    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        x = self.log_softmax(x)

        target=torch.argmax(target,dim=-1)

        loss=self.nllloss(x,target=target)

        return loss

import torch
import torch.nn as nn