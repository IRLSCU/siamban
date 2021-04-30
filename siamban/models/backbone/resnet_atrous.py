import math

import torch.nn as nn
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def split(x, groups):
    out = x.chunk(groups, dim=1)

    return out


def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    return out

class ShufflenetBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(ShufflenetBlock, self).__init__()
        mid_channels = (planes * ShufflenetBlock.expansion) // 2
        self.mid_channels = mid_channels

        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation

        ##print(mid_channels)
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, 3, stride=stride, padding = padding, groups=inplanes, bias=False, dilation=dilation),
                nn.BatchNorm2d(inplanes),
                nn.Conv2d(inplanes, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(inplanes, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=padding, groups=mid_channels, bias=False, dilation=dilation),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.downsample = downsample
            self.stride = stride
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inplanes // 2 , mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                # 第一卷积
                nn.Conv2d(inplanes // 2 , mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # 第二卷积
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding = padding, groups=mid_channels, bias=False, dilation=dilation),
                nn.BatchNorm2d(mid_channels),
                # 第三卷积
                nn.Conv2d(mid_channels, mid_channels , 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        # self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        if self.stride == 1:
            x1, x2 = split(x, 2)
            a1 = self.branch1(x1)
            a2 = self.branch2(x2)
        else:
            a1 = self.branch1(x)
            a2 = self.branch2(x)
        out = torch.cat((a1, a2), dim=1)
        
        # print(out.shape)
        # print(residual.shape)

#        out = self.relu(out)
        out = shuffle(out, 2)
        
        ##out = self.conv3(out)
        ##out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        #out = self.relu(out)
        #print(out.shape)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        """

        Args:
            inplanes ([type]): 输入动刀数目
            planes ([type]): 输入通道数目，存在下采样函数时，由下采样函数定义，否则与输入相同
            stride (int, optional): [description]. Defaults to 1.
            downsample ([type], optional): [description]. Defaults to None.
            dilation (int, optional): [description]. Defaults to 1.
        """
        super(Bottleneck, self).__init__()
        ##print(inplanes)
        ## print(planes)
        # 1*1 换维度卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        # 3*3 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1*1 维度升级卷积
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        ##print(planes *4)
        self.relu = nn.ReLU(inplace=True)

        # 进行下采样
        self.downsample = downsample
        self.stride = stride
        ##print(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 进行维度升级操用
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            #print(residual.shape)

        out += residual

        out = self.relu(out)
        #print(out.shape)
        print("------------")
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, used_layers):
        """

        Args:
            block ([type]): 基础模块类型指针
            layers ([type]): 模块层数数组
            used_layers ([type]): 特征层返回数组 

        model = ResNet(Bottleneck, [3, 4, 6, 3],[2,3,4])
        """
        # 输入通道数
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 进行下采样计算机
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 开始卷积层定义
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers or 4 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        模块组合构建函数
        Args:
            block ([type]): 基础结构类指针
            planes ([type]): 输出通道数目
            blocks ([type]): 模块数量
            stride (int, optional): 步长. Defaults to 1.
            dilation (int, optional): 卷积空洞率. Defaults to 1.

        Returns:
            [type]: layers数组
        """
        downsample = None
        dd = dilation
        # 特殊情况需要使用下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        
        # 首先添加下采样层
        layers.append(block(self.inplanes, planes , stride,
                            downsample, dilation=dilation))
        # 设置输出通道数目
        self.inplanes = planes * block.expansion


        # 设置接下来简单几个层
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 卷积1
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        # 
        x = self.maxpool(x_)

        p1 = self.layer1(x) 
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        out = [x_, p1, p2, p3, p4]
        for temp in out:
            print(temp.shape)
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out


class ShuffleNet(nn.Module):
    def __init__(self, block, layers, used_layers):
        """

        Args:
            block ([type]): 基础模块类型指针
            layers ([type]): 模块层数数组
            used_layers ([type]): 特征层返回数组 

        model = ResNet(Bottleneck, [3, 4, 6, 3],[2,3,4])
        """
        # 输入通道数
        self.inplanes = 64
        super(ShuffleNet, self).__init__()
        # 进行下采样计算机
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 注意这里的下采样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 开始卷积层定义
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers or 4 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        模块组合构建函数
        Args:
            block ([type]): 基础结构类指针
            planes ([type]): 输出通道数目
            blocks ([type]): 模块数量
            stride (int, optional): 步长. Defaults to 1.
            dilation (int, optional): 卷积空洞率. Defaults to 1.

        Returns:
            [type]: layers数组
        """
        downsample = None
        dd = dilation
        # 特殊情况需要使用下采样 -- 特殊步长与输入数目与目标扩充数目不同时
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 标准卷积步长为1的卷积
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                # 非标准卷积
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        
        # 首先添加下采样层
        layers.append(block(self.inplanes, planes , stride,
                            downsample, dilation=dilation))
        # 设置输出通道数目
        self.inplanes = planes * block.expansion


        # 设置接下来简单几个层
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 卷积1
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        # 1 64 125 125 
        x = self.maxpool(x_)
        # 1*256*63*63
        p1 = self.layer1(x)
        # 1*512*31*31 
        p2 = self.layer2(p1)
        # 1*2048*31*31
        p3 = self.layer3(p2)
        # 1*2048*15*15
        p4 = self.layer4(p3)
        out = [x_, p1, p2, p3, p4]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def shufflenet50(**kwargs):
    model = ShuffleNet(ShufflenetBlock, [3, 4, 6, 3], **kwargs)
    return model



if __name__ == '__main__':
    #net = resnet50(used_layers=[2, 3, 4])
    net = shufflenet50(used_layers=[2, 3, 4])
    # print(net.layer1)
    # print(net.layer2)
    # print(net.layer3)
    # print(net.layer4)
    net = net.cuda()

    template_var = torch.FloatTensor(1, 3, 127, 127).cuda()
    search_var = torch.FloatTensor(1, 3, 255, 255).cuda()

    t = net(template_var)
    s = net(search_var)
    print(t[-1].shape, s[-1].shape)
