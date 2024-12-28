import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights


class VGG11(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.cnv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=128, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.cnv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=256, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.cnv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=256, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.cnv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        
        self.cnv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.poo9 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.cnv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = nn.ReLU(inplace=True)
        
        self.cnv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.classifier = nn.Linear(512, 10)


    def forward(self, images):
        x = images
        for _, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                x = x.mean([2, 3])
            x = layer(x)
        return x
    
    
    def load_params(self, checkpoint_url, device):
        checkpoint = torch.load(checkpoint_url, map_location=device)
        state_dict_load = checkpoint['state_dict']
        i = 0
        sd = self.state_dict()
        for layer_name, params in sd.items():
            if layer_name.find('batch') != -1:
                sd[layer_name] = state_dict_load[list(state_dict_load)[i]].to(device)
            else:
                sd[layer_name] = nn.Parameter(state_dict_load[list(state_dict_load)[i]].to(device)) 
            i += 1

        self.load_state_dict(sd)


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.cnv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.cnv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.cnv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.cnv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=256, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        
        self.cnv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=256, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.cnv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(num_features=256, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.poo10 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.cnv11 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = nn.ReLU(inplace=True)
        
        self.cnv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = nn.ReLU(inplace=True)

        self.cnv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.pool14 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.cnv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn15 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = nn.ReLU(inplace=True)

        self.cnv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = nn.ReLU(inplace=True)

        self.cnv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(num_features=512, eps=0.000001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = nn.ReLU(inplace=True)

        self.pool18 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)


        self.fc19 = nn.Linear(512, 4096)
        self.relu19 = nn.ReLU(inplace=True)
        self.fc20 = nn.Linear(4096, 4096)
        self.relu20 = nn.ReLU(inplace=True)
        self.fc21 = nn.Linear(4096, 100)

    
    def forward(self, images):
        x = images
        for _, layer in self.named_children():
            if isinstance(layer, nn.Linear) and len(x.size()) != 2:
                x = torch.flatten(x, 1)
            x = layer(x)
        return x
    
    def make_layers(self, cfg, batch_norm):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers)

    def load_params(self, addr, device):
        i = 0
        state_dict_load = torch.load(addr, map_location=device)
        sd = self.state_dict()
        for layer_name, params in sd.items():
            if layer_name.find('batch') == -1:
                sd[layer_name] = nn.Parameter(state_dict_load[list(state_dict_load)[i]].to(device)) 
            i += 1

        self.load_state_dict(sd)


####MobileNetV2####
        
class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)
    
    def load_params(self, addr, device):
        i = 0
        state_dict_load = torch.load(addr, map_location=device)
        sd = self.state_dict()
        for layer_name, params in sd.items():
            if layer_name.find('batch') == -1:
                sd[layer_name] = nn.Parameter(state_dict_load[list(state_dict_load)[i]].to(device)) 
            i += 1
        self.load_state_dict(sd)


#####ResNet####

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu3 = nn.ReLU()
        
       
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.relu_class  = nn.ReLU()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.relu_class(out)
        return out
    
    def load_params(self, addr, device):
        i = 0
        state_dict_load = torch.load(addr, map_location=device)
        sd = self.state_dict()
        for layer_name, params in sd.items():
            if layer_name.find('batch') == -1:
                sd[layer_name] = nn.Parameter(state_dict_load[list(state_dict_load)[i]].to(device)) 
            i += 1
        self.load_state_dict(sd)


def load_model(network_name, device):
    if network_name == "vgg11":
        net = VGG11().to(device)
        addr = "../vgg11.cifar10.pretrained.pth"
        net.load_params(addr, device)
    
    elif network_name == "vgg16":
        net = VGG16().to(device)
        addr = r'./vgg16_cifar100.pth'
        net.load_params(addr, device)
    
    elif network_name == "resnet18":
        net = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=100).to(device)
        addr = r'./resnet18-199-best.pth'
        net.load_params(addr, device)
    
    elif network_name == "mobilenet":
        net = MobileNetV2().to(device)
        addr = r'./mobilenetv2-162-best.pth'
        net.load_params(addr, device)
    
    elif network_name == "resnet34":
        net = resnet34(weights=ResNet34_Weights.DEFAULT).to(device)
    
    elif network_name == "resnet18_imagenet":
        net = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    
    return net

