import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass
from .modules import Conv2d, LayerNorm, Sequential, Linear, Conv2d_
from utils import print_trainable_parameters, print_modules_grad


@dataclass
class Resnet_ModelArgs:
    input_chans: int = 3
    num_classes: int = 10
    num_blocks: List[int] = None # [3, 3, 3]
    channels1: int = 16
    channels2: int = 32
    channels3: int = 64
    image_size: int = 32

def f1(x, out_channels):
    return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0)

def f2(x, out_channels):
    return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x, *args):
        if args is None:
            return self.lambd(x)
        else:
            return self.lambd(x, args[0])

# Define the basic block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, image_size=32):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_seq = nn.ModuleList([
            Conv2d_(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # LayerNorm(32*32),
            Conv2d_(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # LayerNorm(32*32)
        ])
        # self.block_seq.add_module("norm1", LayerNorm(image_size**2))
        # self.block_seq.add_module("norm2", LayerNorm(image_size**2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.acti = nn.ReLU()
        self.block_len = len(self.block_seq)
        self.stride = stride
        self.downsample = False
        if in_channels != out_channels:
            self.downsample = True
            
            self.shortcut = LambdaLayer(f1)
        else:
            def f(x):
                return x
            self.shortcut = LambdaLayer(f2)

    def forward(self, x, add_noise: List[bool] = None, **kwargs):
        # out = self.block_seq[0](x, add_noise=add_noise[0])
        # shape = out.shape
        # out = self.block_seq[1](out.view(shape[0], shape[1], -1), add_noise=add_noise[1]).view(shape)
        # out = self.acti(out)
        # out = self.block_seq[2](out, add_noise=add_noise[2])
        # out = self.block_seq[3](out.view(shape[0], shape[1], -1), add_noise=add_noise[3]).view(shape)

        out = self.block_seq[0](x, add_noise=add_noise[0])
        if self.downsample:
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.bn1(out)
        out = self.acti(out)
        out = self.block_seq[1](out, add_noise=add_noise[1])
        out = self.bn2(out)
        out += self.shortcut(x, self.out_channels)

        return out
    
    def backward(self, loss):
        if self.block_seq[0].epsilon_buf is not None:
            self.block_seq[0].backward(loss)
        if self.block_seq[1].epsilon_buf is not None:  
            self.block_seq[1].backward(loss)

        # if self.block_seq[2].epsilon_buf is not None:
        #     self.block_seq[2].backward(loss)
        # if self.block_seq[3].epsilon_buf is not None:
        #     self.block_seq[3].backward(loss)


class ResNet(nn.Module):
    def __init__(self, params: Resnet_ModelArgs):
        super(ResNet, self).__init__()
        self.channels1 = params.channels1
        self.channels2 = params.channels2
        self.channels3 = params.channels3
        self.num_classes = params.num_classes
        self.num_blocks = params.num_blocks
        self.input_chans = params.input_chans
        self.image_size = params.image_size

        self.conv1 = Conv2d_(self.input_chans, self.channels1, kernel_size=3, stride=1, padding=1, bias=False)
        self.acti = nn.ReLU()

        self.layer1 = self.make_layer(BasicBlock, self.channels1, self.channels1, self.num_blocks[0], stride=1)
        self.layer2 = self.make_layer(BasicBlock, self.channels1, self.channels2, self.num_blocks[1], stride=2)
        self.layer3 = self.make_layer(BasicBlock, self.channels2, self.channels3, self.num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = Linear(self.channels3*(4**2), self.num_classes)
        self.module_len = 2 + sum(self.num_blocks)*self.layer1[0].block_len
        self.block_len = self.layer1[0].block_len
        print(self.module_len)

    def make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.ModuleList(layers)

    def forward(self, x, add_noise: List[bool] = None, **kwargs):
        if add_noise is None or add_noise == False:
            add_noise = [False] * self.module_len
        if len(add_noise) == 1 and add_noise[0] == True:
            add_noise = [True] * self.module_len
        elif len(add_noise) != self.module_len:
            raise ValueError(f"add_noise should have length {self.module_len}")
        
        out = self.conv1(x, add_noise=add_noise[0])
        out = self.acti(out)
        index = 1

        for layer in self.layer1:
            out = layer(out, add_noise=add_noise[index:index+self.block_len])
            index += self.block_len

        for layer in self.layer2:
            out = layer(out, add_noise=add_noise[index:index+self.block_len])
            index += self.block_len

        for layer in self.layer3:  
            out = layer(out, add_noise=add_noise[index:index+self.block_len])
            index += self.block_len

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out, add_noise=add_noise[-1])

        return out, None
    
    def backward(self, loss):
        if self.conv1.epsilon_buf is not None:
            self.conv1.backward(loss)

        for layer in self.layer1:
            layer.backward(loss)
        for layer in self.layer2:
            layer.backward(loss)
        for layer in self.layer3:
            layer.backward(loss)

        if self.fc.epsilon_buf is not None:
            self.fc.backward(loss)


class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d(1, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d(16, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d(16, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.ReLU(inplace=True)),
                                        #    Sequential(Conv2d(16, 16, (3, 3), (1, 1), 1, 1e-3),
                                        #               nn.ReLU(inplace=True)),
                                        #    Sequential(Conv2d(16, 16, (3, 3), (1, 1), 1, 1e-2),
                                        #               nn.ReLU(inplace=True)),
                                        #    Sequential(Conv2d(16, 16, (3, 3), (1, 1), 1, 1e-2),
                                        #               nn.ReLU(inplace=True)),
                                           Sequential(
                                                    nn.AdaptiveAvgPool2d((4, 4)),
                                                    nn.Flatten(),
                                                    Linear(16 * 4 * 4, 256, 1e-2),
                                                    nn.ReLU(inplace=True)
                                                      ),
                                           Sequential(Linear(256, 10, 1e-2))
                                           )
        self.module_len = len(self.module_w_para)

    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        for i, (block, add) in enumerate(zip(self.module_w_para,add_noise)):
            x = block(x, add)
        return x, None

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)

    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]


class VGG8(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_w_para = nn.Sequential(Sequential(Conv2d_(3, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.BatchNorm2d(16, affine=False),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d_(16, 16, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.BatchNorm2d(16, affine=False),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(Conv2d_(16, 32, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.BatchNorm2d(32, affine=False),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d_(32, 32, (3, 3), (1, 1), 1, 1e-3),
                                                      nn.BatchNorm2d(32, affine=False),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(Conv2d(32, 64, (3, 3), (1, 1), 1, 1e-2),
                                                      nn.BatchNorm2d(64, affine=False),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Conv2d(64, 64, (3, 3), (1, 1), 1, 1e-2),
                                                      nn.BatchNorm2d(64, affine=False),
                                                      nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(2)),
                                           Sequential(nn.Flatten(),
                                                      Linear(64 * 4 * 4, 256, 1e-2),
                                                      nn.ReLU(inplace=True)),
                                           Sequential(Linear(256, 10, 1e-2))
                                           )
        
        self.module_len = len(self.module_w_para)

    def forward(self, x, add_noise=None):
        if add_noise is None:
            add_noise = len(self.module_w_para) * [False]
        for i, (block, add) in enumerate(zip(self.module_w_para,add_noise)):
            x = block(x, add)
        return x, None

    def backward(self, loss):
        for seq in self.module_w_para:
            seq.backward(loss)

    def fetch_gradient(self):
        return [seq.fetch_gradient() for seq in self.module_w_para]


if __name__ == '__main__':
    params = Resnet_ModelArgs(input_chans=3, num_classes=10, num_blocks=[3,3,3], channels1=16, channels2=32, channels3=64)
    resnet = ResNet(params)
    print_trainable_parameters(resnet)
    input = torch.randn(2, 3, 32, 32)
    # Mp = nn.MaxPool2d(2)
    # print(Mp(input).shape)
    add_noise = [False] * resnet.module_len
    # add_noise[2] = True
    y = resnet(input, add_noise)
    print(resnet)
    print(resnet.module_len)
    print(y[0].shape)

    # print_trainable_parameters(resnet)
    # resnet.backward(torch.randn(2))
    # print_modules_grad(resnet)
    
    # # splitting the weights of a conv2d layer
    # input = torch.randn(1, 3, 4, 4)
    # conv1 = nn.Conv2d(3, 4, kernel_size=2, stride=1, padding=1, bias=False)
    # weight = conv1.state_dict()['weight']
    # with torch.no_grad():
    #     conv2 = nn.Conv2d(3, 2, kernel_size=2, stride=1, padding=1, bias=False)
    #     conv2.weight.copy_(weight[:2,])
    #     conv3 = nn.Conv2d(3, 2, kernel_size=2, stride=1, padding=1, bias=False)
    #     conv3.weight.copy_(weight[2:,])
    # print(conv1(input))
    # print(conv2(input))
    # print(conv3(input))

    # model1 = ResNet8()
    # model2 = VGG8()
    # input = torch.randn(2, 3, 32, 32)
    # output = model1(input)
    # output = model2(input)
    # print(output[0].shape)
