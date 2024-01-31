'''
Code adapted from: https://github.com/w-here/ASGO-113lab/tree/main/ADHR-CDNet

Zhang et al. "ADHR-CDNet: Attentive Differential High-Resolution Change Detection Network for Remote Sensing Images." IEEE Transactions on Geoscience and Remote Sensing 60 (2022): 1-13.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.acf= nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.acf(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels,middle_channels,out_channels):
        #super(Bottleneck, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels , kernel_size=1,stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.acf = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 =nn.Conv2d(middle_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.acf(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.acf(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + x
        return out


class Bottleneck_n(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        #super(Bottleneck, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.acf = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv1_1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.acf(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.acf(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.conv1_1(x)
        out1 = self.bn3(out1)

        out = out+out1
        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=False)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)
        return out


class ADHR(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        nb_filter = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv0 = nn.Conv2d(in_channels * 2, nb_filter[0], kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.acf = nn.ReLU(inplace=True)

        self.conv1_1_1 = Bottleneck_n(nb_filter[1],nb_filter[1],nb_filter[3])
        self.conv1_1_2 = Bottleneck(256,64,256)
        self.conv1_1_3 = Bottleneck(256,64,256)
        self.conv1_1_4 = Bottleneck(256,64,256)

        self.conv1_1 = nn.Conv2d(256,32,kernel_size=3,stride=1,padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(256,64,kernel_size=3,stride=2,padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1_1 = BasicBlock(32,32)
        self.conv2_1_2 = BasicBlock(32,32)
        self.conv2_1_3 = BasicBlock(32,32)
        self.conv2_1_4 = BasicBlock(32,32)
        self.conv2_2_1 = BasicBlock(64,64)
        self.conv2_2_2 = BasicBlock(64, 64)
        self.conv2_2_3 = BasicBlock(64, 64)
        self.conv2_2_4 = BasicBlock(64, 64)

        self.conv2_1to2=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn2_1to2=nn.BatchNorm2d(64)
        self.conv2_2to1=nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.bn2_2to1=nn.BatchNorm2d(32)
        #self.up2_2to1=nn.functional.upsample(scale_factor=2,mode='nearest')
        self.conv2_1to3_1=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn2_1to3_1=nn.BatchNorm2d(64)
        self.conv2_1to3_2=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.bn2_1to3_2=nn.BatchNorm2d(128)
        self.conv2_2to3=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.bn2_2to3=nn.BatchNorm2d(128)


        self.conv3_1_1=BasicBlock(nb_filter[0],nb_filter[0])
        self.conv3_1_2=BasicBlock(nb_filter[0],nb_filter[0])
        self.conv3_1_3=BasicBlock(nb_filter[0],nb_filter[0])
        self.conv3_1_4=BasicBlock(nb_filter[0],nb_filter[0])

        self.conv3_2_1=BasicBlock(nb_filter[1],nb_filter[1])
        self.conv3_2_2=BasicBlock(nb_filter[1],nb_filter[1])
        self.conv3_2_3=BasicBlock(nb_filter[1],nb_filter[1])
        self.conv3_2_4=BasicBlock(nb_filter[1],nb_filter[1])

        self.conv3_3_1=BasicBlock(nb_filter[2],nb_filter[2])
        self.conv3_3_2=BasicBlock(nb_filter[2],nb_filter[2])
        self.conv3_3_3=BasicBlock(nb_filter[2],nb_filter[2])
        self.conv3_3_4=BasicBlock(nb_filter[2],nb_filter[2])

        self.conv3_2to1=nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.bn3_2to1=nn.BatchNorm2d(32)
        #self.up3_2to1=nn.functional.upsample(scale_factor=2,mode='nearest')
        self.conv3_3to1=nn.Conv2d(128,32,kernel_size=1,stride=1,padding=0)
        self.bn3_3to1=nn.BatchNorm2d(32)
        #self.up3_3to1=nn.functional.upsample(scale_factor=4,mode='nearest')

        self.conv3_1to2=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn3_1to2=nn.BatchNorm2d(64)
        self.conv3_3to2=nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0)
        self.bn3_3to2=nn.BatchNorm2d(64)
        #self.up3_3to2=nn.functional.upsample(scale_factor=2,mode='nearest')

        self.conv3_1to3_1=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn3_1to3_1=nn.BatchNorm2d(64)
        self.conv3_1to3_2=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.bn3_1to3_2=nn.BatchNorm2d(128)
        self.conv3_2to3=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.bn3_2to3=nn.BatchNorm2d(128)

        self.conv3_1to4_1=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn3_1to4_1=nn.BatchNorm2d(64)
        self.conv3_1to4_2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3_1to4_2 = nn.BatchNorm2d(128)
        self.conv3_1to4_3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3_1to4_3 = nn.BatchNorm2d(256)
        self.conv3_2to4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3_2to4_1 = nn.BatchNorm2d(128)
        self.conv3_2to4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3_2to4_2 = nn.BatchNorm2d(256)
        self.conv3_3to4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3_3to4 = nn.BatchNorm2d(256)

        #self.conv3_3to3_4=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        #self.bn3_3to3_4=nn.BatchNorm2d(256)

        self.conv4_1_1 = BasicBlock(32, 32)
        self.conv4_1_2 = BasicBlock(32, 32)
        self.conv4_1_3 = BasicBlock(32, 32)
        self.conv4_1_4 = BasicBlock(32, 32)

        self.conv4_2_1 = BasicBlock(64, 64)
        self.conv4_2_2 = BasicBlock(64, 64)
        self.conv4_2_3 = BasicBlock(64, 64)
        self.conv4_2_4 = BasicBlock(64, 64)

        self.conv4_3_1 = BasicBlock(128, 128)
        self.conv4_3_2 = BasicBlock(128, 128)
        self.conv4_3_3 = BasicBlock(128, 128)
        self.conv4_3_4 = BasicBlock(128, 128)

        self.conv4_4_1 = BasicBlock(256, 256)
        self.conv4_4_2 = BasicBlock(256, 256)
        self.conv4_4_3 = BasicBlock(256, 256)
        self.conv4_4_4 = BasicBlock(256, 256)

        self.conv2f=nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.bn2f=nn.BatchNorm2d(32)
        self.conv3f=nn.Conv2d(128,32,kernel_size=1,stride=1,padding=0)
        self.bn3f=nn.BatchNorm2d(32)
        self.conv4f=nn.Conv2d(256,32,kernel_size=1,stride=1,padding=0)
        self.bn4f=nn.BatchNorm2d(32)

        self.conv4_2to1=nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.bn4_2to1=nn.BatchNorm2d(32)
        #self.up4_2to1=nn.functional.upsample(scale_factor=2,mode='nearest')
        self.conv4_3to1=nn.Conv2d(128,32,kernel_size=1,stride=1,padding=0)
        self.bn4_3to1=nn.BatchNorm2d(32)
        #self.up4_3to1=nn.functional.upsample(scale_factor=4,mode='nearest')
        self.conv4_4to1=nn.Conv2d(256,32,kernel_size=1,stride=1,padding=0)
        self.bn4_4to1=nn.BatchNorm2d(32)
        #self.up4_4to1=nn.functional.upsample(scale_factor=8,mode='nearest')

        self.conv4_1to2=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn4_1to2=nn.BatchNorm2d(64)
        self.conv4_3to2=nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0)
        self.bn4_3to2=nn.BatchNorm2d(64)
        #self.up4_3to2=nn.functional.upsample(scale_factor=2,mode='nearest')
        self.conv4_4to2=nn.Conv2d(256,64,kernel_size=1,stride=1,padding=0)
        self.bn4_4to2=nn.BatchNorm2d(64)
        #self.up4_4to2=nn.functional.upsample(scale_factor=4,mode='nearest')

        self.conv4_1to3_1=nn.Conv2d(32,32,kernel_size=3,stride=2,padding=1)
        self.bn4_1to3_1=nn.BatchNorm2d(32)
        self.conv4_1to3_2=nn.Conv2d(32,128,kernel_size=3,stride=2,padding=1)
        self.bn4_1to3_2=nn.BatchNorm2d(128)

        self.conv4_2to3=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.bn4_2to3=nn.BatchNorm2d(128)
        self.conv4_4to3=nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0)
        self.bn4_4to3=nn.BatchNorm2d(128)
        #self.up4_4to3=nn.functional.upsample(scale_factor=2,mode='nearest')

        self.conv4_1to4_1=nn.Conv2d(32,32,kernel_size=3,stride=2,padding=1)
        self.bn4_1to4_1=nn.BatchNorm2d(32)
        self.conv4_1to4_2=nn.Conv2d(32,32,kernel_size=3,stride=2,padding=1)
        self.bn4_1to4_2=nn.BatchNorm2d(32)
        self.conv4_1to4_3=nn.Conv2d(32,256,kernel_size=3,stride=2,padding=1)
        self.bn4_1to4_3=nn.BatchNorm2d(256)
        self.conv4_2to4_1=nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1)
        self.bn4_2to4_1=nn.BatchNorm2d(64)
        self.conv4_2to4_2=nn.Conv2d(64,256,kernel_size=3,stride=2,padding=1)
        self.bn4_2to4_2=nn.BatchNorm2d(256)
        self.conv4_3to4=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        self.bn4_3to4=nn.BatchNorm2d(256)

        self.convf2=nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
        self.bnf2=nn.BatchNorm2d(32)
        #self.upf2=nn.functional.upsample(scale_factor=2,mode='nearest')
        self.convf3=nn.Conv2d(128,32,kernel_size=1,stride=1,padding=0)
        self.bnf3=nn.BatchNorm2d(32)
        #self.upf3=nn.functional.upsample(scale_factor=4,mode='nearest')
        self.convf4=nn.Conv2d(256,32,kernel_size=1,stride=1,padding=0)
        self.bnf4=nn.BatchNorm2d(32)
        #self.upf4=nn.functional.upsample(scale_factor=8,mode='nearest')

        self.final=nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0)

        #dpm
        #self.dpms2=nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        #self.bns2=nn.BatchNorm2d(64)
        #self.dpms3=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        #self.bns3 = nn.BatchNorm2d(128)
        #self.dpms4=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        #self.bns4 = nn.BatchNorm2d(256)
        #

        self.conv2=VGGBlock(32,64,64)
        self.c2=nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0)
        self.conv3=VGGBlock(64,128,128)
        self.c3=nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0)
        self.conv4=VGGBlock(128,256,256)
        self.c4=nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)

        self.attention1 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.attention2 = nn.Conv2d(512, 4, kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)
        self.S = nn.Softmax(dim=1)


    def forward(self, x, y):
            x1=self.conv0_0(x)
            y1=self.conv0_0(y)
            d1_xy=torch.abs(x1-y1)
            input =torch.cat((x,y), 1)
            x=self.conv0(input)
            x=torch.cat((d1_xy,x), 1)
            x=self.bn0(x)
            x=self.acf(x)

            #part one
            x_1_1=self.conv1_1_1(x)
            x_1_1=self.acf(x_1_1)
            x_1_2=self.conv1_1_2(x_1_1)
            x_1_2=self.acf(x_1_2)
            x_1_3=self.conv1_1_3(x_1_2)
            x_1_3=self.acf(x_1_3)
            x_1_4=self.conv1_1_4(x_1_3)
            x_1_4=self.acf(x_1_4)

            #trasation one
            x1_1=self.conv1_1(x_1_4)
            x1_1=self.bn1_1(x1_1)
            x1_1=self.acf(x1_1)

            x1_2=self.conv1_2(x_1_4)
            x1_2=self.bn1_2(x1_2)
            x1_2=self.acf(x1_2)

            #x2=self.dpms1(x1)
            #x2=self.bns2(x2)
            #y2=self.dpms1(y1)
            #y2=self.bns2(y2)
            x1=self.pool(x1)
            y1=self.pool(y1)
            x2=self.conv2(x1)
            y2=self.conv2(y1)

            d2_xy = torch.abs(x2 - y2)
            x1_2=torch.cat((x1_2,d2_xy),1)
            x1_2=self.c2(x1_2)

            #part two
            x2_1_1=self.conv2_1_1(x1_1)
            x2_1_1=self.acf(x2_1_1)
            x2_1_2=self.conv2_1_2(x2_1_1)
            x2_1_2=self.acf(x2_1_2)
            x2_1_3=self.conv2_1_3(x2_1_2)
            x2_1_3=self.acf(x2_1_3)
            x2_1_4=self.conv2_1_4(x2_1_3)
            x2_1_4=self.acf(x2_1_4)

            x2_2_1=self.conv2_2_1(x1_2)
            x2_2_1=self.acf(x2_2_1)
            x2_2_2=self.conv2_2_2(x2_2_1)
            x2_2_2=self.acf(x2_2_2)
            x2_2_3=self.conv2_2_3(x2_2_2)
            x2_2_3=self.acf(x2_2_3)
            x2_2_4=self.conv2_2_4(x2_2_3)
            x2_2_4=self.acf(x2_2_4)

            #trasation two
            x2_1_from2=self.conv2_2to1(x2_2_4)
            x2_1_from2=self.bn2_2to1(x2_1_from2)
            x2_1_from2=nn.functional.upsample(x2_1_from2,scale_factor=2,mode='bilinear')
            #x2_1=torch.cat((x2_1_from2,x2_1_4),1)
            x2_1=x2_1_from2+x2_1_4
            x2_1=self.acf(x2_1)

            x2_2_from1=self.conv2_1to2(x2_1_4)
            x2_2_from1=self.bn2_1to2(x2_2_from1)
            #x2_2=torch.cat((x2_2_from1,x2_2_4),1)
            x2_2=x2_2_from1+x2_2_4
            x2_2=self.acf(x2_2)

            x2_3_from1=self.conv2_1to3_1(x2_1_4)
            x2_3_from1=self.bn2_1to3_1(x2_3_from1)
            x2_3_from1=self.acf(x2_3_from1)
            x2_3_from1=self.conv2_1to3_2(x2_3_from1)
            x2_3_from1=self.bn2_1to3_2(x2_3_from1)
            x2_3_from2=self.conv2_2to3(x2_2_4)
            x2_3_from2=self.bn2_2to3(x2_3_from2)
            #x2_3=torch.cat((x2_3_from1,x2_3_from2),1)
            x2_3=x2_3_from1+x2_3_from2
            x2_3=self.acf(x2_3)

            #x3=self.dpms3(x2)
            #x3=self.bns3(x3)
            #y3=self.dpms3(y2)
            #y3=self.bns3(y3)
            x2=self.pool(x2)
            y2=self.pool(y2)
            x3 = self.conv3(x2)
            y3 = self.conv3(y2)

            d3_xy = torch.abs(x3 - y3)
            x2_3 = torch.cat((x2_3, d3_xy), 1)
            x2_3=self.c3(x2_3)


            #part three
            x3_1_1=self.conv3_1_1(x2_1)
            x3_1_1=self.acf(x3_1_1)
            x3_1_2=self.conv3_1_2(x3_1_1)
            x3_1_2=self.acf(x3_1_2)
            x3_1_3=self.conv3_1_3(x3_1_2)
            x3_1_3=self.acf(x3_1_3)
            x3_1_4=self.conv3_1_4(x3_1_3)
            x3_1_4=self.acf(x3_1_4)

            x3_2_1=self.conv3_2_1(x2_2)
            x3_2_1=self.acf(x3_2_1)
            x3_2_2=self.conv3_2_2(x3_2_1)
            x3_2_2=self.acf(x3_2_2)
            x3_2_3=self.conv3_2_3(x3_2_2)
            x3_2_3=self.acf(x3_2_3)
            x3_2_4=self.conv3_2_4(x3_2_3)
            x3_2_4=self.acf(x3_2_4)

            x3_3_1=self.conv3_3_1(x2_3)
            x3_3_1=self.acf(x3_3_1)
            x3_3_2=self.conv3_3_2(x3_3_1)
            x3_3_2=self.acf(x3_3_2)
            x3_3_3=self.conv3_3_3(x3_3_2)
            x3_3_3=self.acf(x3_3_3)
            x3_3_4=self.conv3_3_4(x3_3_3)
            x3_3_4=self.acf(x3_3_4)

            #trasation THREE
            x3_1from2=self.conv3_2to1(x3_2_4)
            x3_1from2=self.bn3_2to1(x3_1from2)
            x3_1from2=nn.functional.upsample(x3_1from2,scale_factor=2,mode='bilinear')
            x3_1from3=self.conv3_3to1(x3_3_4)
            x3_1from3=self.bn3_3to1(x3_1from3)
            x3_1from3=nn.functional.upsample(x3_1from3,scale_factor=4,mode='bilinear')
            #x3_1=torch.cat((x3_1_4,x3_1from2,x3_1from3),1)
            x3_1=x3_1_4+x3_1from2+x3_1from3
            x3_1=self.acf(x3_1)

            x3_2from1=self.conv3_1to2(x3_1_4)
            x3_2from1=self.bn3_1to2(x3_2from1)
            x3_2from3=self.conv3_3to2(x3_3_4)
            x3_2from3=self.bn3_3to2(x3_2from3)
            x3_2from3=nn.functional.upsample(x3_2from3,scale_factor=2,mode='bilinear')
            #x3_2=torch.cat((x3_2from1,x3_2_4,x3_2from3),1)
            x3_2=x3_2from1+x3_2_4+x3_2from3
            x3_2=self.acf(x3_2)

            x3_3from1=self.conv3_1to3_1(x3_1_4)
            x3_3from1=self.bn3_1to3_1(x3_3from1)
            x3_3from1=self.acf(x3_3from1)
            x3_3from1=self.conv3_1to3_2(x3_3from1)
            x3_3from1=self.bn3_1to3_2(x3_3from1)
            x3_3from2=self.conv3_2to3(x3_2_4)
            x3_3from2=self.bn3_2to3(x3_3from2)
            #x3_3=torch.cat((x3_3_4,x3_3from1,x3_3from2),1)
            x3_3=x3_3_4+x3_3from1+x3_3from2
            x3_3=self.acf(x3_3)

            x3_4from1=self.conv3_1to4_1(x3_1_4)
            x3_4from1=self.bn3_1to4_1(x3_4from1)
            x3_4from1=self.acf(x3_4from1)
            x3_4from1 = self.conv3_1to4_2(x3_4from1)
            x3_4from1 = self.bn3_1to4_2(x3_4from1)
            x3_4from1 = self.acf(x3_4from1)
            x3_4from1 = self.conv3_1to4_3(x3_4from1)
            x3_4from1 = self.bn3_1to4_3(x3_4from1)

            x3_4from2 = self.conv3_2to4_1(x3_2_4)
            x3_4from2 = self.bn3_2to4_1(x3_4from2)
            x3_4from2 = self.acf(x3_4from2)
            x3_4from2 = self.conv3_2to4_2(x3_4from2)
            x3_4from2 = self.bn3_2to4_2(x3_4from2)

            x3_4from3 = self.conv3_3to4(x3_3_4)
            x3_4from3 = self.bn3_3to4(x3_4from3)

            #x3_4=torch.cat((x3_4from1,x3_4from2,x3_4from3),1)
            x3_4=x3_4from1+x3_4from2+x3_4from3
            x3_4 = self.acf(x3_4)

            #x4=self.dpms4(x3)
            #x4=self.bns4(x4)
            #y4=self.dpms4(y3)
            #y4=self.bns4(y4)
            x3=self.pool(x3)
            y3=self.pool(y3)
            x4 = self.conv4(x3)
            y4 = self.conv4(y3)
            d4_xy = torch.abs(x4 - y4)
            x3_4 = torch.cat((x3_4, d4_xy), 1)
            x3_4=self.c4(x3_4)


            x4_1_1 = self.conv4_1_1(x3_1)
            x4_1_1 = self.acf(x4_1_1)
            x4_1_2 = self.conv4_1_2(x4_1_1)
            x4_1_2 = self.acf(x4_1_2)
            x4_1_3 = self.conv4_1_3(x4_1_2)
            x4_1_3 = self.acf(x4_1_3)
            x4_1_4 = self.conv4_1_4(x4_1_3)
            x4_1_4 = self.acf(x4_1_4)

            x4_2_1 = self.conv4_2_1(x3_2)
            x4_2_1 = self.acf(x4_2_1)
            x4_2_2 = self.conv4_2_2(x4_2_1)
            x4_2_2 = self.acf(x4_2_2)
            x4_2_3 = self.conv4_2_3(x4_2_2)
            x4_2_3 = self.acf(x4_2_3)
            x4_2_4 = self.conv4_2_4(x4_2_3)
            x4_2_4 = self.acf(x4_2_4)

            x4_3_1 = self.conv4_3_1(x3_3)
            x4_3_1 = self.acf(x4_3_1)
            x4_3_2 = self.conv4_3_2(x4_3_1)
            x4_3_2 = self.acf(x4_3_2)
            x4_3_3 = self.conv4_3_3(x4_3_2)
            x4_3_3 = self.acf(x4_3_3)
            x4_3_4 = self.conv4_3_4(x4_3_3)
            x4_3_4 = self.acf(x4_3_4)

            x4_4_1 = self.conv4_4_1(x3_4)
            x4_4_1 = self.acf(x4_4_1)
            x4_4_2 = self.conv4_4_2(x4_4_1)
            x4_4_2 = self.acf(x4_4_2)
            x4_4_3 = self.conv4_4_3(x4_4_2)
            x4_4_3 = self.acf(x4_4_3)
            x4_4_4 = self.conv4_4_4(x4_4_3)
            x4_4_4 = self.acf(x4_4_4)

            x1f=x4_1_4
            x2f=self.conv2f(x4_2_4)
            x2f=self.bn2f(x2f)
            x2f=nn.functional.upsample(x2f,scale_factor=2,mode='bilinear')
            x3f=self.conv3f(x4_3_4)
            x3f=self.bn3f(x3f)
            x3f=nn.functional.upsample(x3f,scale_factor=4,mode='bilinear')
            x4f=self.conv4f(x4_4_4)
            x4f=self.bn4f(x4f)
            x4f=nn.functional.upsample(x4f,scale_factor=8,mode='bilinear')

            out=torch.cat((x1f,x2f,x3f,x4f),1)

            out = self.attention1(out)
            out = self.attention2(out)
            attention_map = self.S(out)
            w1, w2, w3, w4 = attention_map.split(1, 1)
            # print(w1.shape)

            out1 = x1f.mul(w1) + x2f.mul(w2) + x3f.mul(w3) + x4f.mul(w4)
            out = self.S(self.final(out1))
            #out=x1f+x2f+x3f+x4f
            #out=self.acf(out)
            #out=self.final(out)

            #output1 = self.final1(x0_1)
            #output1=m(output1)
            #output2 = self.final2(x0_2)
            #output2=m(output2)
            #output3 = self.final3(x0_3)
            #output3=m(output3)
            #output4 = self.final4(x0_4)
            #output4=m(output4)
            #z= torch.cat((output1,output2,output3, output4), 1)\
            #z=self.conv_final(z)
            #z=m(z)
            #output = [output1, output2, output3, output4, z]

            return out