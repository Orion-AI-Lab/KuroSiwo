'''
Code adapted from https://github.com/HaiXing-1998/HFANet

Zheng et al. "HFA-Net: High frequency attention siamese network for building change detection in VHR remote sensing images". In Pattern Recognition, vol. 129, 2022. DOI: 10.1016/j.patcog.2022.108717.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as function


class HighFrequencyModule(nn.Module):
    def __init__(self, input_channel, the_filter='Isotropic_Sobel', mode='filtering', parameter_a=1, parameter_k=0.5,
                 smooth=False):
        """High Frequency Module for CNN.
        Extract the high frequency component of features.
        Args:
            input_channel (int): Number of channels inputted.
            the_filter (str): Decide which filter to use ('Isotropic_Sobel','Krisch','Laplacian_1,_2 and _3','LOG'.).
            mode (str): Decide which mode this module to work in ('filtering' or 'high_boost_filtering'.).
            parameter_a (float): When the module work in the high boost filtering mode, the parameter_a decide the
                                strength of the original features.
            parameter_k (float): When the module work in the high boost filtering mode, the parameter_a decide the
                                strength of the high frequency features extracted from the original features.
        """
        super(HighFrequencyModule, self).__init__()
        self.filter = the_filter
        self.mode = mode
        self.channel = input_channel
        self.A = parameter_a
        self.K = parameter_k
        self.smooth = smooth
        # Gaussian Smooth
        kernel_gaussian_smooth = [[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]
        kernel_smooth = torch.FloatTensor(kernel_gaussian_smooth).expand(self.channel, self.channel, 3, 3).clone()
        self.weight_smooth = nn.Parameter(data=kernel_smooth, requires_grad=False)
        # Isotropic Sobel
        kernel_isotropic_sobel_direction_1 = [[1, math.sqrt(2), 1],
                                              [0, 0, 0],
                                              [-1, -math.sqrt(2), -1]]
        kernel_isotropic_sobel_direction_2 = [[0, 1, math.sqrt(2)],
                                              [-1, 0, 1],
                                              [-math.sqrt(2), -1, 0]]
        kernel_isotropic_sobel_direction_3 = [[-1, 0, 1],
                                              [-math.sqrt(2), 0, math.sqrt(2)],
                                              [-1, 0, 1]]
        kernel_isotropic_sobel_direction_4 = [[math.sqrt(2), 1, 0],
                                              [1, 0, -1],
                                              [0, -1, -math.sqrt(2)]]
        # kernel_isotropic_sobel_direction_5 = -1 * kernel_isotropic_sobel_direction_1
        # kernel_isotropic_sobel_direction_6 = -1 * kernel_isotropic_sobel_direction_2
        # kernel_isotropic_sobel_direction_7 = -1 * kernel_isotropic_sobel_direction_3
        # kernel_isotropic_sobel_direction_8 = -1 * kernel_isotropic_sobel_direction_4
        # Krisch
        kernel_krisch_direction_1 = [[5, 5, 5],
                                     [-3, 0, -3],
                                     [-3, -3, -3]]
        kernel_krisch_direction_2 = [[-3, 5, 5],
                                     [-3, 0, 5],
                                     [-3, -3, -3]]
        kernel_krisch_direction_3 = [[-3, -3, 5],
                                     [-3, 0, 5],
                                     [-3, -3, 5]]
        kernel_krisch_direction_4 = [[-3, -3, -3],
                                     [-3, 0, 5],
                                     [-3, 5, 5]]
        kernel_krisch_direction_5 = [[-3, -3, -3],
                                     [-3, 0, -3],
                                     [5, 5, 5]]
        kernel_krisch_direction_6 = [[-3, -3, -3],
                                     [5, 0, -3],
                                     [5, 5, -3]]
        kernel_krisch_direction_7 = [[5, -3, -3],
                                     [5, 0, -3],
                                     [5, -3, -3]]
        kernel_krisch_direction_8 = [[5, 5, -3],
                                     [5, 0, -3],
                                     [-3, -3, -3]]
        # Laplacian
        kernel_laplacian_1 = [[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]]
        kernel_laplacian_2 = [[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]]
        kernel_laplacian_3 = [[1, -2, 1],
                              [-2, 4, -2],
                              [1, -2, 1]]
        # LOG
        kernel_log = [[-2, -4, -4, -4, -2],
                      [-4, 0, 8, 0, -4],
                      [-4, 8, 24, 8, -4],
                      [-4, 0, 8, 0, -4],
                      [-2, -4, -4, -4, -2]]
        if self.filter == 'Isotropic_Sobel':
            kernel_1 = torch.FloatTensor(kernel_isotropic_sobel_direction_1).expand(self.channel, self.channel, 3, 3).clone()
            kernel_2 = torch.FloatTensor(kernel_isotropic_sobel_direction_2).expand(self.channel, self.channel, 3, 3).clone()
            kernel_3 = torch.FloatTensor(kernel_isotropic_sobel_direction_3).expand(self.channel, self.channel, 3, 3).clone()
            kernel_4 = torch.FloatTensor(kernel_isotropic_sobel_direction_4).expand(self.channel, self.channel, 3, 3).clone()
            kernel_5 = -1 * kernel_1
            kernel_6 = -1 * kernel_2
            kernel_7 = -1 * kernel_3
            kernel_8 = -1 * kernel_4
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
            self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)
            self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)
            self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
            self.weight_5 = nn.Parameter(data=kernel_5, requires_grad=False)
            self.weight_6 = nn.Parameter(data=kernel_6, requires_grad=False)
            self.weight_7 = nn.Parameter(data=kernel_7, requires_grad=False)
            self.weight_8 = nn.Parameter(data=kernel_8, requires_grad=False)
        elif self.filter == 'Krisch':
            kernel_1 = torch.FloatTensor(kernel_krisch_direction_1).expand(self.channel, self.channel, 3, 3).clone()
            kernel_2 = torch.FloatTensor(kernel_krisch_direction_2).expand(self.channel, self.channel, 3, 3).clone()
            kernel_3 = torch.FloatTensor(kernel_krisch_direction_3).expand(self.channel, self.channel, 3, 3).clone()
            kernel_4 = torch.FloatTensor(kernel_krisch_direction_4).expand(self.channel, self.channel, 3, 3).clone()
            kernel_5 = torch.FloatTensor(kernel_krisch_direction_5).expand(self.channel, self.channel, 3, 3).clone()
            kernel_6 = torch.FloatTensor(kernel_krisch_direction_6).expand(self.channel, self.channel, 3, 3).clone()
            kernel_7 = torch.FloatTensor(kernel_krisch_direction_7).expand(self.channel, self.channel, 3, 3).clone()
            kernel_8 = torch.FloatTensor(kernel_krisch_direction_8).expand(self.channel, self.channel, 3, 3).clone()
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
            self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)
            self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)
            self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
            self.weight_5 = nn.Parameter(data=kernel_5, requires_grad=False)
            self.weight_6 = nn.Parameter(data=kernel_6, requires_grad=False)
            self.weight_7 = nn.Parameter(data=kernel_7, requires_grad=False)
            self.weight_8 = nn.Parameter(data=kernel_8, requires_grad=False)
        elif self.filter == 'Laplacian_1':
            kernel_1 = torch.FloatTensor(kernel_laplacian_1).expand(self.channel, self.channel, 3, 3).clone()
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        elif self.filter == 'Laplacian_2':
            kernel_1 = torch.FloatTensor(kernel_laplacian_2).expand(self.channel, self.channel, 3, 3).clone()
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        elif self.filter == 'Laplacian_3':
            kernel_1 = torch.FloatTensor(kernel_laplacian_3).expand(self.channel, self.channel, 3, 3).clone()
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        elif self.filter == 'LOG':
            kernel_1 = torch.FloatTensor(kernel_log).expand(self.channel, self.channel, 5, 5).clone()
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)

    def forward(self, x):
        # pretreatment
        if self.smooth:
            x = function.conv2d(x, self.weight_smooth, stride=1, padding=1)
            x = x / 16
        x_result = x
        x_high_frequency = x
        # filter choose
        if self.filter == 'Isotropic_Sobel' or 'Krisch':
            x1 = function.conv2d(x, self.weight_1, stride=1, padding=1)
            x2 = function.conv2d(x, self.weight_2, stride=1, padding=1)
            x3 = function.conv2d(x, self.weight_3, stride=1, padding=1)
            x4 = function.conv2d(x, self.weight_4, stride=1, padding=1)
            x5 = function.conv2d(x, self.weight_5, stride=1, padding=1)
            x6 = function.conv2d(x, self.weight_6, stride=1, padding=1)
            x7 = function.conv2d(x, self.weight_7, stride=1, padding=1)
            x8 = function.conv2d(x, self.weight_8, stride=1, padding=1)
            x_high_frequency = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8) / 8
        elif self.filter == 'Laplacian_1' or 'Laplacian_2' or 'Laplacian_3':
            x_high_frequency = function.conv2d(x, self.weight_1, stride=1, padding=1)
        elif self.filter == 'LOG':
            x_high_frequency = function.conv2d(x, self.weight_1, stride=1, padding=2)
        # mode choose
        if self.mode == 'filtering':
            x_result = x_high_frequency
        elif self.mode == 'high_boost_filtering':
            x_result = self.A * x + self.K * x_high_frequency
        return x_result


class HighFrequencyEnhancementStage(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5):
        super(HighFrequencyEnhancementStage, self).__init__()
        self.input_channel = input_channel
        self.input_size = input_size
        self.ratio_channel = int(ratio * input_channel)
        self.Global_pooling = nn.AvgPool2d(self.input_size)
        self.FC_1 = nn.Linear(self.input_channel, int(self.input_channel * ratio))
        self.ReLU = nn.PReLU(int(self.input_channel * ratio))
        self.FC_2 = nn.Linear(int(self.input_channel * ratio), self.input_channel)
        self.Sigmoid = nn.Sigmoid()
        self.HighFre = HighFrequencyModule(input_channel=self.input_channel,smooth=True)
        self.Channelfusion = nn.Conv2d(2 * self.input_channel, self.input_channel, kernel_size=1, stride=1)

    # ChannelAttention +HighFrequency
    def forward(self, x):
        residual = x  # residual & x's shape [batch size, channel, input size, input size]
        x_hf = self.HighFre(residual)
        x = self.Global_pooling(x)  # x's shape [batch size, channel, 1, 1]
        x = x.view(-1, self.input_channel)  # x's shape [batch size, channel]
        x = self.FC_1(x)  # x's shape [batch size, ratio channel]
        x = self.ReLU(x)
        x = self.FC_2(x)  # x's shape [batch size, channel]
        x = self.Sigmoid(x)
        x = torch.unsqueeze(x, dim=2)  # x's shape [batch size, channel, 1]
        residual_0 = residual.view(-1, self.input_channel, self.input_size ** 2)
        residual_0 = torch.mul(residual_0, x)
        residual_0 = residual_0.contiguous().view(-1, self.input_channel, self.input_size, self.input_size).clone()
        x_output = residual + residual_0
        x_output = torch.cat((x_output, x_hf), dim=1)
        x_output = self.Channelfusion(x_output)
        return x_output


class SpatialAttentionStage(nn.Module):
    def __init__(self, input_channel, last_layer=False):
        super(SpatialAttentionStage, self).__init__()
        self.bn_momentum = 0.1
        self.input_channel = input_channel
        # down 1
        self.conv1_1 = nn.Conv2d(self.input_channel, self.input_channel // 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn1_1 = nn.BatchNorm2d(self.input_channel // 2,
                                    momentum=self.bn_momentum)
        self.ReLU1_1 = nn.PReLU(self.input_channel // 2)
        self.conv1_2 = nn.Conv2d(self.input_channel // 2, self.input_channel // 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn1_2 = nn.BatchNorm2d(self.input_channel // 2,
                                    momentum=self.bn_momentum)
        self.ReLU1_2 = nn.PReLU(self.input_channel // 2)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        # down 2
        self.conv2_1 = nn.Conv2d(self.input_channel // 2, self.input_channel // 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn2_1 = nn.BatchNorm2d(self.input_channel // 4,
                                    momentum=self.bn_momentum)
        self.ReLU2_1 = nn.PReLU(self.input_channel // 4)
        self.conv2_2 = nn.Conv2d(self.input_channel // 4, self.input_channel // 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn2_2 = nn.BatchNorm2d(self.input_channel // 4,
                                    momentum=self.bn_momentum)
        self.ReLU2_2 = nn.PReLU(self.input_channel // 4)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        # bottom
        self.conv_b_1 = nn.Conv2d(self.input_channel // 4, self.input_channel // 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_b_1 = nn.BatchNorm2d(self.input_channel // 8,
                                     momentum=self.bn_momentum)
        self.ReLU_b_1 = nn.PReLU(self.input_channel // 8)
        self.conv_b_2 = nn.Conv2d(self.input_channel // 8, self.input_channel // 8,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_b_2 = nn.BatchNorm2d(self.input_channel // 8,
                                     momentum=self.bn_momentum)
        self.ReLU_b_2 = nn.PReLU(self.input_channel // 8)
        # up 1
        if last_layer:
            self.convtrans_1 = nn.ConvTranspose2d(self.input_channel // 8, self.input_channel // 16,
                                                  kernel_size=3,
                                                  stride=3,
                                                  padding=2,
                                                  output_padding=2)
        else:
            self.convtrans_1 = nn.ConvTranspose2d(self.input_channel // 8, self.input_channel // 16,
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1,
                                                  output_padding=1)
        self.conv3_1 = nn.Conv2d(self.input_channel // 16 + self.input_channel // 4, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn3_1 = nn.BatchNorm2d(self.input_channel // 16,
                                    momentum=self.bn_momentum)
        self.ReLU3_1 = nn.PReLU(self.input_channel // 16)
        self.conv3_2 = nn.Conv2d(self.input_channel // 16, self.input_channel // 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn3_2 = nn.BatchNorm2d(self.input_channel // 16,
                                    momentum=self.bn_momentum)
        self.ReLU3_2 = nn.PReLU(self.input_channel // 16)
        # up 2
        self.convtrans_2 = nn.ConvTranspose2d(self.input_channel // 16, self.input_channel // 32,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1)
        self.conv4_1 = nn.Conv2d(self.input_channel // 32 + self.input_channel // 2, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn4_1 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU4_1 = nn.PReLU(self.input_channel // 32)
        self.conv4_2 = nn.Conv2d(self.input_channel // 32, self.input_channel // 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.bn4_2 = nn.BatchNorm2d(self.input_channel // 32,
                                    momentum=self.bn_momentum)
        self.ReLU4_2 = nn.PReLU(self.input_channel // 32)
        # out
        self.conv5_1 = nn.Conv2d(self.input_channel // 32, self.input_channel // 64,
                                 kernel_size=1,
                                 stride=1)
        self.bn5_1 = nn.BatchNorm2d(self.input_channel // 64,
                                    momentum=self.bn_momentum)
        self.ReLU5_1 = nn.PReLU(self.input_channel // 64)
        self.conv5_2 = nn.Conv2d(self.input_channel // 64, 1,
                                 kernel_size=1,
                                 stride=1)
        self.bn5_2 = nn.BatchNorm2d(1,
                                    momentum=self.bn_momentum)
        # self.ReLU5_2 = nn.PReLU(1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        # down 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.ReLU1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.ReLU1_2(x)
        # skip connection
        skip_1 = x
        x = self.maxpooling1(x)
        # down 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.ReLU2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.ReLU2_2(x)
        # skip connection
        skip_2 = x
        x = self.maxpooling2(x)
        # bottom
        x = self.conv_b_1(x)
        x = self.bn_b_1(x)
        x = self.ReLU_b_1(x)
        x = self.conv_b_2(x)
        x = self.bn_b_2(x)
        x = self.ReLU_b_2(x)
        # up 1
        x = self.convtrans_1(x)
        # cat skip connection
        x = torch.cat((x, skip_2), dim=1)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.ReLU3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.ReLU3_2(x)
        # up 2
        x = self.convtrans_2(x)
        # cat skip connection
        x = torch.cat((x, skip_1), dim=1)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.ReLU4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.ReLU4_2(x)
        # out
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.ReLU5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.Sigmoid(x)
        mask = torch.mul(residual, x)
        output = residual + mask
        return output


class HFAB(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5, last_layer=False):
        super(HFAB, self).__init__()
        self.SA = SpatialAttentionStage(input_channel=input_channel, last_layer=last_layer)
        self.HF = HighFrequencyEnhancementStage(input_channel=input_channel,
                                                               input_size=input_size,
                                                               ratio=ratio)

    def forward(self, x):
        x = self.SA(x)
        x = self.HF(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_channel, input_size):
        super(Encoder, self).__init__()
        bn_momentum = 0.1
        # pre_treat_layer
        self._pre_treat_1 = HighFrequencyModule(input_channel,
                                                mode='high_boost_filtering',
                                                smooth=True)
        self._pre_treat_2 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=1, stride=1)
        # layer_1
        self._layer_1 = nn.Sequential(
            HFAB(input_channel=64, input_size=input_size),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.PReLU(64),
            HFAB(input_channel=64, input_size=input_size),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.PReLU(64)
        )
        # skip_connection_1 & down_sample
        self._down_sample_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_2
        self._layer_2 = nn.Sequential(
            HFAB(input_channel=64, input_size=input_size // 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.PReLU(128),
            HFAB(input_channel=128, input_size=input_size // 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.PReLU(128)
        )
        # skip_connection_2 & down_sample
        self._down_sample_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_3
        self._layer_3 = nn.Sequential(
            HFAB(input_channel=128, input_size=input_size // 4),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.PReLU(256),
            HFAB(input_channel=256, input_size=input_size // 4),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.PReLU(256)
        )
        # skip_connection_3 & down_sample
        self._down_sample_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_4
        self._layer_4 = nn.Sequential(
            HFAB(input_channel=256, input_size=input_size // 8),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.PReLU(512),
            HFAB(input_channel=512, input_size=input_size // 8),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.PReLU(512)
        )
        # skip_connection_4 & down_sample
        self._down_sample_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_5
        self._layer_5 = nn.Sequential(
            HFAB(input_channel=512, input_size=input_size // 16, last_layer=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=bn_momentum),
            nn.PReLU(1024),
            HFAB(input_channel=1024, input_size=input_size // 16, last_layer=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=bn_momentum),
            nn.PReLU(1024)
        )

    def forward(self, x):
        # pre-treat layer
        x = self._pre_treat_1(x)
        x = self._pre_treat_2(x)
        # layer 1
        x = self._layer_1(x)
        skip_1 = x
        x = self._down_sample_1(x)
        # layer 2
        x = self._layer_2(x)
        skip_2 = x
        x = self._down_sample_2(x)
        # layer 3
        x = self._layer_3(x)
        skip_3 = x
        x = self._down_sample_3(x)
        # layer 4
        x = self._layer_4(x)
        skip_4 = x
        x = self._down_sample_4(x)
        x = self._layer_5(x)
        return x, skip_1, skip_2, skip_3, skip_4


class Decoder(nn.Module):
    def __init__(self, input_channel, input_size, num_classes):
        super(Decoder, self).__init__()
        bn_momentum = 0.1
        # up_sample_1
        # self._up_sample_1 = nn.ConvTranspose2d(input_channel, input_channel // 2,
        #                                       kernel_size=2,
        #                                       stride=2)
        self._up_sample_1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel // 2, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_1
        self._up_layer_1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 2, momentum=bn_momentum),
            nn.PReLU(input_channel // 2),
            HFAB(input_channel=input_channel // 2, input_size=input_size * 2),
            nn.Conv2d(input_channel // 2, input_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 2, momentum=bn_momentum),
            nn.PReLU(input_channel // 2),
            HFAB(input_channel=input_channel // 2, input_size=input_size * 2)
        )
        # up_sample_2
        # self._up_sample_2 = nn.ConvTranspose2d(input_channel // 2, input_channel // 4,
        #                                        kernel_size=2,
        #                                        stride=2)
        self._up_sample_2 = nn.Sequential(
            nn.Conv2d(input_channel // 2, input_channel // 4, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_2
        self._up_layer_2 = nn.Sequential(
            nn.Conv2d(input_channel // 2, input_channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 4, momentum=bn_momentum),
            nn.PReLU(input_channel // 4),
            HFAB(input_channel=input_channel // 4, input_size=input_size * 4),
            nn.Conv2d(input_channel // 4, input_channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 4, momentum=bn_momentum),
            nn.PReLU(input_channel // 4),
            HFAB(input_channel=input_channel // 4, input_size=input_size * 4)
        )
        # up_sample_3
        # self._up_sample_3 = nn.ConvTranspose2d(input_channel // 4, input_channel // 8,
        #                                        kernel_size=2,
        #                                        stride=2)
        self._up_sample_3 = nn.Sequential(
            nn.Conv2d(input_channel // 4, input_channel // 8, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_3
        self._up_layer_3 = nn.Sequential(
            nn.Conv2d(input_channel // 4, input_channel // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 8, momentum=bn_momentum),
            nn.PReLU(input_channel // 8),
            HFAB(input_channel=input_channel // 8, input_size=input_size * 8),
            nn.Conv2d(input_channel // 8, input_channel // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 8, momentum=bn_momentum),
            nn.PReLU(input_channel // 8),
            HFAB(input_channel=input_channel // 8, input_size=input_size * 8)
        )
        # up_sample_4
        #self._up_sample_4 = nn.ConvTranspose2d(input_channel // 8, input_channel // 16,
        #                                       kernel_size=2,
        #                                       stride=2)
        self._up_sample_4 = nn.Sequential(
            nn.Conv2d(input_channel // 8, input_channel // 16, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # up_layer_4
        self._up_layer_4 = nn.Sequential(
            nn.Conv2d(input_channel // 8, input_channel // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 16, momentum=bn_momentum),
            nn.PReLU(input_channel // 16),
            HFAB(input_channel=input_channel // 16, input_size=input_size * 16),
            nn.Conv2d(input_channel // 16, input_channel // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 16, momentum=bn_momentum),
            nn.PReLU(input_channel // 16),
            HFAB(input_channel=input_channel // 16, input_size=input_size * 16)
        )
        # out_layer
        self._out_layer = nn.Sequential(
            nn.Conv2d(input_channel // 16, input_channel // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 32, momentum=bn_momentum),
            nn.PReLU(input_channel // 32),
            nn.Conv2d(input_channel // 32, input_channel // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 32, momentum=bn_momentum),
            nn.PReLU(input_channel // 32),
            nn.Conv2d(input_channel // 32, num_classes, kernel_size=1, stride=1)
        )

        self.sm = nn.Softmax(dim=1)

    def forward(self, x, skip_1, skip_2, skip_3, skip_4):
        # up layer 1 & concat skip connection -1
        x = self._up_sample_1(x)
        x = torch.cat((x, skip_4), dim=1)
        x = self._up_layer_1(x)
        # up layer 2 & concat skip connection -2
        x = self._up_sample_2(x)
        x = torch.cat((x, skip_3), dim=1)
        x = self._up_layer_2(x)
        # up layer 3 & concat skip connection -3
        x = self._up_sample_3(x)
        x = torch.cat((x, skip_2), dim=1)
        x = self._up_layer_3(x)
        # up layer 4 & concat skip connection -4
        x = self._up_sample_4(x)
        x = torch.cat((x, skip_1), dim=1)
        x = self._up_layer_4(x)
        # out layer
        x = self._out_layer(x)
        return self.sm(x)


class HFANet(nn.Module):
    def __init__(self, input_channel, input_size, num_classes):
        super(HFANet, self).__init__()

        self.encoder = Encoder(input_channel=input_channel, input_size=input_size)
        self.decoder = Decoder(input_channel=1024, input_size=14, num_classes=num_classes)
        self.skip_connection_feature_fusion_1 = nn.Conv2d(64 * 2, 64,
                                                          kernel_size=1,
                                                          stride=1)
        self.skip_connection_feature_fusion_2 = nn.Conv2d(128 * 2, 128,
                                                          kernel_size=1,
                                                          stride=1)
        self.skip_connection_feature_fusion_3 = nn.Conv2d(256 * 2, 256,
                                                          kernel_size=1,
                                                          stride=1)
        self.skip_connection_feature_fusion_4 = nn.Conv2d(512 * 2, 512,
                                                          kernel_size=1,
                                                          stride=1)
        self.bottom_feature_fusion = nn.Conv2d(1024 * 2, 1024,
                                               kernel_size=1,
                                               stride=1)

    def forward(self, t1, t2):
        bottom_feature_1, skip_connect_1_1, skip_connect_1_2, skip_connect_1_3, skip_connect_1_4 = self.encoder(
            t1)
        bottom_feature_2, skip_connect_2_1, skip_connect_2_2, skip_connect_2_3, skip_connect_2_4 = self.encoder(
            t2)
        skip_connect_fusion_1 = torch.cat((skip_connect_1_1, skip_connect_2_1), dim=1)
        skip_connect_fusion_2 = torch.cat((skip_connect_1_2, skip_connect_2_2), dim=1)
        skip_connect_fusion_3 = torch.cat((skip_connect_1_3, skip_connect_2_3), dim=1)
        skip_connect_fusion_4 = torch.cat((skip_connect_1_4, skip_connect_2_4), dim=1)
        bottom_fusion = torch.cat((bottom_feature_1, bottom_feature_2), dim=1)
        skip_connect_final_1 = self.skip_connection_feature_fusion_1(skip_connect_fusion_1)
        skip_connect_final_2 = self.skip_connection_feature_fusion_2(skip_connect_fusion_2)
        skip_connect_final_3 = self.skip_connection_feature_fusion_3(skip_connect_fusion_3)
        skip_connect_final_4 = self.skip_connection_feature_fusion_4(skip_connect_fusion_4)
        bottom_final = self.bottom_feature_fusion(bottom_fusion)
        output = self.decoder(bottom_final,
                              skip_connect_final_1,
                              skip_connect_final_2,
                              skip_connect_final_3,
                              skip_connect_final_4)
        return output