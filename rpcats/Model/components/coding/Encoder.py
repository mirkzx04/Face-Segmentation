import torch
import torch.nn as nn

from Model.components.DoubleConv import DoubleConv

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size):
        super().__init__()
        
        """ 
        Defines a constructor Encoder, it does all operation of compression from input to bottleneck

        Args : 
            in_channels : Input channel
            out_channel : output channel
            conv_kernel_size : Kernel size of the convolution
            pool_kernel_size : Kernel size of max pooling
        """

        # Defines a DoubleConv class, it does che conv + batch norm + relu two times
        self.double_conv = DoubleConv(
            in_channels= in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size 
        )

        # Defines max pooling operation, it extracts the more important feature from feature map of the double conv operations
        self.pool = nn.MaxPool2d(kernel_size= pool_kernel_size, stride=2)

    def forward(self, input_tensor):
        down_conv = self.double_conv(input_tensor)
        p = self.pool(down_conv)

        return p, down_conv