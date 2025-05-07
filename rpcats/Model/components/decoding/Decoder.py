import torch
import torch.nn as nn

from Model.components.DoubleConv import DoubleConv

class Decoder(nn.Module):
    def __init__(self, in_channles, out_channels, conv_kernel_size, convT_kernel_size):
        super().__init__()

        """
        Defines a constructor Decoder, it does all operation of decoding of the bottleneck output

        Args : 
            in_channles : Iput channel
            out_channels : Output channel
            conv_kernel_size : kernel size of the convolution 
            convT_kernel_size : Kernel size of the Convolution Transpose
        """

        #Defines a ConvTranspose (contrary to the convolution)
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=in_channles,
            out_channels= in_channles // 2,
            kernel_size=convT_kernel_size,
            stride=2)
        
        # Defines a DoubleConv class, it does che conv + batch norm + relu two times
        self.double_conv = DoubleConv(in_channles, out_channels, conv_kernel_size)

    def forward(self, input_tensor, skip_conn):
        """
        Defines a forward pass

        Args:
            input_tensor : output from last Decoder Layer
            skip_conn : Output from the encoder layer parallel to this 
        """

        up = self.transpose_conv(input_tensor)
        
        # up form : [batch_size, C, height, widtg] and skip_conn form : [batch_size, C', height, width] with torch.cat we
        #concat 2 tensor on axis = 1 getting one tensor with form [batch_size, C * C', height, widtg]
        return self.double_conv(torch.cat(([up, skip_conn]), 1))
    