from Model.components.coding.Encoder import Encoder
from Model.components.decoding.Decoder import Decoder
from Model.components.DoubleConv import DoubleConv

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, 
            layer_num, 
            in_channels,
            num_classes, 
            conv_kernel_size, 
            convT_kernel_size,
            pool_kernel_size
        ):
        super().__init__()
        """
        Defines a constructor U-Net

        Args : 
            layer_num -> Layer Number it decrees the model depth
            in_channels -> input channels
            num_classes -> number of classes to predict
            conv_kernel_size -> kernel size of the convolution operations
            convT_kernel_size -> kernel size of the tranpose convolution operations
            pool_kernel_size -> Kernel size of the pool operations 
        """

        #Initialize enc_layers -> all encoder layers
        self.enc_layers, last_enc_channels = self.initialize_enc_layers(
            layer_num, 
            in_channels, 
            conv_kernel_size, 
            pool_kernel_size
        )

        #Initialize the bottle neck (mid part of U-Net)
        self.bottle_neck = DoubleConv(last_enc_channels, last_enc_channels * 2, conv_kernel_size)

        #Initialize dec_layers -> all decoder layers
        self.dec_layers = self.initialize_dec_layers(
            layer_num, 
            conv_kernel_size, 
            convT_kernel_size,
            last_enc_channels
        )

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1
        )

    def initialize_dec_layers(
            self,
            layer_num, 
            conv_kernel_size, 
            convT_kernel_size,
            last_enc_channels
        ):

        # Set in_channels
        in_channels = last_enc_channels * 2
        dec_layers = nn.ModuleList()

        for l in range(layer_num):
            dec_layer = Decoder(
                in_channles=in_channels,
                conv_kernel_size=conv_kernel_size,
                convT_kernel_size=convT_kernel_size,
            )       

            dec_layers.append(dec_layer)  

            # Update in_channels for next layer
            in_channels //= 2
        
        return dec_layers
    
    def initialize_enc_layers(
            self,
            layer_num, 
            in_channels, 
            conv_kernel_size, 
            pool_kernel_size
        ):

        # Set out channels
        out_channels = 64
        enc_layers = nn.ModuleList()

        for l in range(layer_num):            
            enc_layer_conv = Encoder(
                in_channels = in_channels,
                out_channels=out_channels,
                conv_kernel_size=conv_kernel_size,
                pool_kernel_size=pool_kernel_size
            )

            enc_layers.append(enc_layer_conv)

            in_channels = out_channels

            # Update out_channels for next layer
            out_channels *= 2

        return enc_layers, out_channels // 2
    
    def forward(self, X):
        """
        This method defines the forward pass
        """
        skip_conn = []

        # Execute all encoder layers, get a conv output (enc_cov) and pool 
        # output (pool)
        for encoder in self.enc_layers:
            X, skip = encoder(X)

            skip_conn.append(skip)
        
        # Get bottle neck output
        bottle_neck = self.bottle_neck(X)

        X = bottle_neck
        skip_conn = skip_conn[::-1]

        # Execute all decoder layer
        for idx, decoder in enumerate(self.dec_layers):
            X = decoder(X, skip_conn[idx])
        
        output = self.out(X)

        return output