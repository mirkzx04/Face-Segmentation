import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        """
        Define a constructor DoubleConv, it takes care of doing Conv application, BatchNorm(conv) end ReLU(Batch(conv))

        Args:
            in_channels : Input channles
            out_channels : Output channles

            (channels indicates the image channels )
            kernel_size : Size of filter that we applicate to input
        """
        self.kernel_size = kernel_size

        #Defines the sequential of operations of the forward pass
        self.conv_op = nn.Sequential(
            # Firs conv + batch norm + relu operations 
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second conv + batch norm + relu operations
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_tensor):
        return self.conv_op(input_tensor)