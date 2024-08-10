import numpy as np
import torch
import torch.nn as nn

LEAKY_RATE = 0.1

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation_fn=False, use_leaky=True,bn=False):
        super(Conv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bn = bn 
        self.activation_fn = activation_fn

        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        #self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        
        # self.use_bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        #if bn:
        #  self.use_bn = nn.BatchNorm1d(out_channels) 

        # self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)   
        #if activation_fn:    
        #  self.relu = nn.ReLU(inplace=True)
        
        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        )

    def forward(self, input):           # INPUT : B N 3
        
      input = input.permute(0,2,1)      # B 3 N 
        
      #input = self.conv(input)
        
      #if self.bn:
      #  input = self.use_bn(input)

      #if self.activation_fn:
      #  input = self.relu(input)

      input = self.composed_module(input)

      output = input.permute(0, 2, 1)    # B N 3 
        
      return output



class Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=[1, 1],activation_fn=True, bn=False):
        super(Conv2d,self).__init__()

        """ 2D convolution with non-linear operation.

        Args:
            inputs: 4-D tensor variable BxHxWxC
            num_output_channels: int
            kernel_size: a list of 2 ints
            scope: string
            stride: a list of 2 ints
            padding: 'SAME' or 'VALID'
            data_format: 'NHWC' or 'NCHW'
            use_xavier: bool, use xavier_initializer if true
            stddev: float, stddev for truncated_normal init
            weight_decay: float
            activation_fn: function
            bn: bool, whether to use batch norm
            bn_decay: float or float tensor variable in [0,1]
            is_training: bool Tensor variable

        Returns:
            Variable tensor
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_fn = activation_fn
        self.bn = bn

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride)

        if bn:
            self.use_bn = nn.BatchNorm2d(out_channels)
        
        if activation_fn:
            self.relu = nn.ReLU(inplace=True)

    def forward(self,input):

         
        input = input.permute(0,3,2,1)

        output = self.conv(input)

        if self.bn:
            output = self.use_bn(output)

        if self.activation_fn:
            output = self.relu(output)
        
        output = output.permute(0,3,2,1)    # B N 3 

        return output


