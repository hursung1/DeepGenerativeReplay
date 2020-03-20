import torch
import torchvision
import numpy as np


class Generator_FC(torch.nn.Module):
    """
    Fully-Connected Generator
    """
    def __init__(self, input_node_size, output_shape, hidden_node_size=256,
                    hidden_node_num=3, normalize=True):

        """
        input_node_size: shape of latent vector
        output_shape: shape of output data, (# of channels, width, height)
        """
        super(Generator_FC, self).__init__()
        self.output_shape = output_shape
        output_node_size = output_shape[0] * output_shape[1] * output_shape[2]

        HiddenLayerModule = []
        for _ in range(hidden_node_num):
            HiddenLayerModule.append(torch.nn.Linear(hidden_node_size, hidden_node_size))
            if normalize:
                HiddenLayerModule.append(torch.nn.BatchNorm1d(hidden_node_size, 0.8))
            HiddenLayerModule.append(torch.nn.LeakyReLU(0.2))

        self.network = torch.nn.Sequential(
            
            torch.nn.Linear(input_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2),
            
            *HiddenLayerModule,
            
            torch.nn.Linear(hidden_node_size, output_node_size),
            torch.nn.Tanh()
            
        )
        
    def forward(self, x):
        num_data = x.shape[0]
        _x = x.view(num_data, -1)
        return self.network(_x)


class Generator_Conv(torch.nn.Module):
    """
    Generator Class for GAN
    """
    def __init__(self, input_node_size, output_shape, hidden_node_size=256,
                    hidden_node_num=3):
        """
        input_node_size: dimension of latent vector
        output_shape: dimension of output image
        """
        super(Generator_Conv, self).__init__()

        self.input_node_size = input_node_size
        self.output_shape = output_shape
        num_channels, width, _ = output_shape

        layer_channels = []
        if width <= 32:
            layer_channels.append(width//2)
            layer_channels.append(width//4)
            

        # HiddenLayerModule = []
        # for _ in range(hidden_node_num):
        #     HiddenLayerModule.append(torch.nn.ConvTranspose2d(

        #                             ))
        #     if normalize:
        #         HiddenLayerModule.append(torch.nn.BatchNorm2d(num_features=))

        #     HiddenLayerModule.append(torch.nn.ReLU())

        # self.network = torch.nn.Sequential(
        #     torch.nn.ConvTranspose2d(input_node_size, out_channels=, kernel_size=, stride=, padding=, bias=False),
        #     torch.nn.LeakyReLU(0.2),

        #     *HiddenLayerModule,

        #     torch.nn.ConvTranspose2d(input_node_size, out_channels=, kernel_size=, stride=, padding=, bias=False),
        #     torch.nn.Tanh(),
        # )

        conv2d_1 = torch.nn.ConvTranspose2d(in_channels=input_node_size,
                                   out_channels=width*8, 
                                   kernel_size=layer_channels[1], 
                                   stride=1,
                                   padding=0,
                                   bias=False)
        conv2d_2 = torch.nn.ConvTranspose2d(in_channels=width*8, 
                                   out_channels=width*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.ConvTranspose2d(in_channels=width*4, 
                                   out_channels=num_channels, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features = width*8),
            torch.nn.ReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features = width*4),
            torch.nn.ReLU(inplace=True),
            conv2d_3,
            torch.nn.Tanh()
        )

    def forward(self, x):
        _x = x.view(-1, self.input_node_size, 1, 1)
        return self.network(_x)


class Discriminator_FC(torch.nn.Module):
    """
    Fully-Connected Discriminator
    """
    def __init__(self, input_shape, hidden_node_size=256, output_node_size=1):
        super(Discriminator_FC, self).__init__()
        input_node_size = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Linear(hidden_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Linear(hidden_node_size, output_node_size),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        _x = x.view(x.shape[0], -1)
        return self.network(_x).view(-1, 1)


class Discriminator_Conv(torch.nn.Module):
    """
    Discriminator Class for GAN
    """
    def __init__(self, input_shape, hidden_node_size=256, output_node_size=1):
        """
        Parameters
        ----------
        input_shape: (C, W, H)

        """
        super(Discriminator_Conv, self).__init__()
        num_channels, width, _ = input_shape

        conv2d_1 = torch.nn.Conv2d(in_channels=num_channels, 
                                   out_channels=width*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_2 = torch.nn.Conv2d(in_channels=width*4, 
                                   out_channels=width*8, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.Conv2d(in_channels=width*8, 
                                   out_channels=output_node_size, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=width*4),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=width*8),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_3,
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)

    
class Solver(torch.nn.Module):
    """
    Solver Class for Deep Generative Replay
    """
    def __init__(self, T_n):
        super(Solver, self).__init__()
        fc1 = torch.nn.Linear(28*28, 100)
        fc2 = torch.nn.Linear(100, 100)
        fc3 = torch.nn.Linear(100, T_n * 2)
        self.network = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.ReLU(),
            fc3
        )

    def forward(self, x):
        return self.network(x.view(x.shape[0], -1))
