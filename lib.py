import torch
import torchvision
import numpy as np

class PermutedMNISTDataLoader(torchvision.datasets.MNIST):
    """
    Make randomly permuted MNIST Data.
    """
    def __init__(self, source='data/mnist_data', train = True, shuffle_seed = None):
        super(PermutedMNISTDataLoader, self).__init__(source, train, download=True)
        
        self.train = train
        self.num_data = 0
        
        if self.train:
            self.permuted_train_data = torch.stack(
                [img.type(dtype=torch.float32).view(-1)[shuffle_seed] / 255.0
                    for img in self.train_data])
            self.num_data = self.permuted_train_data.shape[0]
            
        else:
            self.permuted_test_data = torch.stack(
                [img.type(dtype=torch.float32).view(-1)[shuffle_seed] / 255.0
                    for img in self.test_data])
            self.num_data = self.permuted_test_data.shape[0]
            
            
    def __getitem__(self, index):
        
        if self.train:
            input, label = self.permuted_train_data[index], self.train_labels[index]
        else:
            input, label = self.permuted_test_data[index], self.test_labels[index]
        
        return input, label

    
    def getNumData(self):
        return self.num_data


def permute_mnist(num_task, batch_size):
    """
    Returns PermutedMNISTDataLoaders

    Parameters
    ------------
    num_task: number of tasks\n
    batch_size: size of minibatch

    Returns
    ------------
    num_task numbers of TrainDataLoader, TestDataLoader

    """
    train_loader = {}
    test_loader = {}
    
    train_data_num = 0
    test_data_num = 0
    
    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        np.random.shuffle(shuffle_seed)
        
        train_PMNIST_DataLoader = PermutedMNISTDataLoader(train=True, shuffle_seed=shuffle_seed)
        test_PMNIST_DataLoader = PermutedMNISTDataLoader(train=False, shuffle_seed=shuffle_seed)
        
        train_data_num += train_PMNIST_DataLoader.getNumData()
        test_data_num += test_PMNIST_DataLoader.getNumData()
        
        train_loader[i] = torch.utils.data.DataLoader(
                train_PMNIST_DataLoader,
                batch_size=batch_size)
        
        test_loader[i] = torch.utils.data.DataLoader(
                test_PMNIST_DataLoader,
                batch_size=batch_size)
    
    return train_loader, test_loader, int(train_data_num/num_task), int(test_data_num/num_task)

class Generator(torch.nn.Module):
    """
    Generator Class for GAN
    """
    def __init__(self):
        super(Generator, self).__init__()
        conv2d_1 = torch.nn.ConvTranspose2d(in_channels=num_noise,
                                   out_channels=28*8, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)
        conv2d_2 = torch.nn.ConvTranspose2d(in_channels=28*8, 
                                   out_channels=28*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.ConvTranspose2d(in_channels=28*4, 
                                   out_channels=1, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features = 28*8),
            torch.nn.ReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features = 28*4),
            torch.nn.ReLU(inplace=True),
            conv2d_3,
            torch.nn.Tanh()
        )

        if torch.cuda.is_available():
            self.network = self.network.cuda()

    def forward(self, x):
        return self.network(x.view(-1, num_noise, 1, 1))


class Discriminator(torch.nn.Module):
    """
    Discriminator Class for GAN
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        conv2d_1 = torch.nn.Conv2d(in_channels=1, 
                                   out_channels=28*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_2 = torch.nn.Conv2d(in_channels=28*4, 
                                   out_channels=28*8, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.Conv2d(in_channels=28*8, 
                                   out_channels=1, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=28*4),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=28*8),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_3,
            torch.nn.Sigmoid()
        )

        if torch.cuda.is_available():
            self.network = self.network.cuda()

    def forward(self, x):
        return self.network(x).view(-1, 1)

class Solver(torch.nn.Module):
    """
    Solver Class for Deep Generative Replay
    """
    def __init__(self):
        super(Solver, self).__init__()
        fc1 = torch.nn.Linear(28*28, 100)
        fc2 = torch.nn.Linear(100, 100)
        fc3 = torch.nn.Linear(100, 100)
        self.network = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.ReLU(),
            fc3
        )

        if torch.cuda.is_available():
            self.network = self.network.cuda()

    def forward(self, x):
        return self.network(x)