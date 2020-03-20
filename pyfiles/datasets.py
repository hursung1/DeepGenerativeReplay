import torch
import torchvision
import numpy as np

class MNIST_PermutedDataset(torchvision.datasets.MNIST):
    """
    Permuted MNIST Dataset for Domain Learning
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

    
class MNIST_IncrementalDataset(torchvision.datasets.MNIST):
    """
    MNIST Dataset for Incremental Learning
    """
    def __init__(self, 
                 source='./mnist_data', 
                 train=True,
                 transform=None,
                 download=False,
                 classes=range(10)):
        
        super(MNIST_IncrementalDataset, self).__init__(source, 
                                                       train, 
                                                       transform, 
                                                       download=True)
        self.train = train
        self.transform = transform

        if train:
            train_data = []
            train_labels = []
            for i in range(len(self.train_data)):
                if self.train_labels[i] in classes:
                    _data = transform(self.train_data[i].numpy()) if transform is not None else self.train_data[i]
                    # train_data.append(self.train_data[i].type(dtype=torch.float32))
                    train_data.append(_data)
                    train_labels.append(self.train_labels[i])
            
            self.TrainData = train_data
            self.TrainLabels = train_labels

        else:
            test_data = []
            test_labels = []
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    _data = transform(self.test_data[i].numpy()) if transform is not None else self.test_data[i]
                    # test_data.append(self.test_data[i].type(dtype=torch.float32))
                    test_data.append(_data)
                    test_labels.append(self.test_labels[i])
            
            self.TestData = test_data
            self.TestLabels = test_labels

    def __getitem__(self, index):
        if self.train:
            return self.TrainData[index], self.TrainLabels[index]
        else:
            return self.TestData[index], self.TestLabels[index]

    def __len__(self):
        if self.train:
            return len(self.TrainLabels)
        else:
            return len(self.TestLabels)