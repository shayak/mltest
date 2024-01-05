import torch
import torchvision


def torch_trainloader(path, transform, batch_size=4, num_workers=0, shuffle=True):
    dataset = torchvision.datasets.CIFAR10(root=path, train=True,
                                           download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader


def torch_testloader(path, transform, batch_size=4, num_workers=0, shuffle=False):
    dataset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader
