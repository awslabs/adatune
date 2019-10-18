# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import torch
from torchvision import datasets, transforms

data_loc = './data'


def data_loader(network, dataset, batch_size):
    if dataset == 'mnist':
        return get_image_data_loader(dataset, batch_size)
    elif dataset == 'cifar_10':
        if network == 'mlp':
            raise Exception('MLP is currently only designed for grayscale images')
        return get_image_data_loader(dataset, batch_size)
    elif dataset == 'cifar_100':
        if network == 'mlp':
            raise Exception('MLP is currently only designed for grayscale images')
        return get_image_data_loader(dataset, batch_size)
    else:
        raise Exception('dataset is not supported')


def get_image_data_loader(dataset_name, batch_size):
    if dataset_name == 'mnist':
        return mnist_data_loader(batch_size)
    elif dataset_name == 'cifar_10':
        return cifar_10_data_loader(batch_size)
    elif dataset_name == 'cifar_100':
        return cifar_100_data_loader(batch_size)
    else:
        raise Exception('dataset is not supported')


def mnist_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_loc, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    test_dataset = datasets.MNIST(root=data_loc, train=False, transform=transform, )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def cifar_10_data_loader(batch_size):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def cifar_100_data_loader(batch_size):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    dataset = datasets.CIFAR100(root=data_loc, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    test_dataset = datasets.CIFAR100(root=data_loc, train=False, transform=transform_test, )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader
