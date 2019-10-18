#!/usr/bin/env python

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

import argparse
import os

import torch.optim as optim

from adatune.data_loader import *
from adatune.hd_adam import AdamHD
from adatune.hd_sgd import SGDHD
from adatune.network import *
from adatune.utils import *


def cli_def():
    parser = argparse.ArgumentParser(description='CLI for running automated Learning Rate scheduler methods')
    parser.add_argument('--network', type=str, default='vgg', choices=['resnet', 'vgg'])
    parser.add_argument('--dataset', type=str, choices=['cifar_10', 'cifar_100'], default='cifar_10')
    parser.add_argument('--num-epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--lr-scheduler', type=str, default='hd', choices=['hd', 'ed', 'cyclic', 'staircase'])
    parser.add_argument('--hyper-lr', type=float, default=1e-8, help='beta, only applicable for HD')
    parser.add_argument('--step-size', type=int, default=30, help='step-size, applicable for staircase')
    parser.add_argument('--lr-decay', type=float, default=1.0, help='Decay factor, applicable for staircase and ED')
    parser.add_argument('--t-0', type=int, default=10, help='T_0 for Cosine Annealing with Restarts')
    parser.add_argument('--t-mult', type=int, default=2, help='T_Mult for Cosine Annealing with Restarts')
    parser.add_argument('--model-loc', type=str, default='./model.pt')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def train_baselines(network_name, dataset, num_epoch, batch_size, optim_name, lr, momentum, wd, lr_scheduler_type,
                    hyper_lr, step_size, lr_decay, t_0, t_mult, model_loc, seed):
    torch.manual_seed(seed)

    # We are using cuda for training - no point trying out on CPU for ResNet
    device = torch.device("cuda")

    net = network(network_name, dataset)
    net.to(device).apply(init_weights)

    # assign argparse parameters
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_accuracy = 0.0
    timestep = 0
    cur_lr = lr
    train_data, test_data = data_loader(network, dataset, batch_size)
    lr_scheduler = None

    if lr_scheduler_type == 'hd':
        if optim_name == 'adam':
            optimizer = AdamHD(net.parameters(), lr=lr, weight_decay=wd, eps=1e-4, hypergrad_lr=hyper_lr)
        else:
            optimizer = SGDHD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd, hypergrad_lr=hyper_lr)
    else:
        if optim_name == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd, eps=1e-4)
        else:
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        if lr_scheduler_type == 'ed':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        elif lr_scheduler_type == 'staircase':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        elif lr_scheduler_type == 'cyclic':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult,
                                                                                eta_min=lr * 1e-4)

    for epoch in range(num_epoch):
        train_correct = 0
        train_loss = 0

        iter_len = len(train_data)

        for i, (inputs, labels) in enumerate(train_data):
            net.train()
            timestep += 1

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            train_pred = outputs.argmax(1)
            train_correct += train_pred.eq(labels).sum().item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler and lr_scheduler_type == 'cyclic':
                lr_scheduler.step(epoch + (i / iter_len))

        train_acc = 100.0 * (train_correct / len(train_data.dataset))
        val_loss, val_acc = compute_loss_accuracy(net, test_data, criterion, device)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(net.state_dict(), model_loc)

        if lr_scheduler and lr_scheduler_type != 'cyclic':
            lr_scheduler.step(epoch)

        print('train_accuracy at epoch :{} is : {}'.format(epoch, train_acc))
        print('val_accuracy at epoch :{} is : {}'.format(epoch, val_acc))
        print('best val_accuracy is : {}'.format(best_val_accuracy))


if __name__ == '__main__':
    args = cli_def().parse_args()
    print(args)

    if os.path.exists(args.model_loc):
        os.remove(args.model_loc)

    train_baselines(args.network, args.dataset, args.num_epoch, args.batch_size, args.optimizer, args.lr, args.momentum,
                    args.wd, args.lr_scheduler, args.hyper_lr, args.step_size, args.lr_decay, args.t_0, args.t_mult,
                    args.model_loc, args.seed)
