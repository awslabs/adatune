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
from adatune.mu_adam import MuAdam
from adatune.mu_sgd import MuSGD
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
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--adapt-hyper-lr', action='store_true')
    parser.add_argument('--hyper-lr', type=float, default=1e-8)
    parser.add_argument('--hyper-hyper-lr', type=float, default=1e-6)
    parser.add_argument('--model-loc', type=str, default='./model.pt')
    parser.add_argument('--grad-clipping', type=float, default=100.0)
    parser.add_argument('--adapt-mu', action='store_true')
    parser.add_argument('--mu', type=float, default=0.99999)
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def train_rtho(network_name, dataset, num_epoch, batch_size, optim_name, lr, momentum, wd, adapt_hyper_lr, hyper_lr,
               hyper_hyper_lr, model_loc, grad_clipping, first_order, seed, adapt_mu, mu):
    torch.manual_seed(seed)

    # We are using cuda for training - no point trying out on CPU for ResNet
    device = torch.device("cuda")

    net = network(network_name, dataset)
    net.to(device).apply(init_weights)

    # assign argparse parameters
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_accuracy = 0.0
    cur_lr = lr
    timestep = 0

    train_data, test_data = data_loader(network, dataset, batch_size)

    if optim_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd, eps=1e-4)
        hyper_optim = MuAdam(optimizer, hyper_lr, adapt_hyper_lr, grad_clipping, first_order, mu, adapt_mu,
                             hyper_hyper_lr, device)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        hyper_optim = MuSGD(optimizer, hyper_lr, adapt_hyper_lr, grad_clipping, first_order, mu, adapt_mu,
                            hyper_hyper_lr, device)

    vg = ValidationGradient(test_data, nn.CrossEntropyLoss(), device)
    for epoch in range(num_epoch):
        train_correct = 0
        train_loss = 0

        for inputs, labels in train_data:
            net.train()
            timestep += 1

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            train_pred = outputs.argmax(1)
            train_correct += train_pred.eq(labels).sum().item()

            first_grad = ag.grad(loss, net.parameters(), create_graph=True, retain_graph=True)

            hyper_optim.compute_hg(net, first_grad)

            for params, gradients in zip(net.parameters(), first_grad):
                params.grad = gradients

            optimizer.step()
            hyper_optim.hyper_step(vg.val_grad(net))
            clear_grad(net)

        train_acc = 100.0 * (train_correct / len(train_data.dataset))
        val_loss, val_acc = compute_loss_accuracy(net, test_data, criterion, device)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(net.state_dict(), model_loc)

        print('train_accuracy at epoch :{} is : {}'.format(epoch, train_acc))
        print('val_accuracy at epoch :{} is : {}'.format(epoch, val_acc))
        print('best val_accuracy is : {}'.format(best_val_accuracy))

        cur_lr = 0.0
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate after epoch :{} is : {}'.format(epoch, cur_lr))


if __name__ == '__main__':
    args = cli_def().parse_args()
    print(args)

    if os.path.exists(args.model_loc):
        os.remove(args.model_loc)

    train_rtho(args.network, args.dataset, args.num_epoch, args.batch_size, args.optimizer, args.lr, args.momentum,
               args.wd, args.adapt_hyper_lr, args.hyper_lr, args.hyper_hyper_lr, args.model_loc, args.grad_clipping,
               args.first_order, args.seed, args.adapt_mu, args.mu)
