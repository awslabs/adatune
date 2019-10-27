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

import numpy as np
import torch.optim as optim

from adatune.utils import *


class MuSGD(object):

    def __init__(self, optimizer, hyper_lr, grad_clipping, first_order, mu, alpha, device):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.momentum = self.optimizer.param_groups[0]['momentum']
        self.hyper_lr = hyper_lr
        self.hyper_lr_tensor = torch.tensor(self.lr, requires_grad=True, device=device)
        self.hyper_optim = optim.SGD([self.hyper_lr_tensor], lr=self.hyper_lr)
        self.grad_clipping = grad_clipping
        self.first_order = first_order
        self.device = device
        self.z_0 = None
        self.z_1 = None
        self.c = 0.0
        self.alpha = alpha
        self.mu = mu
        self.mu_mode = 'auto' if self.mu < 0.0 else 'manual'
        self.b = None
        self.state_init = False

    def flatten_state(self, net):
        return torch.cat([self.optimizer.state[v]['momentum_buffer'].view(-1) for v in net.parameters()])

    def clip_grad(self, net):
        if self.grad_clipping:
            for params in net:
                params.clamp_(-self.grad_clipping, self.grad_clipping)

    def compute_hg(self, net, first_grad):
        # SGD-momentum needs at least one update to initialize the momentum buffer
        if self.momentum > 0.0 and not self.state_init:
            self.state_init = True
            return

        self.clip_grad(first_grad)

        grad_flatten = torch.cat([g.view(-1) for g in first_grad]).requires_grad_(True)

        if self.first_order or self.z_0 is None:
            self.z_0 = torch.neg(grad_flatten)
            if self.momentum > 0.0:
                v_t = self.flatten_state(net)
                self.z_0 = torch.neg(self.momentum * v_t + grad_flatten)
                self.z_1 = torch.zeros_like(v_t)
        else:
            hvp = ag.grad(grad_flatten @ self.z_0, net.parameters())
            self.clip_grad(hvp)
            hvp_flatten = torch.cat([h.view(-1) for h in hvp])
            if self.momentum > 0.0:
                v_t = self.flatten_state(net)
                self.z_0 = self.mu * (self.z_0 - (self.lr * hvp_flatten + self.lr * self.momentum * self.z_1))
                self.z_0 = self.z_0 + torch.neg(self.momentum * v_t + grad_flatten)
                self.z_1 = self.mu * (hvp_flatten + self.momentum * self.z_1)
            else:
                self.z_0 = self.mu * (self.z_0 - self.lr * hvp_flatten)
                self.z_0 = self.z_0 + torch.neg(grad_flatten)

        self.z_0 = self.z_0.detach()
        if self.momentum > 0.0:
            self.z_1 = self.z_1.detach()

        self.b = grad_flatten.detach()

    def hyper_step(self, val_grad):

        if self.z_0 is None:
            return

        self.clip_grad(val_grad)
        val_grad_flatten = torch.cat([f.view(-1) for f in val_grad])

        mat_mul = val_grad_flatten @ self.z_0
        hyper_grad = mat_mul.item()

        self.hyper_lr_tensor.grad = torch.tensor(hyper_grad, device=self.device)
        self.hyper_optim.step()
        new_lr = self.hyper_lr_tensor.data.item()

        # Update LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Update hyper-LR
        for param_group in self.hyper_optim.param_groups:
            param_group['lr'] = np.max([param_group['lr'] + self.alpha * hyper_grad * new_lr, 0.0])

        # Update mu
        if self.mu_mode == 'auto':
            grad_mult = (val_grad_flatten @ self.b).item()
            q_norm = new_lr / grad_mult
            z = np.maximum(np.minimum(q_norm, 1.), 0.)
            self.c = self.c * np.sign(self.mu) + self.mu
            self.mu = np.power(z, 1 / (self.c + 1))
