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

import math

import numpy as np
import torch.optim as optim

from adatune.utils import *


class MuAdam(object):

    def __init__(self, optimizer, hyper_lr, grad_clipping, first_order, mu, alpha, device):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.beta1 = self.optimizer.param_groups[0]['betas'][0]
        self.beta2 = self.optimizer.param_groups[0]['betas'][1]
        self.eps = self.optimizer.param_groups[0]['eps']
        self.hyper_lr = hyper_lr
        self.hyper_lr_tensor = torch.tensor(self.lr, requires_grad=True, device=device)
        self.hyper_optim = optim.SGD([self.hyper_lr_tensor], lr=self.hyper_lr)
        self.grad_clipping = grad_clipping
        self.first_order = first_order
        self.device = device
        self.mu = mu
        self.mu_mode = 'auto' if self.mu < 0.0 else 'manual'
        self.alpha = alpha
        self.z_0 = None
        self.z_1 = None
        self.z_2 = None
        self.step = 0
        self.b = None
        self.c = 0.0
        self.state_init = False

    def flatten_state(self, net):
        return (torch.cat([self.optimizer.state[v]['exp_avg'].view(-1) for v in net.parameters()]),
                torch.cat([self.optimizer.state[v]['exp_avg_sq'].view(-1) for v in net.parameters()]))

    def clip_grad(self, net):
        if self.grad_clipping:
            for params in net:
                params.clamp_(-self.grad_clipping, self.grad_clipping)

    def compute_hg(self, net, first_grad):
        # Adam needs at least one update to initialize the gradient and sqauared-gradient buffers
        if not self.state_init:
            self.state_init = True
            self.step += 1
            return

        self.clip_grad(first_grad)
        grad_flatten = torch.cat([g.view(-1) for g in first_grad]).requires_grad_(True)

        coeff = (math.sqrt(1.0 - self.beta2 ** self.step)) / (1.0 - self.beta1 ** self.step)

        if self.first_order or self.z_2 is None:
            m_t, v_t = self.flatten_state(net)
            self.z_0 = torch.zeros_like(grad_flatten)
            self.z_1 = torch.zeros_like(grad_flatten)
            self.z_2 = torch.neg(coeff * (m_t / torch.sqrt(v_t + self.eps)))
        else:
            hvp = ag.grad(grad_flatten @ self.z_2, net.parameters())
            self.clip_grad(hvp)
            grad_flatten = grad_flatten.detach()
            hvp_flatten = torch.cat([h.view(-1) for h in hvp])

            m_t, v_t = self.flatten_state(net)

            a_31 = -self.lr * coeff * self.beta1 * torch.reciprocal(torch.sqrt(v_t + self.eps))
            a_32 = self.lr * coeff * 0.5 * self.beta2 * (m_t / torch.pow(v_t + self.eps, 1.5))
            a_33_inner_1 = (1.0 - self.beta1) * torch.reciprocal(torch.sqrt(v_t + self.eps))
            a_33_inner_2 = (1.0 - self.beta2) * ((m_t * grad_flatten) / torch.pow(v_t + self.eps, 1.5))
            a_33 = (1.0 - self.lr * coeff) * (a_33_inner_1 - a_33_inner_2) * hvp_flatten

            self.z_2 = self.mu * (a_31 * self.z_0 + a_32 * self.z_1 + a_33)
            self.z_2 = self.z_2 + torch.neg(coeff * (m_t / torch.sqrt(v_t + self.eps)))

            self.z_0 = self.mu * (self.beta1 * self.z_0 + (1.0 - self.beta1) * hvp_flatten)
            self.z_1 = self.mu * (self.beta2 * self.z_1 + 2.0 * (1.0 - self.beta2) * grad_flatten * hvp_flatten)

        self.step += 1

        self.z_0 = self.z_0.detach()
        self.z_1 = self.z_1.detach()
        self.z_2 = self.z_2.detach()

        self.b = grad_flatten.detach()

    def hyper_step(self, val_grad):
        if self.z_2 is None:
            return

        self.clip_grad(val_grad)
        val_grad_flatten = torch.cat([f.view(-1) for f in val_grad])

        mat_mul = val_grad_flatten @ self.z_2
        hyper_grad = mat_mul.item()

        self.hyper_lr_tensor.grad = torch.tensor(hyper_grad, device=self.device)
        self.hyper_optim.step()
        new_lr = self.hyper_lr_tensor.data.item()

        # Update LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # # Update hyper-LR
        for param_group in self.hyper_optim.param_groups:
            param_group['lr'] = np.max([param_group['lr'] + self.alpha * hyper_grad * new_lr, 0.0])

        # Update mu
        if self.mu_mode == 'auto':
            grad_mult = (val_grad_flatten @ self.b).item()
            q_norm = new_lr / grad_mult
            z = np.maximum(np.minimum(q_norm, 1.), 0.)
            self.c = self.c * np.sign(self.mu) + self.mu
            self.mu = np.power(z, 1. / (self.c + 1.))
