from typing import Callable, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine
from scipy.stats.qmc import Halton
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm

# using log_prob to compute the gradient
# class Linear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=None,
#                  init_std: float = 1e-0) -> None:
#         super().__init__(in_features, out_features, bias, dtype)
#         self.log_noise_std = nn.Parameter(torch.full((out_features,), np.log(init_std)))

#     def forward(self, input, add_noise=False):
#         self.sum_log_prob = torch.zeros(input.shape[0]).to(self.log_noise_std.device)
#         logit_output = super().forward(input)

#         if add_noise:
#             dist = Normal(loc=logit_output, scale=torch.exp(self.log_noise_std))
#             logit_output = dist.sample()
#             log_prob = dist.log_prob(logit_output)
#             self.sum_log_prob += torch.sum(log_prob, dim=tuple(range(1, log_prob.dim())))

#         return logit_output

#     def backward(self, loss):
#         loss_new = (loss.detach() * self.sum_log_prob)
#         torch.mean(loss_new).backward()

# adding noise to the logits to compute the gradient
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, init_std=1e-3, bias=True, device=None, dtype=None, anti_variable=True):
        """
        weight: (out_features, in_features)
        bias: (out_features,)
        input_buf: (bs, in_features)
        epsilon_buf: (bs, out_features)
        noise_std: (out_features,)
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_features,), np.log(init_std), device=device))
        self.input_buf = None
        self.epsilon_buf = None
        self.anti_variable= anti_variable

    def forward(self, input, add_noise=False):
        """
        input: (bs, in_features)
        logit_output: (bs, out_features)
        """
        logit_output = super().forward(input)
        shape_o = logit_output.shape
        logit_output = logit_output.reshape((-1, shape_o[-1]))
        if add_noise:
            bs, out_features = logit_output.shape
            epsilon = torch.zeros_like(logit_output, device=self.log_noise_std.device)

            if self.anti_variable:
                epsilon[:bs//2] += torch.randn((bs//2, out_features), device=self.log_noise_std.device)
                epsilon[bs//2:] -= epsilon[:bs//2]
            else:
                epsilon += torch.randn((bs, out_features), device=self.log_noise_std.device)

            noise = epsilon * torch.exp(self.log_noise_std)
            self.input_buf = input
            self.epsilon_buf = epsilon
            return (logit_output + noise).reshape(shape_o)
        else:
            return logit_output.reshape(shape_o)

    def backward(self, loss):
        """
        loss: (bs,)
        """

        shape_i = self.input_buf.shape
        shape_e = self.epsilon_buf.shape
        is_seq = True if len(shape_i) == 3 else False
        self.input_buf = self.input_buf.reshape((-1, shape_i[-1]))
        self.epsilon_buf = self.epsilon_buf.reshape((-1, shape_e[-1]))
        batch_size = self.input_buf.shape[0]
        if is_seq:
            loss = loss.unsqueeze(-1).repeat_interleave(shape_i[1], dim=0)
        else:
            loss = loss.unsqueeze(-1)

        noise_std = torch.exp(torch.unsqueeze(self.log_noise_std,-1))
        self.weight.grad = torch.einsum('ni,nj->ji', self.input_buf * loss, self.epsilon_buf) / (noise_std * batch_size)
        self.bias.grad = torch.einsum('ni,nj->j', loss, self.epsilon_buf) / (torch.exp(self.log_noise_std) * batch_size)
        self.log_noise_std.grad = torch.einsum('ni,nj->j', loss, self.epsilon_buf ** 2 - 1) / batch_size

        self.input_buf = None
        self.epsilon_buf = None


# 同一个数据点上加的噪声应该一样的，在三维的情形下有bug

class Linear_(nn.Linear):
    def __init__(self, in_features, out_features, init_std=0.1, bias=True, device=None, dtype=None, qmc_method=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_features,), np.log(init_std), device=device))
        self.epsilon_buf = None
        self.epsilon_buf_b = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = SobolEngine(2 * (self.weight.data.numel()+self.bias.data.numel()))
        elif self.qmc_method == 'halton':
            self.halton_engine = Halton(2 * (self.weight.data.numel()+self.bias.data.numel()))

    def forward(self, input, add_noise=False):
        if add_noise:
            bs = input.shape[0]
            w = self.weight.unsqueeze(0).repeat(bs,1,1)
            b = self.bias.unsqueeze(0).repeat(bs,1)
            epsilon_w = torch.zeros_like(w, device=self.log_noise_std.device)
            epsilon_b = torch.zeros_like(b, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                num_w, num_b = self.weight.data.numel(), self.bias.data.numel()
                if self.qmc_method == 'sobol':
                    samples = self.sobol_engine.draw(bs//2)
                else:
                    samples = torch.from_numpy(self.halton_engine.random(bs//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :num_w + num_b] + 1e-12)) \
                                 * torch.cos(2 * np.pi * samples[:, num_w + num_b:])
                epsilon_w[:bs // 2] += normal_samples[:, :num_w].reshape(bs // 2, self.out_features, self.in_features).to(self.log_noise_std.device)
                epsilon_b[:bs // 2] += normal_samples[:, num_w:].reshape(bs // 2, self.out_features).to(self.log_noise_std.device)
            else:
                epsilon_w[:bs // 2] += torch.randn((bs // 2, self.out_features, self.in_features), device=self.log_noise_std.device)
                epsilon_b[:bs // 2] += torch.randn((bs // 2, self.out_features), device=self.log_noise_std.device)

            epsilon_w[bs // 2:] -= epsilon_w[:bs // 2]
            epsilon_b[bs // 2:] -= epsilon_b[:bs // 2]
            self.epsilon_buf = epsilon_w
            self.epsilon_buf_b = epsilon_b
            
            w += epsilon_w * torch.exp(self.log_noise_std[None, :, None]) # memory increase when execute this line
            b += epsilon_b * torch.exp(self.log_noise_std[None, :])

            logit_output = torch.bmm(w,input[:,:,None]).squeeze(-1) + b
            del epsilon_b, epsilon_w, w, b
            torch.cuda.empty_cache()
        else:
            logit_output = super().forward(input)
        return logit_output

    def backward(self, loss):
        bs = loss.shape[0]
        tmp = loss[:, None, None] * self.epsilon_buf
        self.weight.grad = torch.sum(tmp, 0) / (bs * torch.exp(self.log_noise_std[:, None]))

        tmp = loss[:, None] * self.epsilon_buf_b
        self.bias.grad = torch.sum(tmp, 0) / (bs * torch.exp(self.log_noise_std))

        tmp = torch.sum((self.epsilon_buf ** 2) - 1, dim=2) + ((self.epsilon_buf_b ** 2) - 1)
        self.log_noise_std.grad = torch.sum(tmp * loss[:, None], 0) / bs

        self.epsilon_buf = None
        self.epsilon_buf_b = None

# # adding noise to the weight to compute the gradient
# # modified
# class Linear_(nn.Linear):
#     def __init__(self, in_features, out_features, init_std=1e-3, bias=True, device=None, dtype=None, anti_variable=True):
#         super().__init__(in_features, out_features, bias, device, dtype)
#         self.log_noise_std = nn.Parameter(torch.full((out_features,), np.log(init_std), device=device))
#         self.epsilon_buf = None
#         self.epsilon_buf_b = None
#         self.anti_variable = anti_variable

#     def forward(self, input, add_noise=False):
#         if add_noise:
#             shape_o = input.shape
#             bs = shape_o[0]
#             w = self.weight.unsqueeze(0).repeat(bs,1,1)
#             b = self.bias.unsqueeze(0).repeat(bs,1)
#             epsilon_w = torch.zeros_like(w, device=self.log_noise_std.device)
#             epsilon_b = torch.zeros_like(b, device=self.log_noise_std.device)

#             if self.anti_variable:
#                 epsilon_w[:bs // 2] += torch.randn((bs // 2, self.out_features, self.in_features), device=self.log_noise_std.device)
#                 epsilon_b[:bs // 2] += torch.randn((bs // 2, self.out_features), device=self.log_noise_std.device)
#                 epsilon_w[bs // 2:] -= epsilon_w[:bs // 2]
#                 epsilon_b[bs // 2:] -= epsilon_b[:bs // 2]

#             else:
#                 epsilon_w += torch.randn((bs, self.out_features, self.in_features), device=self.log_noise_std.device)
#                 epsilon_b += torch.randn((bs, self.out_features), device=self.log_noise_std.device)
#             self.epsilon_buf = epsilon_w
#             self.epsilon_buf_b = epsilon_b
#             w += epsilon_w * torch.exp(self.log_noise_std[None, :, None])
#             b += epsilon_b * torch.exp(self.log_noise_std[None, :])

#             # print(input.shape)
#             # print(w.shape)
#             # print(b.shape)
#             input = input.reshape((-1, shape_o[-1]))
#             seq_len = input.shape[0] // bs
#             if seq_len > 1:
#                 w = w.repeat_interleave(seq_len, dim=0)
#                 b = b.repeat_interleave(seq_len, dim=0)
#             # print('---')
#             # print(input.shape)
#             # print(w.shape)
#             # print(b.shape)
#             shape_o = list(shape_o)
#             shape_o[-1] = self.out_features
#             logit_output = torch.bmm(w,input[:,:,None]).squeeze(-1) + b
#             logit_output = logit_output.reshape(shape_o)
#             # print(logit_output.shape)
#         else:
#             logit_output = super().forward(input)
#         return logit_output

#     def backward(self, loss):
#         bs = loss.shape[0]
#         tmp = loss[:, None, None] * self.epsilon_buf
#         self.weight.grad = torch.sum(tmp, 0) / (bs * torch.exp(self.log_noise_std[:, None]))

#         tmp = loss[:, None] * self.epsilon_buf_b
#         self.bias.grad = torch.sum(tmp, 0) / (bs * torch.exp(self.log_noise_std))

#         tmp = torch.sum((self.epsilon_buf ** 2) - 1, dim=2) + ((self.epsilon_buf_b ** 2) - 1)
#         self.log_noise_std.grad = torch.sum(tmp * loss[:, None], 0) / bs

#         self.epsilon_buf = None
#         self.epsilon_buf_b = None


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 bias=True, device=None, dtype=None, init_std=1e-3, qmc_method=None):
        """
        weight: (out_channels, in_channels, H, W)
        bias: (out_channels,)
        input_buf: (N, in_channels, H, W)
        epsilon_buf: (N, out_channels, H_, W_)
        noise_std: (out_channels,)
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         bias=bias, device=device, dtype=dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_channels,), np.log(init_std), device=device))
        self.input_buf = None
        self.epsilon_buf = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = None
        elif self.qmc_method == 'halton':
            self.halton_engine = None

    def forward(self, input, add_noise=False):
        """
        input: (N, in_channels, H, W)
        logit_output: (N, out_channels, H_, W_)
        """
        logit_output = super().forward(input)
        if add_noise:
            N, out_channels, H_, W_ = logit_output.shape
            epsilon = torch.zeros_like(logit_output, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                if self.qmc_method == 'sobol':
                    if self.sobol_engine is None:
                        self.sobol_engine = SobolEngine(2 * out_channels * H_ * W_)
                    samples = self.sobol_engine.draw(N//2)
                else:
                    if self.halton_engine is None:
                        self.halton_engine = Halton(2 * out_channels * H_ * W_)
                    samples = torch.from_numpy(self.halton_engine.random(N//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :out_channels * H_ * W_] + 1e-8)) \
                                 * torch.cos(2 * np.pi * samples[:, out_channels * H_ * W_:])
                epsilon[:N//2] += normal_samples.view(N // 2, out_channels, H_, W_).to(self.log_noise_std.device)
            else:
                epsilon[:N//2] += torch.randn((N//2, out_channels, H_, W_), device=self.log_noise_std.device)
            epsilon[N//2:] -= epsilon[:N//2]
            noise = epsilon * torch.exp(self.log_noise_std[None,:,None,None])
            self.input_buf = input
            self.epsilon_buf = epsilon
            # del epsilon, input
            # import gc
            # gc.collect()
            return logit_output + noise
        else:
            return logit_output

    def backward(self, loss):
        """
        loss: (N,)
        """
        N, C_in, H_in, W_in = self.input_buf.shape
        _, C_out, H_out, W_out = self.epsilon_buf.shape

        # weight
        tmp = self.input_buf * loss[:,None,None,None]
        tmp = tmp.transpose(0,1)
        epsilon_buf = self.epsilon_buf.transpose(0,1)
        grad = F.conv2d(tmp,epsilon_buf,torch.zeros(size=(C_out,),device=self.log_noise_std.device),
                        stride=self.stride, padding=self.padding)
        self.weight.grad = grad.transpose(0,1) / (N * torch.exp(self.log_noise_std[:,None,None,None]))

        if self.bias is not None:
            # bias
            tmp = torch.sum(self.epsilon_buf,(2,3)) * loss[:,None]
            self.bias.grad = torch.sum(tmp, 0) / (N * torch.exp(self.log_noise_std))

        # noise std
        tmp = torch.sum(self.epsilon_buf**2 - 1, (2,3)) * loss[:,None]
        self.log_noise_std.grad = torch.sum(tmp, 0) / N

        self.input_buf = None
        self.epsilon_buf = None


class Conv2d_(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, init_std=1e-3,
                 bias=True, device=None, dtype=None, qmc_method=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         bias=bias, device=device, dtype=dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_channels,), np.log(init_std), device=device))
        self.epsilon_buf = None
        self.epsilon_buf_b = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = SobolEngine(2 * (self.weight.data.numel()+self.bias.data.numel()))
        elif self.qmc_method == 'halton':
            self.halton_engine = Halton(2 * (self.weight.data.numel()+self.bias.data.numel()))

    def forward(self, input, add_noise=False):
        if add_noise:
            C_out, C_in, H, W = self.weight.data.shape
            N, _, H_in, W_in = input.shape

            noise_std = torch.exp(self.log_noise_std).repeat(N)
            w = self.weight.repeat(N,1,1,1)
            epsilon_w = torch.zeros_like(w, device=self.log_noise_std.device)
            if self.bias is not None:
                b = self.bias.repeat(N)
                epsilon_b = torch.zeros_like(b, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                num_w, num_b = self.weight.data.numel(), self.bias.data.numel()
                if self.qmc_method == 'sobol':
                    samples = self.sobol_engine.draw(N//2)
                else:
                    samples = torch.from_numpy(self.halton_engine.random(N//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :num_w + num_b] + 1e-12)) \
                                 * torch.cos(2 * np.pi * samples[:, num_w + num_b:])
                epsilon_w[:N * C_out // 2] += normal_samples[:, :num_w].reshape(N * C_out // 2, C_in, H, W).to(self.log_noise_std.device)
                if self.bias is not None:
                    epsilon_b[:N * C_out // 2] += normal_samples[:, num_w:].reshape(N * C_out // 2).to(self.log_noise_std.device)
            else:
                epsilon_w[:N * C_out // 2] += torch.randn((N * C_out // 2, C_in, H, W), device=self.log_noise_std.device)
                if self.bias is not None:
                    epsilon_b[:N * C_out // 2] += torch.randn((N * C_out // 2,), device=self.log_noise_std.device)

            epsilon_w[N * C_out // 2:] -= epsilon_w[:N * C_out // 2]
            if self.bias is not None:
                epsilon_b[N * C_out // 2:] -= epsilon_b[:N * C_out // 2]
            self.epsilon_buf = epsilon_w
            if self.bias is not None:
                self.epsilon_buf_b = epsilon_b

            w += epsilon_w * noise_std[:, None, None, None]
            if self.bias is not None:
                b += epsilon_b * noise_std

            logit_output = F.conv2d(input.reshape(1,N*C_in,H_in, W_in), w, stride=self.stride, padding=self.padding, groups=N)   
            if self.bias is not None:
                logit_output = F.conv2d(input.reshape(1,N*C_in,H_in, W_in), w, b, stride=self.stride, padding=self.padding, groups=N)
            _, _, H_out, W_out = logit_output.shape
            logit_output = logit_output.reshape(N,C_out,H_out,W_out)
        else:
            logit_output = super().forward(input)
        return logit_output

    def backward(self, loss):
        N = loss.shape[0]
        # weight
        tmp_w = torch.stack(torch.split(self.epsilon_buf, split_size_or_sections=self.out_channels, dim=0))    # N, C_out, C_in, H, W
        tmp = loss[:,None,None,None,None] * tmp_w
        self.weight.grad = torch.sum(tmp, dim=0) / (N * torch.exp(self.log_noise_std[:,None,None,None]))

        if self.bias is not None:
            # bias
            tmp_b = torch.stack(torch.split(self.epsilon_buf_b, split_size_or_sections=self.out_channels, dim=0))  # N, C_out
            tmp = loss[:, None] * tmp_b
            self.bias.grad = torch.sum(tmp, 0) / (N * torch.exp(self.log_noise_std))

        # noise std
        if self.bias is not None:
            tmp = torch.sum((tmp_w**2) - 1, dim=[2, 3, 4]) + ((tmp_b**2) - 1)
        else:
            tmp = torch.sum((tmp_w**2) - 1, dim=[2, 3, 4])
        
        self.log_noise_std.grad = torch.sum(tmp * loss[:,None], 0) / N

        self.epsilon_buf = None
        self.epsilon_buf_b = None

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, init_std=1e-3, anti_variable=True):
        super().__init__()
        self.log_noise_std = nn.Parameter(torch.full((1,), np.log(init_std)))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.input_buf = None
        self.epsilon_buf = None
        self.anti_variable = anti_variable

    def forward(self, x, add_noise=False):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / (std + self.eps)
        logit_output = self.weight * normalized_x + self.bias
        shape_o = logit_output.shape
        logit_output = logit_output.reshape((-1, shape_o[-1]))
        if add_noise:
            bs, out_features = logit_output.shape
            epsilon = torch.zeros_like(logit_output, device=self.log_noise_std.device)
            if self.anti_variable:
                epsilon[:bs // 2] += torch.randn((bs // 2, out_features), device=self.log_noise_std.device)
                epsilon[bs // 2:] -= epsilon[:bs // 2]
            else:
                epsilon += torch.randn((bs, out_features), device=self.log_noise_std.device)
            noise = epsilon * torch.exp(self.log_noise_std)
            self.input_buf = normalized_x
            self.epsilon_buf = epsilon
            # del normalized_x
            # import gc
            # gc.collect()
            return (logit_output+noise).reshape(shape_o)
        else:
            return logit_output.reshape(shape_o)

    def backward(self, loss):
        shape_i = self.input_buf.shape
        shape_e = self.epsilon_buf.shape
        is_seq = True if len(shape_i) == 3 else False
        self.input_buf = self.input_buf.reshape((-1, shape_i[-1]))
        self.epsilon_buf = self.epsilon_buf.reshape((-1, shape_e[-1]))
        batch_size = self.input_buf.shape[0]
        if is_seq:
            loss = loss.unsqueeze(-1).repeat_interleave(shape_i[1], dim=0)
        else:
            loss = loss.unsqueeze(-1)
        self.weight.grad = torch.einsum('ni,ni->i', self.input_buf * loss, self.epsilon_buf) / (torch.exp(self.log_noise_std) * batch_size)
        self.bias.grad = torch.einsum('ni,nj->j', loss, self.epsilon_buf) / (torch.exp(self.log_noise_std) * batch_size)
        self.input_buf = None
        self.epsilon_buf = None

class MyBatchNorm2d(torch.nn.Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, init_std=1e-3):
            super(MyBatchNorm2d, self).__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.log_noise_std = nn.Parameter(torch.full((num_features,), np.log(init_std)))
            self.epsilon_buf = None
            self.epsilon_buf_b = None

            # Learnable parameters
            self.gamma = torch.nn.Parameter(torch.ones(num_features))
            self.beta = torch.nn.Parameter(torch.zeros(num_features))

            # Running statistics
            self.running_mean = torch.zeros(num_features)
            self.running_var = torch.ones(num_features)

        def forward(self, x, add_noise=False):
            if self.training:
                
                # Compute batch mean and variance
                batch_mean = x.mean([0, 2, 3])
                batch_var = x.var([0, 2, 3], unbiased=False)

                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                
                # Normalize using batch statistics
                x_hat = (x - batch_mean.view(1, -1, 1, 1)) / torch.sqrt(batch_var.view(1, -1, 1, 1) + self.eps)
            else:
                # Normalize using running statistics
                x_hat = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)

            if add_noise:
                bs = x.shape[0]
                g = self.gamma.unsqueeze(0).repeat(bs,1)
                b = self.beta.unsqueeze(0).repeat(bs,1)
                epsilon_g = torch.zeros_like(g, device=self.log_noise_std.device)
                epsilon_b = torch.zeros_like(b, device=self.log_noise_std.device)
                epsilon_g[:bs // 2] += torch.randn((bs // 2, self.num_features), device=self.log_noise_std.device)
                epsilon_b[:bs // 2] += torch.randn((bs // 2, self.num_features), device=self.log_noise_std.device)
                epsilon_g[bs // 2:] -= epsilon_g[:bs // 2]
                epsilon_b[bs // 2:] -= epsilon_b[:bs // 2]
                self.epsilon_buf = epsilon_g
                self.epsilon_buf_b = epsilon_b
                g += epsilon_g * torch.exp(self.log_noise_std[None, :])
                b += epsilon_b * torch.exp(self.log_noise_std[None, :])
                
            else:
                # Apply scale (gamma) and shift (beta)
                out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
            return out

class Sequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    # # modified forward function
    # def forward(self, input, add_noise: List[bool]=None):
    #     # if add_noise is None:
    #     #     add_noise = [False] * len(self)
    #     # if len(add_noise) != len(self):
    #     #     raise ValueError("The length of add_noise should be equal to the number of modules in the Sequential.")
        
    #     for i, module in enumerate(self):
    #         try:
    #             input = module(input, add_noise[i])
    #         except TypeError:
    #             input = module(input)
    #     return input

    # original forward function
    def forward(self, input, add_noise=False):
        for module in self:
            try:
                input = module(input, add_noise)
            except TypeError:
                input = module(input)
        return input

    def backward(self, loss):
        for module in self:
            try:
                if module.epsilon_buf is not None:
                    module.backward(loss)
            except AttributeError:
                continue

    def fetch_gradient(self):
        gradient_list = []
        for module in self:
            try:
                gradient_list.append(module.weight.grad.detach().cpu())
            except AttributeError:
                continue
        if len(gradient_list)==1:
            return gradient_list[0]
        else:
            return gradient_list


class mlp_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm(5, anti_variable=False)
        self.fc1 = Linear(5,5, anti_variable=True)
        self.fc2 = nn.Linear(5, 5)
        self.acti = nn.ReLU()

    def forward(self, x, add_noise=False):
        x = self.fc1(x, add_noise=add_noise)
        x = self.acti(x)
        x = self.norm(x, add_noise=False)
        x = self.acti(x)
        x = self.fc2(x)
        return x, None

    def backward(self, loss):
        self.fc1.backward(loss)
        # self.norm.backward(loss)


if __name__ == '__main__':
    # batch_size = 2
    # seqlen = 3
    # hidden_dim = 4
    
    # # 初始化一个随机tensor
    # x = torch.randn(batch_size, seqlen, hidden_dim)
    # print(x)
    
    # # 初始化LayerNorm
    # layer_norm = LayerNorm(num_features=hidden_dim)
    # output_tensor = layer_norm(x)
    # print("output after layer norm:\n,", output_tensor)
    
    # torch_layer_norm = torch.nn.LayerNorm(normalized_shape=hidden_dim)
    # torch_output_tensor = torch_layer_norm(x)
    # print("output after torch layer norm:\n", torch_output_tensor)

    # for n,p in layer_norm.named_parameters():
    #     print(n)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input = torch.ones(size=(1, 5), dtype=torch.float, device=device, requires_grad=True)
    # label = torch.full((1, 5), fill_value=0, dtype=torch.float, device=device)
    # criterion = nn.MSELoss(reduction='none')
    # model_bp = mlp_norm().to(device)
    # model_lr = mlp_norm()
    # model_lr.load_state_dict(model_bp.state_dict())
    # model_lr.to(device)

    # output, _ = model_bp(input)
    # loss = criterion(output, label)
    # loss.mean().backward()
    # grad_bp = deepcopy(model_bp.fc1.weight.grad.data)
    # # grad_bp = deepcopy(input.grad.data)

    # optimizer = torch.optim.SGD(model_lr.parameters(), lr=1.)
    # n_repeat = [2 ** i for i in range(1, 20)]
    # n_macro = 500
    # plt.rc('font', family='serif')
    # buf1, buf2 = [], []
    # for repeat in n_repeat:
    #     input_ = input.repeat(repeat, 1).clone().detach().requires_grad_(True)
    #     label_ = label.repeat(repeat, 1)
    #     for i in tqdm(range(n_macro)):
    #         # version 1
    #         output, log_prob = model_lr(input_, add_noise=True)
    #         loss = torch.mean(criterion(output, label_), dim=1)
    #         optimizer.zero_grad()
    #         # loss_new = loss.detach() * log_prob
    #         # loss_new.mean().backward()
    #         model_lr.backward(loss)
    #         grad_lr = deepcopy(model_lr.fc1.weight.grad.data)

    #         # grad_lr = torch.sum(input_.grad, dim=0)
    #         cos_sim = float(nn.functional.cosine_similarity(grad_bp.reshape(-1), grad_lr.reshape(-1), dim=0).detach().cpu().numpy())

    #         # # version 2
    #         # noise = torch.randn(input_.shape).to(device)
    #         # output, _ = model_lr(input_+noise)
    #         # loss = torch.mean(criterion(output, label_), dim=1).unsqueeze(-1)
    #         # grad_lr = torch.einsum('ni,nj->j', loss, noise) / repeat
    #         # cos_sim = float(nn.functional.cosine_similarity(grad_bp.reshape(-1), grad_lr.reshape(-1), dim=0).detach().cpu().numpy())

    #         buf1.append(repeat)
    #         buf2.append(cos_sim)
    # sns.lineplot(x=buf1, y=buf2)
    # plt.xscale("log")
    # plt.tight_layout()
    # plt.show()

    # print(grad_bp)
    # print(grad_lr)



    # # 手写的 BatchNorm2d 和 PyTorch 的 BatchNorm2d
    # my_bn = MyBatchNorm2d(num_features=3)
    # torch_bn = torch.nn.BatchNorm2d(num_features=3)

    # x = torch.randn(10, 3, 32, 32)
    # my_bn_output = my_bn(x)
    # torch_bn_output = torch_bn(x)

    # print("MyBatchNorm2d Output:", my_bn_output[0, :, 0, 0])
    # print("Torch BatchNorm2d Output:", torch_bn_output[0, :, 0, 0])
    # print("Difference:", torch.abs(my_bn_output - torch_bn_output).mean().item())

    model = Linear_(in_features=2, out_features=3)
    repeat_n = 10
    x = torch.randn(2,2,2)
    repeat_s = len(x.shape) * [1]
    repeat_s[0] = 10
    x = x.repeat(*repeat_s)
    y = model(x, add_noise=True)
    y_hat = torch.ones_like(y)
    y = y.view(y.shape[0], -1)
    y_hat = y_hat.view(y_hat.shape[0], -1)
    criterion = nn.MSELoss(reduction='none')
    loss = torch.mean(criterion(y, y_hat), dim=1)
    print(loss.shape)
    model.backward(loss)
    

    # x = torch.randint(0, 10, (2,2,2))
    # shape_o = x.shape
    # x = x.reshape((-1, shape_o[-1]))
    # print(x)
    # x = x.reshape(shape_o)
    # a = torch.randint(0, 10, (3,2))
    
    # print(x)
    # y = x @ a.T
    # print(y)
    # y1 = F.linear(x, a)
    # print(y1)

