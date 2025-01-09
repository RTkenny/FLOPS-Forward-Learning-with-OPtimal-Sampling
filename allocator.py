import math
import torch
import numpy as np

def gaussian_allocater(data: torch.Tensor, target: torch.Tensor, loss_values: torch.Tensor, sample_n: int, length_scale: float = 1.0, variance: float = 1.0):
    bs = data.shape[0]
    budget = bs * sample_n

    # Reparameterize the mean vector using a linear structure
    mean = torch.matmul(loss_values, torch.randn(len(loss_values), len(loss_values[0]), device=loss_values.device))

    # Reparameterize the covariance matrix using the RBF kernel
    def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return variance * torch.exp(-0.5 / length_scale**2 * sqdist)

    cov = rbf_kernel(loss_values, loss_values, length_scale, variance)

    allocated_budget = torch.distributions.MultivariateNormal(mean, cov).sample()

    # Normalize the allocated budget to match the total budget
    allocated_budget = torch.maximum(allocated_budget, torch.tensor(0.0, device=allocated_budget.device))  # Ensure no negative allocations
    allocated_budget = (allocated_budget / torch.sum(allocated_budget)) * budget
    allocated_budget = allocated_budget.int()

    diff = budget - torch.sum(allocated_budget)
    diff = diff.item()
    if diff >= 0:
        for i in range(diff):
            random_index = np.random.choice(bs)
            allocated_budget[random_index] += 1
    else:
        raise ValueError("The allocated budget is greater than the total budget")

    data = data.repeat_interleave(allocated_budget, dim=0)
    target = target.repeat_interleave(allocated_budget, dim=0)

    dataset = torch.utils.data.TensorDataset(data, target)
    gaussian_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    return gaussian_dataloader


def bernoulli_allocater(data: torch.Tensor, target: torch.Tensor, loss_values: torch.Tensor, sample_n: int, p: int):
    # p is a number between 0 and 1
    bs = data.shape[0]
    budget = bs * sample_n
    mean_loss = torch.mean(loss_values)
    equal_budget, allocated_budget = torch.full((bs,), sample_n, dtype=torch.int, device=data.device), torch.full((bs,), sample_n, dtype=torch.int, device=data.device)

    # print(equal_budget)
    for i in range(bs):
        if loss_values[i] < mean_loss:
            if np.random.rand() < p:
                allocated_budget[i] = allocated_budget[i] // 2

    pruned_budget = budget - torch.sum(allocated_budget)
    unpruned_indices = ~(allocated_budget < equal_budget)
    # print(f'unpruned_indices: {unpruned_indices}')
    
    if torch.sum(unpruned_indices)>0:
        reallocated_budget = pruned_budget // torch.sum(unpruned_indices)
        allocated_budget[unpruned_indices] += reallocated_budget

    diff = torch.sum(equal_budget) - torch.sum(allocated_budget)
    diff = diff.item()
    if diff >= 0:
        unpruned_indexs = torch.where(unpruned_indices)[0]
        for i in range(diff):
            random_index = np.random.choice(unpruned_indexs.cpu().numpy())
            allocated_budget[random_index] += 1
        # unpruned_indexs = torch.where(unpruned_indices)[0]
        # random_index = np.random.choice(unpruned_indexs.numpy())
        # random_index = np.random.choice(bs)
        # allocated_budget[random_index] += diff
    else:
        raise ValueError("The allocated budget is greater than the total budget")
    
    # print(f"diff: {diff}")
    # print(f'sum of allocated_budget: {torch.sum(allocated_budget)}')

    data = data.repeat_interleave(allocated_budget, dim=0)
    target = target.repeat_interleave(allocated_budget, dim=0)

    dataset = torch.utils.data.TensorDataset(data, target)
    bernoulli_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    return bernoulli_dataloader
