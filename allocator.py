import math
import torch
import numpy as np

def allocate_budget(current_epoch: int, total_epochs: int, n: int, per_repeat: int, ratio: float = 0.8):
    budget = n * per_repeat
    sequence = np.zeros(n)
    d = math.floor((2*budget) / (n*(n+1)))
    sequence = np.arange(d, d*(n+1), d)
    gap = budget - sum(sequence)
    realoc = np.full(n, gap // n, dtype=int)
    realoc[0] += gap % n
    sequence += realoc

    sequence = np.full(n, fill_value=per_repeat)
    num = math.floor(n * (ratio/2))
    index = -1
    for i in range(num):
        if current_epoch <= (total_epochs):
            sequence[i] = per_repeat / 4
            sequence[index] = per_repeat
        elif current_epoch <=(total_epochs-2):
            pass
        else:
            sequence[i] = per_repeat * 3
            sequence[index] = per_repeat / 2

        sequence[i] = per_repeat + (per_repeat * (2 / 3))
        sequence[index] = per_repeat - (per_repeat * (2 / 3))
        index -= 1
    return sequence


def gaussain_allocater(loss_values, budget, correlation_matrix):
    # Reparameterize the mean vector using a linear structure
    mean = np.dot(loss_values, np.random.randn(len(loss_values), len(loss_values[0])))

    # Reparameterize the covariance matrix using the RBF kernel
    def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return variance * np.exp(-0.5 / length_scale**2 * sqdist)

    cov = rbf_kernel(loss_values, loss_values)

    allocated_budget = np.random.multivariate_normal(mean, cov, size=len(loss_values))
    
    # Normalize the allocated budget to match the total budget
    allocated_budget = np.maximum(allocated_budget, 0)  # Ensure no negative allocations
    allocated_budget = (allocated_budget / np.sum(allocated_budget)) * budget
    
    return allocated_budget


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

if __name__ == '__main__':

    # budget = 6400
    # n = 128
    # per_repeat = 20
    # sequence = allocate_budget(4,4, n, per_repeat, 0.5)
    # if isinstance(sequence, list):
    #     print(f"The sequence is: {sequence}")
    # else:
    #     print(sequence)

    a = np.array([4, 2, 4])
    b = np.array([4, 4, 4])
    c = np.where(~(a<b))
    print(np.sum(~(a<b)))
    print(c)
    