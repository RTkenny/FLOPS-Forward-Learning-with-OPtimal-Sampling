import math
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


def bernoulli_allocater(loss_values, budget, p):
    mean_loss = np.mean(loss_values)
    allocated_budget = np.full(len(loss_values), budget / len(loss_values))

    for i in range(len(loss_values)):
        if loss_values[i] < mean_loss:
            if np.random.rand() < p:
                allocated_budget[i] /= 2

    pruned_budget = budget - np.sum(allocated_budget)
    unpruned_indices = [i for i in range(len(loss_values)) if loss_values[i] >= mean_loss or np.random.rand() >= p]

    if unpruned_indices:
        reallocated_budget = pruned_budget / len(unpruned_indices)
        for i in unpruned_indices:
            allocated_budget[i] += reallocated_budget

    return allocated_budget

if __name__ == '__main__':

    budget = 6400
    n = 128
    per_repeat = 20

    sequence = allocate_budget(4,4, n, per_repeat, 0.5)
    if isinstance(sequence, list):
        print(f"The sequence is: {sequence}")
    else:
        print(sequence)