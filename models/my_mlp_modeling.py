import torch
from torch import nn
# from torch.nn import Linear
import loralib as lora
from .modules import Linear, LayerNorm

class LoRA_Net(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.acti = nn.ReLU()
        self.fc1 = lora.Linear(28 * 28, 128, r=r)
        self.fc2 = lora.Linear(128, 64, r=r)
        self.fc3 = lora.Linear(64, 10, r=r)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = self.acti(x)
        x = self.fc3(x)
        return x

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.log_noise_std = 1e-1
        self.acti = nn.ReLU()
        self.fc1 = Linear(28 * 28, 128)
        # self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, 64)
        self.fc5 = Linear(64, 10)
        self.checkpoint1 = None
        self.start1 = None

    def forward(self, x, add_noise=False):
        x = x.view(-1, 28 * 28)
        x = self.acti(self.fc1(x, add_noise=add_noise))
        # x = self.acti(self.fc2(x, add_noise=add_noise))
        x = self.acti(self.fc3(x, add_noise=add_noise))
        x = self.acti(self.fc4(x, add_noise=add_noise))
        self.checkpoint1 = x
        # y = x.detach()
        y = x
        # y.requires_grad = True
        # y.retain_grad()
        self.start1 = y
        y = self.fc5(self.start1, add_noise=add_noise)
        return y, None

    def backward(self, loss):
        # print(type(self.start1.grad))
        # local_loss = self.checkpoint1 * self.start1.grad
        # local_loss.sum().backward()
        self.fc1.backward(loss)
        # self.fc2.backward(loss)
        self.fc3.backward(loss)
        self.fc4.backward(loss)
        self.fc5.backward(loss)


class mlp_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = LayerNorm(128, anti_variable=True)
        self.norm2 = LayerNorm(64, anti_variable=True)
        self.fc1 = Linear(28*28, 128, anti_variable=True)
        self.fc2 = Linear(128, 64, anti_variable=True)
        self.fc3 = Linear(64, 10, anti_variable=True)
        self.acti = nn.ReLU()

    def forward(self, x, add_noise=False):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x, add_noise=add_noise)
        x = self.norm1(x, add_noise=add_noise)
        x = self.acti(x)
        x = self.fc2(x, add_noise=add_noise)
        x = self.norm2(x, add_noise=add_noise)
        x = self.acti(x)
        x = self.fc3(x, add_noise=add_noise)
        return x, None

    def backward(self, loss):
        self.fc1.backward(loss)
        self.fc2.backward(loss)
        self.fc3.backward(loss)
        self.norm1.backward(loss)
        self.norm2.backward(loss)

class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.acti = nn.ReLU()
        self.fc1 = Linear(3 * 32 * 32, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, 64)
        self.fc5 = Linear(64, 10)

    def forward(self, x, add_noise=False):
        x = x.view(-1, 3 * 32 * 32)  # Correct input size
        x = self.acti(self.fc1(x, add_noise=add_noise))
        x = self.acti(self.fc2(x, add_noise=add_noise))
        x = self.acti(self.fc3(x, add_noise=add_noise))
        x = self.acti(self.fc4(x, add_noise=add_noise))
        x = self.fc5(x, add_noise=add_noise)
        return x, None

    def backward(self, loss):
        self.fc1.backward(loss)
        self.fc2.backward(loss)
        self.fc3.backward(loss)
        self.fc4.backward(loss)
        self.fc5.backward(loss)

if __name__ == "__main__":
    torch.manual_seed(0)
    import torch.optim as optim
    model = MLP2()
    optimizer = optim.SGD(MLP2().parameters(), lr=0.01)
    x1 = torch.randn(1, 1, 28, 28)
    x2 = torch.randn(1, 1, 28, 28)
    x3 = torch.randn(1, 1, 28, 28)
    x = torch.cat([x1, x2, x3], dim=0)
    out, _ = model(x)
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(out, torch.ones_like(out))
    loss.mean().backward()
    print(model.fc3.weight.grad)
    print(model.start1.grad.shape)
    model.backward(loss)
    print(model.fc3.weight.grad)
