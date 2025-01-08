import re, os
import datetime
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from transformers import Trainer, TrainingArguments
from allocator import allocate_budget

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

@dataclass
class TrainerArgs(TrainingArguments):
    output_dir: str = './checkpoints'
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 16
    epochs: int = 10
    is_log: bool = False
    log_interval: int = 50
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate: float = 1e-3
    sample_n: int = 10
    zo_eps: float = 1e-3
    dataset: str = 'ImageNet'


class Trainer(Trainer):
    def __init__(self, model, train_dataset, eval_dataset, args: TrainerArgs):
        self.random_seed = 0
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.epochs = args.epochs
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.is_log = args.is_log
        self.log_interval = args.log_interval
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.learning_rate = args.learning_rate
        self.sample_n = args.sample_n
        self.zo_eps = args.zo_eps
        

    def train(self, is_lr: bool = False, lr_type: str = 'vanilla', is_parallel: bool = False, is_ddp: bool = False, is_grad_clip: bool = False):
        if is_parallel:
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_worker, args=(world_size, self.model, self.epochs, is_lr, lr_type, is_ddp, is_grad_clip), nprocs=world_size, join=True)
        else:
            self.train_worker(-1, 1, self.model, self.epochs, is_lr, lr_type)

    def train_worker(self, rank, world_size, model, epochs=8, is_lr=False, lr_type='vanilla', is_ddp=False, is_grad_clip=False):
        writer = None

        if rank != -1:
            device = torch.device(f'cuda:{rank % 4}')
            setup(rank, world_size)
            model = model.to(device)
            if is_ddp:
                ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

            if is_lr:
                train_loader = DataLoader(self.train_dataset, batch_size=self.per_device_train_batch_size, shuffle=False, num_workers=4)
            else:
                train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
                train_loader = DataLoader(self.train_dataset, batch_size=self.per_device_train_batch_size, shuffle=False, sampler=train_sampler, num_workers=16)

        else:
            device = self.device
            train_loader = DataLoader(self.train_dataset, self.per_device_train_batch_size, shuffle=True, num_workers=64)
            model = model.to(device)

        if is_ddp:
            optimizer = optim.Adam(ddp_model.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        

        if self.is_log and (rank == -1 or rank == 0):
            current_time = re.sub(r'\D', '', str(datetime.datetime.now())[4:-7])
            log_dir = './logs/' + type(model).__name__ + '/LR_' + current_time if is_lr else \
                './logs/' + type(model).__name__ + '/BP_' + current_time
            writer = SummaryWriter(log_dir=log_dir)

            checkpoint_dir = './checkpoints/' + type(model).__name__ + '/LR_' + current_time if is_lr else \
                './checkpoints/' + type(model).__name__ + '/BP_' + current_time
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)


        step_count = 0
        for epoch in range(epochs):
            train_count = torch.tensor(0, dtype=torch.double, device=device)
            train_loss = torch.tensor(0, dtype=torch.double, device=device)
            train_accuracy = torch.tensor(0, dtype=torch.double, device=device)
            # train_count = 0
            # train_loss = 0.
            # train_accuracy = 0

            model.train()
            with tqdm(train_loader) as tqdm_range:
                for data, target in tqdm_range:
                    data = data.to(device)
                    target = target.to(device)
                    if is_lr:
                        
                        if lr_type == 'allocater':
                            # version 3 budget allocating
                            with torch.no_grad():
                                bs = data.shape[0]
                                output, _ = self.model(data, add_noise=False)
                                loss = self.criterion(output, target)
                                sorted_loss, indices = torch.sort(loss, descending=False)
                                loss_bsl = torch.mean(loss).detach()
                                ranks = torch.empty_like(loss, device=self.device)
                                ranks[indices] = torch.arange(1, len(loss) + 1, dtype=torch.float, device=self.device)
                                repeats = torch.empty_like(loss, dtype=torch.int, device=self.device)
                                repeats[indices] = torch.tensor(allocate_budget(epoch + 1, self.epochs, bs, self.sample_n),
                                                                dtype=torch.int, device=self.device)
                                
                            repeats = torch.full((bs,), fill_value=self.sample_n, dtype=torch.int, device=self.device)
                            data = data.repeat_interleave(repeats, dim=0)
                            target = target.repeat_interleave(repeats, dim=0)
                            optimizer.zero_grad()
                            output, _ = model(data, add_noise=True)
                            loss = self.criterion(output, target)
                            model.backward(loss-loss_bsl)
                            
                        elif lr_type == 'vanilla':
                            # with torch.no_grad():
                            #     bs = data.shape[0]
                            #     output, _ = model(data)
                            #     loss = criterion(output, target)

                            bs = data.shape[0]
                            repeats = torch.full((bs,), fill_value=self.sample_n, dtype=torch.int, device=device)
                            data = data.repeat_interleave(repeats, dim=0)
                            target = target.repeat_interleave(repeats, dim=0)
                            # loss_bsl = loss.repeat_interleave(repeats, dim=0)

                            # data = data.repeat(self.sample_n, 1, 1, 1)
                            # target = target.repeat(self.sample_n)

                            optimizer.zero_grad()
                            output, _ = model(data, add_noise=True)
                            loss = self.criterion(output, target)
                            # mloss = torch.full((data.shape[0],), fill_value=torch.mean(loss), device=device)
                            # model.backward(mloss)
                            model.backward(loss)
                            # model.backward(loss-loss_bsl)
                            
                        elif lr_type == 'tensor_wise_perturb':
                            module_len = model.module_len
                            optimizer.zero_grad()

                            # bs = data.shape[0]
                            # repeats = torch.full((bs,), fill_value=sample_n, dtype=torch.int, device=self.device)
                            # data = data.repeat_interleave(repeats, dim=0)
                            # target = target.repeat_interleave(repeats, dim=0)

                            data = data.repeat(self.sample_n, 1, 1, 1)
                            target = target.repeat(self.sample_n)

                            with torch.no_grad():
                                for i in range(module_len):
                                    add_noise = [True if j == i else False for j in range(module_len)]
                                    output, _ = model(data, add_noise=add_noise)
                                    loss = self.criterion(output, target)
                                    model.backward(loss)
                                    # torch.cuda.empty_cache()
                            
                        elif lr_type == 'tensor_wise_perturb_reuse':
                            # with torch.no_grad():
                            #     bs = data.shape[0]
                            #     output, _ = self.model(data, add_noise=[False, False])
                            #     loss = self.criterion(output, target)
                            #     loss_bsl = torch.mean(loss).detach()

                            module_len = model.module_len
                            optimizer.zero_grad()
                            # bs = data.shape[0]
                            # repeats = torch.full((bs,), fill_value=self.sample_n, dtype=torch.int, device=self.device)
                            # data = data.repeat_interleave(repeats, dim=0)
                            # target = target.repeat_interleave(repeats, dim=0)
                            target = target.repeat(self.sample_n)
                            with torch.no_grad():
                                for i in range(module_len):
                                    add_noise = [True if j == i else False for j in range(module_len)]
                                    output, _ = model(data, add_noise=add_noise, repeats=self.sample_n)
                                    loss = self.criterion(output, target)
                                    model.backward(loss)

                        elif lr_type == 'perturb_outside_all_params':
                            optimizer.zero_grad()
                            loss, output = self.zo_step_all_params(model, data, target, self.sample_n)
                            bs = data.shape[0]
                            loss = loss*bs

                        elif lr_type == 'perturb_outside_tensor_wise':
                            optimizer.zero_grad()
                            loss, output = self.zo_step_tensor_wise(model, data, target, self.sample_n)
                            bs = data.shape[0]
                            loss = loss*bs

                        elif lr_type == 'non-diff':
                            # optimizing non-differentiable objective
                            bs = data.shape[0]
                            repeats = torch.full((bs,), fill_value=self.sample_n, dtype=torch.int, device=self.device)
                            data = data.repeat_interleave(repeats, dim=0)
                            target = target.repeat_interleave(repeats, dim=0)
                            optimizer.zero_grad()
                            output, _ = model(data, add_noise=[False, True])
                            loss = self.criterion(output, target)
                            pred = output.argmax(dim=1, keepdim=True)
                            acc = (pred.eq(target.view_as(pred))/-1).squeeze(-1)
                            model.backward(acc)
                            
                    else:
                        # print('BP training')
                        optimizer.zero_grad()
                        if is_ddp:
                            output, _ = ddp_model(data, add_noise=None)
                        else:
                            output, _ = model(data, add_noise=None)
                        loss = self.criterion(output, target)
                        loss_new = torch.mean(loss)
                        loss_new.backward()
                    
                    if rank != -1 and not is_ddp:
                        # All-reduce gradients
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad = param.grad.contiguous()
                                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                                param.grad /= world_size
                        dist.barrier()

                    if is_grad_clip and epoch > -1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    train_loss += torch.sum(loss)
                    train_accuracy += torch.sum(torch.where(target == torch.argmax(output, 1), 1, 0))
                    train_count += torch.tensor(output.shape[0], device=device)

                    # train_loss += torch.sum(loss).cpu().detach().numpy()
                    # train_accuracy += torch.sum(torch.where(target == torch.argmax(output, 1), 1, 0)).cpu().detach().numpy()
                    # train_count += output.shape[0]
                    
                    step_count += 1
                    if rank !=-1:
                        dist.all_reduce(train_accuracy, op=dist.ReduceOp.SUM)
                        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
                        train_accuracy /= world_size
                        train_loss /= world_size
                    
                    tl = (train_loss/train_count).cpu().detach().numpy()
                    ta = (train_accuracy/(train_count)).cpu().detach().numpy()
                    tqdm_range.set_description(f"train loss:{tl}; train accuracy {100*ta:.2f}%")
                    # print(f"Rank: {rank}, train loss: {train_loss}, Train Accuracy: {train_accuracy}, train_count:{train_count}")
                    if step_count % self.log_interval == 0 and self.is_log:
                        if rank == -1 or rank == 0:
                            ttl = (train_loss/train_count).cpu().detach().numpy()
                            tta = (train_accuracy/(train_count)).cpu().detach().numpy()
                            writer.add_scalar('loss/train_loss', ttl, step_count)
                            writer.add_scalar('accuracy/train_accuracy', tta, step_count)

            scheduler.step()

            if rank == -1 or rank == 0:
                valid_loss, correct = self.evaluate(model, device)
                tl = (train_loss/train_count).cpu().detach().numpy()
                ta = (train_accuracy/(train_count)).cpu().detach().numpy()
                if self.is_log:
                    writer.add_scalar('loss/valid_loss', valid_loss, step_count)
                    writer.add_scalar('accuracy/valid_accuracy', correct, step_count)
                    torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch}.pt')
                print(f'Epoch {epoch + 1}, train loss: {tl}, train acc: {ta}, valid loss: {valid_loss}, valid_acc: {correct}, lr:{scheduler.get_last_lr()[0]}')

        if rank != -1:
            cleanup()
    
    def compute_loss(self, model, data, target):
        output, _ = model(data)
        loss = self.criterion(output, target)
        return loss.mean(), output
    
    def perturb_all_params(self, random_seed=None, scaling_factor=1):
        """
        Perturb the all parameters with random vector z.
        Input: 
        - random_seed: random seed for in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zo_eps

    def perturb_single_module(self, layer_name, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            if name == layer_name:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.zo_eps

    def zo_forward(self, model, data, target, retain_graph=False):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if not retain_graph:
            with torch.inference_mode():
                loss, output = self.compute_loss(model, data, target)
        else:
            loss, output = self.compute_loss(model, data, target)
        return loss.detach(), output

    def zo_step_all_params(self, model, data, target, sample_budget=1):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        # print(len(self.named_parameters_to_optim))

        for _ in range(sample_budget):
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            self.perturb_all_params(scaling_factor=1)
            loss1, _ = self.zo_forward(model, data, target)

            # Second function evaluation
            self.perturb_all_params(scaling_factor=-2)
            loss2, _ = self.zo_forward(model, data, target)

            self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
            self.projected_grad = self.projected_grad / float(sample_budget)

            # Reset model back to its parameters at start of step
            self.perturb_all_params(scaling_factor=1)
            self.zo_backward()

        loss, output = self.zo_forward(model, data, target)
        return loss, output

    def zo_step_tensor_wise(self, model, data, target, sample_budget=1):
        self.named_parameters_to_optim = []
        # print(len(self.named_parameters_to_optim))
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        for name, param in self.named_parameters_to_optim:
            for _ in range(sample_budget):
                self.zo_random_seed = np.random.randint(1000000000)
                self.perturb_single_module(name, scaling_factor=1)
                loss1, _ = self.zo_forward(model, data, target)

                self.perturb_single_module(name, scaling_factor=-2)
                loss2, _ = self.zo_forward(model, data, target)

                self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
                self.projected_grad = self.projected_grad / float(sample_budget)

                self.perturb_single_module(name, scaling_factor=1)
                self.zo_backward(name)

        loss, output = self.zo_forward(model, data, target)
        return loss, output

    def zo_backward(self, target_name=None):
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            if target_name is not None and target_name != name:
                continue
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if param.grad is None:
                param.grad = self.projected_grad * z
            else:
                param.grad += self.projected_grad * z

    def evaluate(self, model, device):
        valid_loss = 0.
        valid_count = 0
        correct = 0
        test_loader = DataLoader(self.eval_dataset, batch_size=self.per_device_eval_batch_size, shuffle=False, num_workers=4)
        criterion = nn.CrossEntropyLoss(reduction='none')
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data, add_noise=None)
                loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                valid_loss += torch.sum(loss).cpu().detach().numpy()
                valid_count += data.shape[0]
            valid_loss /= valid_count
            correct /= valid_count
        return valid_loss, correct

if __name__ == '__main__':
    a = torch.tensor([[1,2,3],[4,5,6]])
    print(a.shape)
    print(a.repeat(3,))
