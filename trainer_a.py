import time
import re, os
import datetime
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Any, Dict, Literal, Optional, Tuple
from torch.distributions import Normal
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
from accelerate.utils import DistributedDataParallelKwargs
from allocator import bernoulli_allocater, gaussian_allocater

@dataclass
class TrainerArgs(TrainingArguments):
    output_dir: str = './checkpoints' # checkpoint_dir
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    tracker_project_name: str = "flops"
    allocator_type: str = 'bernoulli'
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 64
    train_max_grad_norm: float = 1.0
    epochs: int = 10
    is_log: bool = False
    log_interval: int = 10
    gradient_estimation_strategy: str = 'bp' # 'perturb_outside_all_params', 'perturb_outside_tensor_wise', 'perturb_outside_all_params_allocator', 'bp'
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    accelerator_kwargs: Dict[str, Any] = field(default_factory=dict)
    project_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate: float = 1e-3
    sample_n: int = 1
    zo_eps: float = 1e-3
    dataset: str = 'ImageNet'
    is_grad_clip: bool = False


class Trainer(Trainer):
    def __init__(self, model, train_dataset, eval_dataset, args: TrainerArgs):
        self.random_seed = 0
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.epochs = args.epochs
        self.allocator_type = args.allocator_type
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.gradient_estimation_strategy = args.gradient_estimation_strategy
        self.train_max_grad_norm = args.train_max_grad_norm
        self.is_log = args.is_log
        self.output_dir = args.output_dir
        self.log_interval = args.log_interval
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.learning_rate = args.learning_rate
        self.sample_n = args.sample_n
        self.zo_eps = args.zo_eps
        self.is_grad_clip = args.is_grad_clip

        accelerator_project_config = ProjectConfiguration(**args.project_kwargs)
        print('init accelerator')
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            log_with=args.log_with,
            project_config=accelerator_project_config,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[kwargs],
            **args.accelerator_kwargs,
        )

        
        is_using_tensorboard = args.log_with is not None and args.log_with == "tensorboard"
        config_dict = args.to_dict()
        if self.accelerator.is_main_process:
            print('init tracker')
            self.accelerator.init_trackers(
                args.tracker_project_name,
                config=dict(flops_config=config_dict)
                if not is_using_tensorboard
                else args.to_dict(),
                init_kwargs=args.tracker_kwargs,
            )
        
        print('init model')
        device = self.accelerator.device
        self.train_loader = DataLoader(self.train_dataset, self.per_device_train_batch_size, shuffle=True, num_workers=64)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.95)
        if self.gradient_estimation_strategy == 'bp':
            self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.scheduler)

    def train(self):
        global_step = 0
        self.grad_marker = False 
        epochs = self.epochs
        for epoch in range(epochs):
            global_step = self.step(epoch, global_step)

    def step(self, epoch, global_step):
        info = defaultdict(list)
        print(f"Epoch {epoch + 1} started")
        self.model.train()
        with tqdm(self.train_loader) as tqdm_range:
            for data, target in tqdm_range:
                device = self.accelerator.device
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                if self.gradient_estimation_strategy == 'perturb_outside_all_params':
                    loss, output = self.zo_step_all_params(self.model, data, target, self.sample_n)
                    bs = data.shape[0]

                elif self.gradient_estimation_strategy == 'perturb_outside_tensor_wise':
                    loss, output = self.zo_step_tensor_wise(self.model, data, target, self.sample_n)
                    bs = data.shape[0]
                    # try:
                    #     self.accelerator.backward(loss)
                    # except:
                    #     pass
                
                elif self.gradient_estimation_strategy == 'perturb_outside_all_params_allocator':
                    if epoch <1:
                        p = 0.3
                    else:
                        p = 0.01
                    loss, output = self.zo_step_all_params_allocator(self.model, data, target, self.sample_n, p)
                    bs = data.shape[0]

                elif self.gradient_estimation_strategy == 'bp':
                    output, _ = self.model(data, add_noise=None)
                    loss = self.criterion(output, target)
                    loss = torch.mean(loss)
                    self.accelerator.backward(loss)

                else:
                    raise ValueError(f"Gradient estimation strategy {self.gradient_estimation_strategy} not supported")
                
                if self.accelerator.sync_gradients and self.is_grad_clip:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_max_grad_norm,
                        )
                
                # if not self.grad_marker:
                #     self.grad_marker = True
                #     # print('grad is not None')
                #     for name, param in self.model.named_parameters():
                #         if param.grad is None:
                #             param.grad = torch.zeros_like(param)
                #     # print('-'*100)
                #     # print('grad is not None')
                self.optimizer.step()

                global_step += 1
                info['loss'].append(loss)
                info['accuracy'].append(torch.sum(torch.where(target == torch.argmax(output, 1), 1, 0), dtype=torch.float)/target.shape[0])
                
                tl = torch.mean(torch.tensor(info['loss'])).cpu().detach().numpy()
                ta = torch.mean(torch.tensor(info['accuracy'])).cpu().detach().numpy() # dtype=torch.float
                tqdm_range.set_description(f"train loss:{tl}; train accuracy {100*ta:.2f}%")
                # print(f"Rank: {rank}, train loss: {train_loss}, Train Accuracy: {train_accuracy}, train_count:{train_count}")
                if global_step % self.log_interval == 0 and self.is_log:
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
                        info.update({"epoch": epoch})
                        self.accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                    else:
                        raise ValueError(
                            "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                        )

        self.scheduler.step()
        if self.accelerator.is_main_process and self.is_log:
            device = self.accelerator.device 
            valid_loss, correct = self.evaluate(device)
            tl = torch.mean(torch.tensor(info['loss'])).cpu().detach().numpy()
            ta = torch.mean(torch.tensor(info['accuracy'])).cpu().detach().numpy()
            # self._save_checkpoint(epoch)
            print(f'Epoch {epoch + 1}, train loss: {tl}, train acc: {100*ta:.2f}%, valid loss: {valid_loss}, valid_acc: {100*correct:.2f}%, lr:{self.scheduler.get_last_lr()[0]}')
        return global_step
    
    def compute_loss(self, model, data, target, return_mean=True):
        output, _ = model(data)
        loss = self.criterion(output, target)
        if return_mean:
            return loss.mean(), output
        return loss, output
    
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

    def zo_forward(self, model, data, target, retain_graph=False, return_mean=True):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if not retain_graph:
            with torch.inference_mode():
                loss, output = self.compute_loss(model, data, target, return_mean)
        else:
            loss, output = self.compute_loss(model, data, target, return_mean)
        return loss.detach(), output

    def zo_step_all_params(self, model, data, target, sample_budget=1):
        """
        Estimate gradient by perturbing all params together. Return the loss from f(theta + z)
        """

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        for _ in range(sample_budget):
            self.zo_random_seed = np.random.randint(1000000000)

            self.perturb_all_params(scaling_factor=1)
            loss1, _ = self.zo_forward(model, data, target)
            self.perturb_all_params(scaling_factor=-2)
            loss2, _ = self.zo_forward(model, data, target)

            self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
            self.projected_grad = self.projected_grad / float(sample_budget)

            self.perturb_all_params(scaling_factor=1)
            self.zo_backward()

        loss, output = self.zo_forward(model, data, target)
        return loss, output
    
    def zo_step_all_params_allocator(self, model, data, target, sample_budget=1, p=0.1):
        """
        Estimate gradient with optimal sampling, either using bernoulli allocator or gaussian one. Return the loss from f(theta + z)
        """
        loss_values, _ = self.zo_forward(model, data, target, return_mean=False)
        if self.allocator_type == 'gaussian':
            ops_dataloader = gaussian_allocater(data, target, loss_values, sample_budget)
        elif self.allocator_type == 'bernoulli':
            ops_dataloader = bernoulli_allocater(data, target, loss_values, sample_budget, p)

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        for alloc_data, alloc_target in ops_dataloader:
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            self.perturb_all_params(scaling_factor=1)
            loss1, _ = self.zo_forward(model, alloc_data, alloc_target)

            # Second function evaluation
            self.perturb_all_params(scaling_factor=-2)
            loss2, _ = self.zo_forward(model, alloc_data, alloc_target)

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

    def evaluate(self, device):
        valid_loss = 0.
        valid_count = 0
        correct = 0
        test_loader = DataLoader(self.eval_dataset, batch_size=self.per_device_eval_batch_size, shuffle=False, num_workers=4)
        criterion = nn.CrossEntropyLoss(reduction='none')
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = self.model(data, add_noise=None)
                loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                valid_loss += torch.sum(loss).cpu().detach().numpy()
                valid_count += data.shape[0]
            valid_loss /= valid_count
            correct /= valid_count
        return valid_loss, correct
    
    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f'{self.output_dir}/model_{epoch}.pt')
        print(f"Model saved at {self.output_dir}/model_{epoch}.pt")
