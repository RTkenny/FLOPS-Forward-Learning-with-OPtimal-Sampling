import os
import torch
import argparse
from dataclasses import dataclass, field
from models.my_mlp_modeling import MLP2
from trainer_a import Trainer, TrainerArgs
from transformers import HfArgumentParser
from data.dataset import get_train_valid_food101, get_train_valid_cifar10, get_train_valid_cifar100, get_train_valid_ImageNet, get_train_valid_MNIST
from utils import weight_mapping_vit, weight_mapping_vit_embeddings , mark_only_classifier_as_trainable, print_trainable_parameters

@dataclass
class ScriptArguments:
    fuck_yijiepeng: int = field(default=0, metadata={"help": "fuck the devil yijie peng"})

if __name__ == "__main__":
    parser = HfArgumentParser((TrainerArgs, ScriptArguments))
    training_args, script_args = parser.parse_args_into_dataclasses()
    train_set, test_set = get_train_valid_MNIST('/home/rt/share/')
    model = MLP2()
    print_trainable_parameters(model)
    trainer = Trainer(model, train_set, test_set, training_args)
    print('Training...')
    trainer.train()

    # from torch.utils.data import DataLoader
    # from allocator import bernoulli_allocater
    # dataloader = DataLoader(train_set, batch_size=64)
    # dataloader_iter = iter(dataloader)
    # data, target = next(dataloader_iter)
    # output,_ = model(data)
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # loss = criterion(output, target)
    # alloc_budget = bernoulli_allocater(data, target, loss, 10, 0.5)
    # print(alloc_budget)

    # print(train_set[0])
    # trainer.compute_loss(model, train_set[0])
    