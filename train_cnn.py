import os
import random
import torch
import numpy as np
import argparse
from dataclasses import dataclass, field
from trainer_a import Trainer, TrainerArgs
from transformers import HfArgumentParser
from models.my_CNN_modeling import ResNet, Resnet_ModelArgs, ResNet8, VGG8
from data.dataset import get_train_valid_cifar10, get_train_valid_cifar100, get_train_valid_MNIST
from utils import  print_trainable_parameters


# cifar using batch size 256
@dataclass
class ScriptArguments:
    fuck: int = field(default=0, metadata={"help": "seed"})
    # dataset: str = field(default='MNIST', metadata={"help": "dataset name"})

if __name__ == "__main__":
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29501"
    # p = argparse.ArgumentParser()
    # p.add_argument('--seed', type=int, default=0, help='seed')
    # p.add_argument('--model', type=str, default='resnet', help='model name')
    # p.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    # p.add_argument('--image_size', type=int, default=32, help='image size')
    # p.add_argument('--device', type=str, default='cuda:0', help='device')
    # p.add_argument('--epochs', type=int, default=100, help='epochs')
    # p.add_argument('--per_device_batch_size', type=int, default=200, help='batch size')
    # p.add_argument('--is_log', type=bool, default=False, help='is log')
    # p.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    # p.add_argument('--sample_budget', type=int, default=2, help='repeat n')
    # args = p.parse_args()
    # print(args)
    # print('-'*100)
    # train_args = TrainerArgs(
    #     per_device_train_batch_size=args.per_device_batch_size,
    #     per_device_eval_batch_size=args.per_device_batch_size,
    #     epochs=args.epochs,
    #     is_log=args.is_log,
    #     device=args.device,
    #     learning_rate=args.lr,
    #     sample_n=args.sample_budget
    # )
    
    parser = HfArgumentParser((TrainerArgs, ScriptArguments))
    training_args, script_args = parser.parse_args_into_dataclasses()
    print(training_args.dataset)
    print(training_args.seed)

    
    if training_args.dataset == 'cifar10':
        train_set, test_set = get_train_valid_cifar10('/home/rt/share/', image_size=32)
        # model_args = Resnet_ModelArgs(input_chans=3, num_classes=10, num_blocks=[2,2,2], channels1=16, channels2=32, channels3=64)
        # model_args = Resnet_ModelArgs(input_chans=3, num_classes=10, num_blocks=[1,1,1], channels1=32, channels2=32, channels3=32)
        model_args = Resnet_ModelArgs(input_chans=3, num_classes=10, num_blocks=[2,2,2], channels1=16, channels2=32, channels3=64)
        # model_args = Resnet_ModelArgs(input_chans=3, num_classes=10, num_blocks=[1,1,1], channels1=16, channels2=32, channels3=64)
    elif training_args.dataset == 'cifar100':
        train_set, test_set = get_train_valid_cifar100('/home/rt/share/', image_size=32)
        model_args = Resnet_ModelArgs(input_chans=3, num_classes=100, num_blocks=[3,3,3], channels1=32, channels2=32, channels3=32)
    elif training_args.dataset == 'MNIST':
        train_set, test_set = get_train_valid_MNIST('/home/rt/share/', image_size=28)
        model_args = Resnet_ModelArgs(input_chans=1, num_classes=10, num_blocks=[1,1,1], channels1=16, channels2=16, channels3=16)
    model = ResNet(model_args)
    # model = ResNet8()
    # model = VGG8()

    print_trainable_parameters(model)
    print('-'*100)
    print(model)

    trainer = Trainer(model, train_set, test_set, training_args)
    trainer.train()