import os
import torch
import argparse
from models.my_mlp_modeling import MLP2
from trainer import Trainer, TrainerArgs
from data.dataset import get_train_valid_food101, get_train_valid_cifar10, get_train_valid_cifar100, get_train_valid_ImageNet, get_train_valid_MNIST
from utils import weight_mapping_vit, weight_mapping_vit_embeddings , mark_only_classifier_as_trainable, print_trainable_parameters

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29520"

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0, help='seed')
    p.add_argument('--model', type=str, default='base', help='model name')
    p.add_argument('--output_dir', type=str, default=None, help='output_dir')
    p.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
    p.add_argument('--image_size', type=int, default=224, help='image size')
    p.add_argument('--device', type=str, default='cuda:0', help='device')
    p.add_argument('--epochs', type=int, default=10, help='epochs')
    p.add_argument('--per_device_train_batch_size', type=int, default=64, help='train batch size') # 64 for LR, and 128 for BP wo parrallel and 100 for parallel
    p.add_argument('--per_device_eval_batch_size', type=int, default=1024, help='eval batch size')
    p.add_argument('--is_log', type=bool, default=True, help='is log')
    p.add_argument('--log_interval', type=int, default=5, help='log interval')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate') # 1e-5 for LR, and 1e-4 for BP
    p.add_argument('--sample_budget', type=int, default=24, help='repeat n')
    args = p.parse_args()

    print(args)
    print('-'*100)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    train_args = TrainerArgs(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        epochs=args.epochs,
        is_log=args.is_log,
        log_interval=args.log_interval,
        device=args.device,
        learning_rate=args.lr,
        sample_n=args.sample_budget,
        dataset=args.dataset
    )

    train_set, test_set = get_train_valid_MNIST('./data/')

    model = MLP2()
    
    print_trainable_parameters(model)

    trainer = Trainer(model, train_set, test_set, train_args)
    trainer.train(is_lr=True, lr_type='vanilla')

    # print(train_set[0])
    # trainer.compute_loss(model, train_set[0])
    