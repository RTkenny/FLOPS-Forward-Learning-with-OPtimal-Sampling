import os
import torch
import argparse
from trainer import Trainer, TrainerArgs
from models.my_vit_modeling import Transformer, ModelArgs
from data.dataset import get_train_valid_food101, get_train_valid_cifar10, get_train_valid_cifar100, get_train_valid_ImageNet
from utils import weight_mapping_vit, weight_mapping_vit_embeddings , mark_only_classifier_as_trainable, print_trainable_parameters

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29520"

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0, help='seed')
    p.add_argument('--model', type=str, default='large', help='model name')
    p.add_argument('--dataset', type=str, default='ImageNet', help='dataset name')
    p.add_argument('--image_size', type=int, default=224, help='image size')
    p.add_argument('--device', type=str, default='cuda:2', help='device')
    p.add_argument('--epochs', type=int, default=10, help='epochs')
    p.add_argument('--per_device_train_batch_size', type=int, default=12, help='train batch size') # 64 for LR, and 128 for BP wo parrallel and 100 for parallel
    p.add_argument('--per_device_eval_batch_size', type=int, default=1024, help='eval batch size')
    p.add_argument('--is_log', type=bool, default=False, help='is log')
    p.add_argument('--log_interval', type=int, default=5, help='log interval')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate') # 1e-5 for LR, and 1e-4 for BP
    p.add_argument('--sample_budget', type=int, default=10, help='repeat n')
    args = p.parse_args()

    print(args)
    print('-'*100)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    train_args = TrainerArgs(
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

    
    if args.dataset == 'CIFAR10':
        train_set, test_set = get_train_valid_cifar10('./data/cifar10', image_size=args.image_size)
        class_num = 10
    if args.dataset == 'CIFAR100':
        train_set, test_set = get_train_valid_cifar100('./data/cifar100', image_size=args.image_size)
        class_num = 100
    if args.dataset == 'food101':
        train_set, test_set = get_train_valid_food101('./data/food101/')
        class_num = 101
    if args.dataset == 'ImageNet':
        train_set, test_set = get_train_valid_ImageNet('/home/rt/share/imagenet/ILSVRC2012', image_size=args.image_size)
        class_num = 1000
    
    if args.model == 'base':
        params = ModelArgs(dim=768, n_layers=12, n_heads=12, output_size=class_num, hidden_dim=3072, image_size=224, tuning_layers=0, per_layer_blocks=4)
    if args.model == 'large':
        params = ModelArgs(dim=1024, n_layers=24, n_heads=16, output_size=class_num, hidden_dim=4096, image_size=224, tuning_layers=0, per_layer_blocks=4)


    # params = ModelArgs(input_chans=3, dim=144, n_layers=6, n_heads=12, output_size=10, hidden_dim=128, image_size=32)
    # params = ModelArgs(dim=64, n_layers=6, n_heads=4, output_size=10, hidden_dim=64, image_size=28)
    # params = ModelArgs(dim=768, n_layers=1, n_heads=12, output_size=100, hidden_dim=258, image_size=224)
    model = Transformer(params)
    if args.model == 'base':
        model_checkpoint = './data/google/vit-base-patch16-224-in21k'
    elif args.model == 'large':
        model_checkpoint = './data/google/vit-large-patch16-224-in21k'
    weights = torch.load(model_checkpoint+'/pytorch_model.bin')
    weight_mapping_vit(model, weights, params)

    # mark_only_classifier_as_trainable(model)
    print_trainable_parameters(model)

    trainer = Trainer(model, train_set, test_set, train_args)
    trainer.train(is_lr=False, lr_type='perturb_outside_all_params', is_parallel=False, is_ddp=False, is_grad_clip=True) # tensor_wise_perturb_reuse
