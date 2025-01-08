import math
import numpy as np
import torch
import random
import torch.nn as nn
from typing import OrderedDict
from models.my_vit_modeling import ModelArgs


def weight_mapping_vit(model: nn.Module, weights: OrderedDict, params: ModelArgs):
    model_state_dict = model.state_dict()
    model_state_dict['embeddings.cls_token'] = weights['embeddings.cls_token']
    model_state_dict['embeddings.position_embeddings'] = weights['embeddings.position_embeddings']
    model_state_dict['embeddings.patch_embeddings.projection.weight'] = weights['embeddings.patch_embeddings.projection.weight']
    model_state_dict['embeddings.patch_embeddings.projection.bias'] = weights['embeddings.patch_embeddings.projection.bias']
    model_state_dict['norm.weight'] = weights['layernorm.weight']
    model_state_dict['norm.bias'] = weights['layernorm.bias']

    for i in range(params.n_layers):
        model_state_dict[f'layers.{i}.attention.w_q.weight'] = weights[f'encoder.layer.{i}.attention.attention.query.weight']
        model_state_dict[f'layers.{i}.attention.w_q.bias'] = weights[f'encoder.layer.{i}.attention.attention.query.bias']
        model_state_dict[f'layers.{i}.attention.w_k.weight'] = weights[f'encoder.layer.{i}.attention.attention.key.weight']
        model_state_dict[f'layers.{i}.attention.w_k.bias'] = weights[f'encoder.layer.{i}.attention.attention.key.bias']
        model_state_dict[f'layers.{i}.attention.w_v.weight'] = weights[f'encoder.layer.{i}.attention.attention.value.weight']
        model_state_dict[f'layers.{i}.attention.w_v.bias'] = weights[f'encoder.layer.{i}.attention.attention.value.bias']
        model_state_dict[f'layers.{i}.attention.w_concat.weight'] = weights[f'encoder.layer.{i}.attention.output.dense.weight']
        model_state_dict[f'layers.{i}.attention.w_concat.bias'] = weights[f'encoder.layer.{i}.attention.output.dense.bias']

        model_state_dict[f'layers.{i}.feed_forward.linear1.weight'] = weights[f'encoder.layer.{i}.intermediate.dense.weight']
        model_state_dict[f'layers.{i}.feed_forward.linear1.bias'] = weights[f'encoder.layer.{i}.intermediate.dense.bias']
        model_state_dict[f'layers.{i}.feed_forward.linear2.weight'] = weights[f'encoder.layer.{i}.output.dense.weight']
        model_state_dict[f'layers.{i}.feed_forward.linear2.bias'] = weights[f'encoder.layer.{i}.output.dense.bias']

        model_state_dict[f'layers.{i}.attention_norm.weight'] = weights[f'encoder.layer.{i}.layernorm_before.weight']
        model_state_dict[f'layers.{i}.attention_norm.bias'] = weights[f'encoder.layer.{i}.layernorm_before.bias']
        model_state_dict[f'layers.{i}.ffn_norm.weight'] = weights[f'encoder.layer.{i}.layernorm_after.weight']
        model_state_dict[f'layers.{i}.ffn_norm.bias'] = weights[f'encoder.layer.{i}.layernorm_after.bias']

    model.load_state_dict(model_state_dict)

def weight_mapping_vit_embeddings(model: nn.Module, weights: OrderedDict, params: ModelArgs):
    model_state_dict = model.state_dict()
    model_state_dict['embeddings.cls_token'] = weights['embeddings.cls_token']
    model_state_dict['embeddings.position_embeddings'] = weights['embeddings.position_embeddings']
    model_state_dict['embeddings.patch_embeddings.projection.weight'] = weights['embeddings.patch_embeddings.projection.weight']
    model_state_dict['embeddings.patch_embeddings.projection.bias'] = weights['embeddings.patch_embeddings.projection.bias']

    model.load_state_dict(model_state_dict)


def mark_only_classifier_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'classifier' not in n:
            p.requires_grad = False


def print_trainable_parameters(model: nn.Module) -> None:
    count = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
            count += p.numel()
    print('Total number of trainable parameters: {}'.format(count))


def print_modules_grad(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(n, p.grad)
        else:
            print(n, 'None')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # params = ModelArgs(dim=768, n_layers=12, n_heads=12, output_size=101, hidden_dim=3072)
    # model = Transformer(params)
    # model_checkpoint = './data/google/vit-base-patch16-224-in21k'
    # weights = torch.load(model_checkpoint + '/pytorch_model.bin')
    # weight_mapping(model, weights, params)
    # weight1 = model.state_dict()

    model_checkpoint = './data/FacebookAI/roberta-base'
    weights = torch.load(model_checkpoint + '/pytorch_model.bin')

