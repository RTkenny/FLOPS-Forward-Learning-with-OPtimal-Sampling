import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

# from torch.nn import Linear, Embedding, LayerNorm
from .modules import Linear, LayerNorm, Linear_
from torch import nn

# only tuning QV of the transformer

@dataclass
class ModelArgs:
    input_chans: int = 3
    dim: int = 128
    n_layers: int = 6
    n_heads: int = 4
    output_size: int = 1
    hidden_dim: int = 2048
    image_size: int = 224
    norm_eps: float = 1e-5
    patch_size: int = 16
    max_seq_len: int = 2048
    tuning_layers: int = 1
    per_layer_blocks: int = 4


def positional_encoding(d_model, seq_len, device):
    encoding = torch.zeros(seq_len, d_model, device=device)
    encoding.requires_grad = False  # we don't need to compute gradient

    pos = torch.arange(0, seq_len, device=device)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, d_model, step=2, device=device).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    return encoding.unsqueeze(0)


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, input_chans: int,hidden_dim: int, patch_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(input_chans, hidden_dim, kernel_size=patch_size, stride=patch_size)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, pixel_values: torch.Tensor):
        return self.projection(pixel_values).flatten(start_dim=2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, input_chans: int, hidden_dim: int, patch_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.patch_embeddings = ViTPatchEmbeddings(input_chans, hidden_dim, patch_size, image_size)
        num_patches = self.patch_embeddings.num_patches
        # self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.position_embeddings = nn.Parameter(positional_encoding(hidden_dim, num_patches + 1, self.cls_token.device))

    def forward(self, pixel_values: torch.Tensor):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        return embeddings

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_concat = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, add_noise: List[bool] = None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q, add_noise=add_noise[0]), self.w_k(k, add_noise=add_noise[1]), self.w_v(v, add_noise=add_noise[2])

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out, add_noise=add_noise[3])

        # 5. visualize attention map
        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        return tensor

    @staticmethod
    def concat(tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    def backward(self, loss):
        if self.w_q.epsilon_buf is not None:
            self.w_q.backward(loss)
        if self.w_k.epsilon_buf is not None:
            self.w_k.backward(loss)
        if self.w_v.epsilon_buf is not None:
            self.w_v.backward(loss)
        if self.w_concat.epsilon_buf is not None:
            self.w_concat.backward(loss)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
    ):
        super().__init__()
        self.linear1 = Linear(dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, dim)

    def forward(self, x, add_noise=False):
        x = self.linear1(x, add_noise=add_noise)
        x = F.gelu(x)
        # x = F.tanh(x)
        x = self.linear2(x, add_noise=add_noise)
        return x

    def backward(self, loss):
        self.linear1.backward(loss)
        self.linear2.backward(loss)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = MultiHeadAttention(self.dim, self.n_heads)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
        )
        self.layer_id = layer_id
        self.attention_norm = LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = LayerNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor],
            add_noise: List[bool] = None,
    ):  
        x_norm = self.attention_norm(x, add_noise=False)
        h = x + self.attention(
            q=x_norm,
            k=x_norm,
            v=x_norm,
            mask=mask,
            add_noise=add_noise
        )
        h_norm = self.ffn_norm(h, add_noise=False)
        out = h + self.feed_forward(h_norm, add_noise=False)
        return out

    def backward(self, loss, tuning_range: List[str]):
        if 'attention' in tuning_range:
            self.attention.backward(loss)
        if 'feed_forward' in tuning_range:
            self.feed_forward.backward(loss)
        if 'attention_norm' in tuning_range:
            self.attention_norm.backward(loss)
        if 'ffn_norm' in tuning_range:
            self.ffn_norm.backward(loss)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.embeddings = ViTEmbeddings(params.input_chans, params.dim, params.patch_size, params.image_size)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = LayerNorm(params.dim, eps=params.norm_eps)
        self.classifier = Linear(params.dim, params.output_size)

        self.module_len = 1 + params.tuning_layers*params.per_layer_blocks
        self.per_layer_blocks = params.per_layer_blocks
        self.tuning_layers = params.tuning_layers

    def forward(self, input_pixel: torch.Tensor, add_noise: List[bool] = None, **kwargs):
        # there should be at most one true the the add_noise list!
        if add_noise is None:
            add_noise = [False] * self.module_len
        
        if kwargs is not None:
            repeats = kwargs.get('repeats', 50)
            # repeats = kwargs.get('repeats', [50]*self.module_len)
            # if len(repeats) != 1+len(self.layers):
            #     raise ValueError("repeats should have the same length as the model")
        bsz, in_channel, height, width = input_pixel.shape
        h = self.embeddings(input_pixel)
        bsz, seqlen, dim = h.shape

        mask = None
        j=0
        for i, layer in enumerate(self.layers):
            if i >= len(self.layers)-self.tuning_layers:
                tmp_add_noise = add_noise[j*self.per_layer_blocks:(j+1)*self.per_layer_blocks]
                if True in tmp_add_noise:
                    # repeat interleave
                    h = h.repeat(repeats, 1, 1)
                j += 1
            else:
                tmp_add_noise = [False]*self.per_layer_blocks

            # print(h.shape)
            h = layer(h, mask, add_noise=tmp_add_noise)
            
        h = self.norm(h, add_noise = False)
        cls_token = h[:, 0, :]

        if add_noise[-1]:
            # cls_token = cls_token.repeat_interleave(repeats, dim=0)
            cls_token = cls_token.repeat(repeats, 1)

        logits = self.classifier(cls_token, add_noise=add_noise[-1]).float()
        return logits, None

    def backward(self, loss, tuning_range: List[str]=None):
        if tuning_range is None:
            tuning_range = ['attention']
        for layer in self.layers:
            layer.backward(loss, tuning_range)
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 0.05)
        if self.classifier.epsilon_buf is not None:
            self.classifier.backward(loss)
