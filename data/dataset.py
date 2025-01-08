import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def calculate_mean_std(dataset):
    """
    计算数据集的均值和方差
    :param dataset: 数据集
    :return: mean, std
    """
    # init mean and std
    mean = 0.
    std = 0.
    total_samples = len(dataset)
    print("dataset len:", total_samples)


    # Traversing the dataset calculates the mean and std
    for data, _ in dataset:
        """
        data张量是一个代表图像的张量，通常具有三个维度：(通道, 高度, 宽度)。
        dim=(1, 2)参数指定了要在高度和宽度维度上进行求均值的操作。
        计算每个通道上的像素值的平均值，得到的结果是一个包含每个通道上的平均值的张量。
        """
        mean += torch.mean(data, dim=(1, 2))
        std += torch.std(data, dim=(1, 2))

    # Calculate the population mean and std
    mean /= total_samples
    std /= total_samples

    return mean, std


def get_train_valid_MNIST(data_dir, download=True, image_size=28):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    valid_set = datasets.MNIST(root=data_dir, train=False, transform=transform)

    return train_set, valid_set

# cifar 10
def get_train_valid_cifar10(data_dir, download=True, image_size=32):
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize((image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    transform_valid = transforms.Compose([transforms.Resize((image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])

    train_set = datasets.CIFAR10(root=data_dir, train=True, transform=transform_train, download=download)
    valid_set = datasets.CIFAR10(root=data_dir, train=False, transform=transform_valid, download=False)

    return train_set, valid_set


# cifar 100
def get_train_valid_cifar100(data_dir, download=True, image_size=32):
    cifar_norm_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_norm_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize((image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    transform_valid = transforms.Compose([transforms.Resize((image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])

    train_set = datasets.CIFAR100(root=data_dir, train=True, transform=transform_train, download=download)
    valid_set = datasets.CIFAR100(root=data_dir, train=False, transform=transform_valid, download=False)

    return train_set, valid_set

def get_train_valid_flower102(data_dir, batch_size=32):
    # Load the Flower102 dataset


    # Define transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = datasets.ImageFolder(root=os.path.join(data_dir, 'dataset/train'), transform=train_transforms)
    val_ds = datasets.ImageFolder(root=os.path.join(data_dir, 'dataset/valid'), transform=val_transforms)
    # # Apply transformations
    # dataset.transform = train_transforms
    #
    # # Split the dataset into training and validation sets
    # train_size = int(0.9 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_ds, val_ds = random_split(dataset, [train_size, val_size])
    #
    # # Apply validation transforms
    # val_ds.dataset.transform = val_transforms

    # # Create DataLoaders
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds

def get_train_valid_eurosat(data_dir, batch_size=32):
    # Load the EuroSAT dataset
    dataset = datasets.ImageFolder(root=data_dir)

    # Define transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Apply transformations
    dataset.transform = train_transforms

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Apply validation transforms
    val_ds.dataset.transform = val_transforms

    # Create DataLoaders
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds

def get_train_valid_food101(data_dir):
    dataset = load_dataset(data_dir, split="train[:10000]", num_proc=8)
    normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transforms = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    splits = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    datas, labels = [], []
    for item in train_ds:
        print(item)
        datas.append(item['pixel_values'])
        labels.append(item['label'])
    train_dst = myDataset(datas, labels)

    datas, labels = [], []
    for item in val_ds:
        datas.append(item['pixel_values'])
        labels.append(item['label'])
    val_dst = myDataset(datas, labels)

    return train_dst, val_dst

def get_train_valid_caltech101(data_dir, download=True):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    data_set = datasets.Caltech101(root=data_dir, transform=transform_train, download=download)
    # valid_set = datasets.Caltech101(root=data_dir, train=False, transform=transform_valid, download=False)

def get_train_valid_caltech256(data_dir, download=True):
    transform_caltech256 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    date_set = datasets.Caltech256(root=data_dir, transform=transform_caltech256, download=download)
    train_set, test_set = random_split(date_set, [int(len(date_set) * 0.8), int(len(date_set) * 0.2)])
    
def get_train_valid_ImageNet(data_dir, image_size=224):
    # 定义训练集的预处理
    train_transforms = transforms.Compose([
        transforms.Resize(256),                # 调整图像大小为256x256
        transforms.RandomResizedCrop(image_size),     # 随机裁剪224x224的图像
        transforms.RandomHorizontalFlip(),     # 随机水平翻转
        transforms.ToTensor(),                 # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化：ImageNet的均值和标准差
                            std=[0.229, 0.224, 0.225])
    ])

    # 定义测试集的预处理
    test_transforms = transforms.Compose([
        transforms.Resize(256),                # 调整图像大小为256x256
        transforms.CenterCrop(image_size),            # 中心裁剪224x224的图像
        transforms.ToTensor(),                 # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化：ImageNet的均值和标准差
                            std=[0.229, 0.224, 0.225])
    ])


    train_set = datasets.ImageFolder(root = data_dir + '/train/', transform = train_transforms)
    val_set = datasets.ImageFolder(root = data_dir + '/val/', transform = test_transforms)
    return train_set, val_set


if __name__ == '__main__':
    # train_data = datasets.Caltech256("./data/", download=True)
    # print(train_data)
    # dataset = load_dataset('./data/caltech256')
    # print(dataset)

    # data_dir = './data/cifar100'
    # trainloader, testloader = get_train_valid_cifar100(data_dir, True)

    # transform_valid = transforms.Compose([transforms.ToTensor(),
    #                                       # transforms.Normalize([0.4942, 0.4851, 0.4504], [0.2020, 0.1991, 0.2011])
    #                                      ]
    #                                      )

    # transform_valid = transforms.Compose([transforms.ToTensor(),
    #                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #                                       ]
    #                                      )
    # dataset = datasets.CIFAR10(root='./data/cifar10', train=True, transform=transform_valid, download=True)
    # dataset = datasets.Food101(root='./data/', split='train', transform=transform_valid, download=True)

    # dataset = load_dataset('./data/caltech256/', split='train')
    # count = 0
    # dataset.set_transform(transform_valid)
    # print(len(dataset))
    # loader = DataLoader(dataset, batch_size=14)
    # for item in loader:
    #     print(item)
    #     count += 1
    #     if count ==3:
    #         break

    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # 将图片转换为Tensor
    #     transforms.Normalize((0.1307,), (0.3081,))  # 归一化
    # ])
    #
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Dataset()
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    #
    # print(next(iter(train_loader)))

    # get_train_valid_caltech101('./data/', True)


    import time
    time.sleep(1)
    a, b = get_train_valid_ImageNet('/home/rt/data/ILSVRC2012')
    time.sleep(10)