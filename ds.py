from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST




def load_mnist():

    preprocess = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor()
    ])

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=preprocess
    )

    dataloader = DataLoader(
        mnist_train,
        batch_size=32,
        shuffle=True
    )

    mnist_test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=preprocess
    )

    test_dataloader = DataLoader(
        mnist_test,
        batch_size=32,
        shuffle=True
    )

    return test_dataloader, dataloader

def load_cifar():

    preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=preprocess
    )

    dataloader = DataLoader(
        cifar_train,
        batch_size=32,
        shuffle=True
    )

    cifar_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=preprocess
    )

    test_dataloader = DataLoader(
        cifar_test,
        batch_size=32,
        shuffle=False
    )

    return test_dataloader, dataloader