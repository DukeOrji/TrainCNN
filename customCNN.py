import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST
from sklearn.model_selection import train_test_split


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            16,
            32,
            3,
            padding=1
        )

        
        self.fc = nn.Sequential(
            nn.Linear(1568, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.pool(
            torch.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            torch.relu(
                self.conv2(x)
            )
        )

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def fit(self, num_batch, dataloader):
        losses = []
        correct = 0
        total = 0
        self.train()
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx > num_batch:
                break
            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = self(images)
            loss = loss_fn(pred, labels)
            pred_label = pred.argmax(dim=1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            avg_loss = round(sum(losses)/len(losses), 3)
            correct += (pred_label == labels).sum().item()
            total += len(labels)

            acc = correct/total
        return avg_loss, acc



    def evaluate(self, num_batch, test_dataloader):
        losses = []
        correct = 0
        total = 0
        self.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_dataloader):
                if batch_idx > num_batch:
                    break
                images = images.to(device)#send to gpu
                labels = labels.to(device)

                pred = self(images)
                loss = loss_fn(pred, labels)
                pred_label = pred.argmax(dim=1)

                
                losses.append(loss.item())
                avg_loss = round(sum(losses)/len(losses), 3)
                correct += (pred_label == labels).sum().item()
                total += len(labels)

                acc = correct/total
        return avg_loss, acc
    

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

model = MyCNN()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)

opt = optim.Adam(
    model.parameters(),
    lr=1e-2
)
loss_fn = nn.CrossEntropyLoss()
num_batch = 64

#load previous model parameters
if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    opt.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
else:
    print("No checkpoint found. Starting from scratch.")

#0 - train
#1 - test
switch = 0

print(next(model.parameters()).device)
for epoch in range(5):
    print(f"Round{epoch+1}")
    
    if switch == 0:
        loss, acc = model.fit(num_batch, dataloader)
    else:
        loss, acc = model.evaluate(num_batch, test_dataloader)
    print(f"Loss: {loss}, Acc: {acc}")

#save model parameters
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict()
}, "checkpoint.pth")