import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split


#going to be trianed on CIFAR10 - (3 x 32 x 32)
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
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

        
        self.pool = nn.MaxPool2d(2)
    
        self.fc1 = nn.Linear(
            32 * 8 * 8,
            10
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
        x = self.fc1(x)
        return x
    
    def train(self, batch_size, dataloader):
        losses = []
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx > batch_size:
                break
            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = model(images)
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



    def evaluate(self, batch_size, test_dataloader):
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_dataloader):
                if batch_idx > batch_size:
                    break
                images = images.to(device)#send to gpu
                labels = labels.to(device)

                pred = model(images)
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

model = MyCNN()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)

opt = optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.CrossEntropyLoss()
batch_size = 64

print(next(model.parameters()).device)
for epoch in range(20):
    print(f"Round{epoch+1}")
    
    loss, acc = model.train(batch_size, dataloader)
    print(f"Loss: {loss}, Acc: {acc}")
