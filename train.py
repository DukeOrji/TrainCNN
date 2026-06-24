import torch
import torch.nn as nn
import torch.optim as optim
from customShuffleNet import ShuffleNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ShuffleNet()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)


def fit(num_batch, dataloader):
    losses = []
    correct = 0
    total = 0
    model.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx > num_batch:
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



def evaluate(num_batch, test_dataloader):
    losses = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            if batch_idx > num_batch:
                break
            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = model(images)
            loss = loss_fn(pred, labels)
            pred_label = pred.argmax(dim=1)

            
            losses.append(loss.item())
            avg_loss = round(sum(losses)/len(losses), 3)
            correct += (pred_label == labels).sum().item()
            total += len(labels)

            acc = correct/total
    return avg_loss, acc