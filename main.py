import torch
import torch.nn as nn
import torch.optim as optim

from ds import load_cifar, load_mnist
from train import fit, evaluate, model

test_dataloader, dataloader = load_cifar()
num_batch = 64
switch = 0

print(next(model.parameters()).device)
for epoch in range(5):
    print(f"Round{epoch+1}")
    
    if switch == 0:
        loss, acc = fit(num_batch, dataloader)
        
    else:
        loss, acc = evaluate(num_batch, test_dataloader)
    print(f"Loss: {loss}, Acc: {acc}")