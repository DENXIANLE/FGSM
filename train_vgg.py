import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11_bn
import os

def train_vgg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 定义模型：VGG-11 (带 Batch Norm)
    print("Building VGG-11...")
    model = vgg11_bn(pretrained=False)
    # 修改输出层为 10 类
    model.classifier[6] = nn.Linear(4096, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30) # 简化的训练轮数

    print("Start Training VGG...")
    epochs = 10 # 为了节省时间，先跑10轮，精度能到80%左右即可用于测试
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f}")

    # 保存模型
    if not os.path.exists("./data"):
        os.makedirs("./data")
    torch.save(model.state_dict(), "./data/cifar_vgg11.pt")
    print("VGG Model Saved!")

if __name__ == '__main__':
    train_vgg()