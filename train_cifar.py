import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

def train_cifar():
    # 1. 基础配置
    batch_size = 128
    epochs = 15  # 建议跑 15-20 轮，如果想要更高精度可以设为 30+
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 数据预处理 (CIFAR-10 标准化参数)
    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 数据增强
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    # 3. 定义模型 (ResNet18 适配版)
    print("Building Model (ResNet18)...")
    model = resnet18(pretrained=False)
    
    # === 关键修改 ===
    # ResNet 原本是为 ImageNet (224x224) 设计的，第一层卷积核是 7x7，池化层会把图变太小
    # 针对 CIFAR-10 (32x32)，我们要把第一层改成 3x3，并去掉第一层池化
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # 移除 maxpool
    model.fc = nn.Linear(512, 10) # 最后一层分类改为 10 类
    
    model = model.to(device)

    # 4. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. 开始训练
    print("Start Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        # 每个 Epoch 测一下精度
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Train Acc: {acc:.2f}%")

    # 6. 保存模型
    if not os.path.exists("./data"):
        os.makedirs("./data")
    save_path = "./data/cifar_resnet18.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 7. 最终测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f"Final Test Accuracy: {100.*correct/total:.2f}%")

if __name__ == '__main__':
    train_cifar()