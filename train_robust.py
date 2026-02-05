import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, 
    vgg11_bn, 
    mobilenet_v2, 
    shufflenet_v2_x1_0, 
    densenet121, 
    googlenet
)
import os
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
# 在这里填入你想训练的模型名字
# 可选: "resnet", "vgg", "mobilenet", "shufflenet", "densenet", "googlenet"
MODELS_TO_TRAIN = ["shufflenet", "mobilenet"] 

EPOCHS = 15          # 训练轮数 (15轮通常足够用于迁移测试)
BATCH_SIZE = 128     # 批次大小
LEARNING_RATE = 0.01 # 学习率

# ==========================================
# 2. 模型工厂 (Model Factory)
# ==========================================
def get_model(name, device):
    name = name.lower().strip() # 自动转小写并去空格，防止手误
    model = None

    if name == "resnet":
        print(f"🏗️  Initializing ResNet-18...")
        model = resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)

    elif name == "vgg":
        print(f"🏗️  Initializing VGG-11 (BN)...")
        model = vgg11_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)

    elif name == "mobilenet":
        print(f"🏗️  Initializing MobileNet-V2...")
        model = mobilenet_v2(pretrained=False)
        # 适配 CIFAR-10 (32x32)
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.last_channel, 10)

    elif name == "shufflenet":
        print(f"🏗️  Initializing ShuffleNet-V2...")
        model = shufflenet_v2_x1_0(pretrained=False)
        # 适配 CIFAR-10
        out_channels = model.conv1[0].out_channels
        model.conv1[0] = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    
    elif name == "densenet":
        print(f"🏗️  Initializing DenseNet-121 (Warning: Slow)...")
        model = densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(1024, 10)

    elif name == "googlenet":
        print(f"🏗️  Initializing GoogLeNet...")
        model = googlenet(pretrained=False)
        model.aux_logits = False
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool1 = nn.Identity()
        model.fc = nn.Linear(1024, 10)

    else:
        raise ValueError(f"❌ 错误：不支持模型名称 '{name}'。请检查拼写。")

    return model.to(device)

# ==========================================
# 3. 辅助功能：画图 (Plotting)
# ==========================================
def plot_history(name, train_acc, train_loss):
    plt.figure(figsize=(10, 5))
    
    # 画 Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Acc', color='green')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # 画 Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss', color='red')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    # 保存图片
    if not os.path.exists("./plots"): os.makedirs("./plots")
    plt.savefig(f"./plots/training_{name}.png")
    plt.close()
    print(f"📊 训练曲线已保存至 ./plots/training_{name}.png")

# ==========================================
# 4. 训练主逻辑
# ==========================================
def train_one_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*40}")
    print(f"🚀 开始训练: {model_name.upper()} on {device}")
    print(f"{'='*40}")

    # 数据准备
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 使用 download=True 自动处理
    trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    try:
        model = get_model(model_name, device)
    except ValueError as e:
        print(e)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 记录数据用于画图
    history_acc = []
    history_loss = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
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
        
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(trainloader)
        
        history_acc.append(epoch_acc)
        history_loss.append(epoch_loss)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"✅ {model_name} 训练完成! 耗时: {total_time/60:.1f} 分钟")

    # 保存模型
    if not os.path.exists("./data"): os.makedirs("./data")
    save_path = f"./data/cifar_{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"💾 模型已保存: {save_path}")

    # 画图
    plot_history(model_name, history_acc, history_loss)

if __name__ == '__main__':
    # 循环训练配置列表中的所有模型
    for name in MODELS_TO_TRAIN:
        train_one_model(name)
    
    print("\n🎉 所有任务全部完成！请运行 multi_model_test.py 进行测试。")