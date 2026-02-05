import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

# 定义模型 B：一个简单的全连接网络 (MLP)
# 这与之前的 LeNet (卷积网络) 结构完全不同
class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        # MNIST 图片大小是 28x28 = 784
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784) # 把图片展平成一维向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_and_save():
    # 基础设置
    batch_size = 64
    epochs = 5 # 跑5轮就够用了
    lr = 0.01
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # 数据加载 (保持和之前一致)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data_row', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型 B
    model = NetB().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 开始训练
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 500 == 0:
                print(f'Train Epoch: {epoch} \tLoss: {loss.item():.6f}')

    # 保存模型
    if not os.path.exists("./data"):
        os.makedirs("./data")
    save_path = "./data/mnist_mlp.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model B (MLP) saved to {save_path}")

if __name__ == '__main__':
    train_and_save()