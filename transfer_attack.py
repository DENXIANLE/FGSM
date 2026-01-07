from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义两个模型的结构 (必须和训练时一致)
# ==========================================

# Model A: LeNet (CNN) - 攻击源模型
class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Model B: MLP (全连接) - 被攻击的目标模型
class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 2. 定义攻击方法
# ==========================================

# 普通 FGSM
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# 频域攻击 (你之前写的)
# 将 transfer_attack.py 中的 freq_attack 函数替换为：

def freq_attack(image, epsilon, data_grad, radius=10):
    # 1. 梯度转频域
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    
    # 2. 获取中心坐标
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    
    # 3. 制作掩膜 (Mask)
    mask = torch.ones_like(grad_freq_shifted)
    
    # 防止 radius 越界，取个最小值
    r = int(min(radius, crow, ccol))
    
    # 如果 r > 0，就挖掉中心的低频部分
    if r > 0:
        mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    
    # 4. 应用掩膜：只保留高频梯度
    grad_freq_high = grad_freq_shifted * mask
    
    # 5. 逆变换回像素域
    grad_freq_high_ishift = torch.fft.ifftshift(grad_freq_high, dim=(-2, -1))
    perturbation = torch.fft.ifftn(grad_freq_high_ishift, dim=(-2, -1)).real
    
    # 6. 生成对抗样本
    perturbed_image = image + epsilon * perturbation.sign()
    
    # 限制在 [0, 1] 范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

# ==========================================
# 3. 迁移测试主逻辑
# ==========================================

# 修改 test_transfer 函数，增加 radius 参数 (默认 None)
def test_transfer(model_source, model_target, device, test_loader, epsilon, attack_type="fgsm", radius=10):
    correct = 0
    total = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model_source(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        if init_pred.item() != target.item():
            continue
            
        loss = F.nll_loss(output, target)
        model_source.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        if attack_type == "fgsm":
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack_type == "freq":
            # 这里传入 radius
            perturbed_data = freq_attack(data, epsilon, data_grad, radius=radius)
            
        output_target = model_target(perturbed_data)
        final_pred = output_target.max(1, keepdim=True)[1]
        
        if final_pred.item() == target.item():
            correct += 1
        total += 1

    acc = correct / float(total)
    return acc
# ==========================================
# 4. 主程序运行
# ==========================================
if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"Running on {device}")

    # 加载数据
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data_row', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=1, shuffle=True)

    # 加载 Model A (Source)
    model_a = NetA().to(device)
    model_a.load_state_dict(torch.load("./data/mnist_cnn.pt", map_location=device))
    model_a.eval()

    # 加载 Model B (Target)
    model_b = NetB().to(device)
    model_b.load_state_dict(torch.load("./data/mnist_mlp.pt", map_location=device))
    model_b.eval()

    print("Models loaded successfully.")
    
    epsilons = [0, 0.1, 0.2, 0.3]
    
    # 定义我们要测试的几种配置
    # 格式: (攻击名称, 类型, 半径)
    configs = [
        ("FGSM (Baseline)", "fgsm", 0),      # 半径对FGSM没用，设为0
        ("Freq (Radius=10)", "freq", 10),    # 你之前的设定
        ("Freq (Radius=5)", "freq", 5),      # 中间值
        ("Freq (Radius=2)", "freq", 2)       # 只有极低频被屏蔽
    ]
    
    results = {name: [] for name, _, _ in configs}

    print(f"{'Epsilon':<10} | {'Method':<20} | {'Accuracy':<10}")
    print("-" * 45)

    for eps in epsilons:
        for name, atype, r in configs:
            acc = test_transfer(model_a, model_b, device, test_loader, eps, attack_type=atype, radius=r)
            results[name].append(acc)
            print(f"{eps:<10} | {name:<20} | {acc:.4f}")

    # 画图
    plt.figure(figsize=(10, 8))
    markers = ['*-', 'o-', 's-', '^-'] # 不同的标记
    
    for (name, _, _), marker in zip(configs, markers):
        plt.plot(epsilons, results[name], marker, label=name)
        
    plt.title("Transferability: Effect of Frequency Mask Radius")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy of Target Model (Net B)")
    plt.legend()
    plt.grid()
    plt.show()