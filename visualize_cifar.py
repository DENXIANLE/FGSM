import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 模型定义 (必须与训练代码完全一致)
# ==========================================
def get_cifar_model(device):
    # 加载标准 ResNet18
    model = resnet18(pretrained=False)
    # 修改第一层 (适配 32x32 输入)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 修改全连接层 (10类)
    model.fc = nn.Linear(512, 10)
    
    # 加载你训练好的权重
    try:
        model.load_state_dict(torch.load("./data/cifar_resnet18.pt", map_location=device))
        print("Success: Model loaded!")
    except FileNotFoundError:
        print("Error: 找不到模型文件，请确认 train_cifar.py 是否运行成功")
        exit()
        
    return model.to(device)

# ==========================================
# 2. 攻击算法
# ==========================================
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image # 注意：这里 clamp 并不严格准确因为有归一化，但在可视化时影响不大

def freq_attack(image, epsilon, data_grad, radius=5):
    # 梯度转频域 (自动处理 RGB 3通道)
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    
    # 制作掩膜
    mask = torch.ones_like(grad_freq_shifted)
    r = int(min(radius, crow, ccol))
    if r > 0:
        mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    
    # 应用掩膜
    grad_freq_high = grad_freq_shifted * mask
    grad_freq_high_ishift = torch.fft.ifftshift(grad_freq_high, dim=(-2, -1))
    perturbation = torch.fft.ifftn(grad_freq_high_ishift, dim=(-2, -1)).real
    
    perturbed_image = image + epsilon * perturbation.sign()
    return perturbed_image

# ==========================================
# 3. 辅助函数：反归一化 (用于显示图片)
# ==========================================
def unnormalize(tensor):
    # CIFAR-10 的 mean 和 std
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# CIFAR-10 类别名称
classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# ==========================================
# 4. 主程序
# ==========================================
def main():
    epsilon = 0.1  # CIFAR-10 稍微小一点 epsilon 效果更真实 (通常 0.03 - 0.1)
    radius = 4      # 频域半径
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载 (必须包含 Normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 只需要 Test Set
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    model = get_cifar_model(device)
    model.eval()

    # 找一张分类正确的图片来攻击
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        if init_pred.item() == target.item():
            break # 找到了，跳出循环
            
    data.requires_grad = True

    output = model(data)

    # --- 执行攻击 ---
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data

    # 1. FGSM
    adv_fgsm = fgsm_attack(data, epsilon, data_grad)
    out_fgsm = model(adv_fgsm)
    pred_fgsm = out_fgsm.max(1)[1].item()
    
    # 2. Freq Attack
    adv_freq = freq_attack(data, epsilon, data_grad, radius=radius)
    out_freq = model(adv_freq)
    pred_freq = out_freq.max(1)[1].item()

    # --- 准备画图 ---
    # 先把图片从 Tensor 转回正常的 RGB 格式
    img_orig = unnormalize(data).detach().cpu().squeeze().permute(1, 2, 0).clip(0, 1)
    img_fgsm = unnormalize(adv_fgsm).detach().cpu().squeeze().permute(1, 2, 0).clip(0, 1)
    img_freq = unnormalize(adv_freq).detach().cpu().squeeze().permute(1, 2, 0).clip(0, 1)
    
    # 计算噪点 (放大 10 倍显示，否则可能太隐蔽看不清)
    noise_fgsm = (img_fgsm - img_orig) * 10 + 0.5 
    noise_freq = (img_freq - img_orig) * 10 + 0.5
    
    # --- 画图 ---
    plt.figure(figsize=(12, 7))
    
    titles = [
        f"Original\n({classes[target.item()]})", 
        f"FGSM (Eps={epsilon})\nPred: {classes[pred_fgsm]}", 
        "FGSM Noise (x10)",
        f"Original\n({classes[target.item()]})", 
        f"Freq Attack (R={radius})\nPred: {classes[pred_freq]}", 
        "Freq Noise (x10)"
    ]
    
    images = [img_orig, img_fgsm, noise_fgsm, img_orig, img_freq, noise_freq]
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()