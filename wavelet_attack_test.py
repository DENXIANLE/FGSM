import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# 1. 定义小波变换 (DWT / IDWT)
# ===========================
# 这是一个基于卷积实现的简单 Haar 小波变换，无需安装额外库
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        # Haar 小波核
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.filters = self.filters.float()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # 为了处理多通道，我们需要对每个通道分别卷积
        # 使用 group conv
        filters = self.filters.to(x.device).repeat(c, 1, 1, 1)
        output = F.conv2d(x, filters, stride=2, groups=c)
        return output # shape: [B, C*4, H/2, W/2]

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        # Haar 逆变换核
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        self.filters = torch.stack([ll, lh, hl, hh], dim=1).unsqueeze(0)
        self.filters = self.filters.float()

    def forward(self, x):
        # x: [B, C*4, H/2, W/2]
        b, c_4, h, w = x.shape
        c = c_4 // 4
        filters = self.filters.to(x.device).repeat(c, 1, 1, 1)
        output = F.conv_transpose2d(x, filters, stride=2, groups=c)
        return output

# ===========================
# 2. 攻击算法
# ===========================

# 你的 FFT 方法
def fft_attack(image, epsilon, data_grad):
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask = torch.ones_like(grad_freq_shifted)
    r = 2
    mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    grad_freq_high = grad_freq_shifted * mask
    perturbation = torch.fft.ifftn(torch.fft.ifftshift(grad_freq_high, dim=(-2, -1)), dim=(-2, -1)).real
    return image + epsilon * perturbation.sign()

# 【新】基于 AdvSwap 思想的小波攻击
def dwt_attack(image, epsilon, data_grad):
    dwt = DWT().to(image.device)
    idwt = IDWT().to(image.device)
    
    # 1. 梯度做 DWT
    # Output shape: [B, C*4, H/2, W/2]
    # Channels 排列: [C1_LL, C1_LH, C1_HL, C1_HH, C2_LL, ...]
    grad_dwt = dwt(data_grad)
    
    # 2. 制作掩膜 (屏蔽 LL 低频分量)
    mask = torch.ones_like(grad_dwt)
    c = data_grad.shape[1]
    
    # 将每个通道的第 0 个分量 (LL - 低频近似) 设为 0
    # 对应 AdvSwap 论文中只保留 "high-frequency sub-bands"
    for i in range(c):
        mask[:, i*4, :, :] = 0 
        
    # 3. 应用掩膜并逆变换
    grad_dwt_high = grad_dwt * mask
    perturbation = idwt(grad_dwt_high)
    
    # 4. 生成
    return image + epsilon * perturbation.sign()

# ===========================
# 3. 可视化对比
# ===========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load("./data/cifar_resnet18.pt", map_location=device))
    model = model.to(device).eval()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    # 找一张图
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    
    # 算梯度
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    data_grad = data.grad.data
    
    # 生成
    eps = 0.075 # 统一用 1.5倍 Epsilon
    adv_fft = fft_attack(data, eps, data_grad)
    adv_dwt = dwt_attack(data, eps, data_grad) # AdvSwap 思想
    
    # 准备显示
    def to_img(t): return (t * torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1).to(device) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(device)).clip(0,1).squeeze().cpu().permute(1,2,0)
    
    img_orig = to_img(data)
    img_fft = to_img(adv_fft)
    img_dwt = to_img(adv_dwt)
    
    noise_fft = (img_fft - img_orig) * 10 + 0.5
    noise_dwt = (img_dwt - img_orig) * 10 + 0.5
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 3, 1); plt.imshow(img_orig); plt.title("Original")
    plt.subplot(2, 3, 2); plt.imshow(img_fft); plt.title("FFT Attack (Yours)")
    plt.subplot(2, 3, 3); plt.imshow(img_dwt); plt.title("DWT Attack (AdvSwap-Like)")
    
    plt.subplot(2, 3, 5); plt.imshow(noise_fft); plt.title("FFT Noise")
    plt.subplot(2, 3, 6); plt.imshow(noise_dwt); plt.title("DWT Noise")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()