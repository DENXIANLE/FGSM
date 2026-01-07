import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import os

# ===========================
# 1. 稳健的 Haar 小波变换
# ===========================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        # Haar Filters
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        # Shape: [4, 1, 2, 2]
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.filters = self.filters.float()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # 为了让每个通道独立进行DWT，我们需要把 batch 和 channel 合并
        # [B*C, 1, H, W]
        x_reshaped = x.view(b * c, 1, h, w)
        
        filters = self.filters.to(x.device)
        # Conv output: [B*C, 4, H/2, W/2]
        output = F.conv2d(x_reshaped, filters, stride=2, padding=0)
        
        # Reshape back: [B, C, 4, H/2, W/2] -> [B, C*4, H/2, W/2]
        output = output.view(b, c * 4, h // 2, w // 2)
        return output

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        # Haar Inverse Filters (与 DWT 对称)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        # Shape: [4, 1, 2, 2] -> IDWT weight shape is [in_channels, out_channels, k, k]
        # 对于 conv_transpose2d, weight shape 是 [in, out/groups, k, k]
        # 这里 in=4, out=1, groups=1 (我们依然用 reshape trick)
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.filters = self.filters.float()

    def forward(self, x):
        # x: [B, C*4, H/2, W/2]
        b, c_4, h_2, w_2 = x.shape
        c = c_4 // 4
        
        # Reshape to [B*C, 4, H/2, W/2]
        x_reshaped = x.view(b * c, 4, h_2, w_2)
        
        filters = self.filters.to(x.device)
        
        # Conv Transpose output: [B*C, 1, H, W]
        output = F.conv_transpose2d(x_reshaped, filters, stride=2, padding=0)
        
        # Reshape back: [B, C, H, W]
        output = output.view(b, c, h_2 * 2, w_2 * 2)
        return output

# ===========================
# 2. 攻击算法
# ===========================

# FFT Attack (你的旧方法)
def fft_attack(image, epsilon, data_grad):
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    
    # Mask
    mask = torch.ones_like(grad_freq_shifted)
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    r = 2
    mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    
    grad_freq_high = grad_freq_shifted * mask
    perturbation = torch.fft.ifftn(torch.fft.ifftshift(grad_freq_high, dim=(-2, -1)), dim=(-2, -1)).real
    return image + epsilon * perturbation.sign()

# DWT Attack (AdvSwap 思想)
def dwt_attack(image, epsilon, data_grad):
    dwt = DWT().to(image.device)
    idwt = IDWT().to(image.device)
    
    # 1. DWT: [B, 12, H/2, W/2]
    grad_dwt = dwt(data_grad)
    
    # 2. Mask: 只要 LH, HL, HH，扔掉 LL
    # grad_dwt 的通道排列是 [R_LL, R_LH, R_HL, R_HH, G_LL, ...]
    # 每 4 个一组，第 0 个是 LL
    mask = torch.ones_like(grad_dwt)
    c = data_grad.shape[1] # 3
    for i in range(c):
        mask[:, i*4, :, :] = 0 # 挖掉每个通道的 LL 分量
        
    grad_dwt_high = grad_dwt * mask
    
    # 3. IDWT
    perturbation = idwt(grad_dwt_high)
    
    return image + epsilon * perturbation.sign()

# ===========================
# 3. 准备模型与显示
# ===========================
def get_model(device):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    path = "./data/cifar_resnet18.pt"
    if not os.path.exists(path): path = "./data/cifar_resnet.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def unnormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# ===========================
# 4. 主程序
# ===========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")
    
    model = get_model(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    # 找一张图
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    data_grad = data.grad.data
    
    # 生成
    eps = 0.05
    # FFT (你的)
    adv_fft = fft_attack(data, eps, data_grad)
    # DWT (AdvSwap)
    adv_dwt = dwt_attack(data, eps, data_grad)
    
    # 绘图
    img_orig = unnormalize(data).detach().cpu().squeeze().permute(1, 2, 0).clip(0, 1)
    img_fft = unnormalize(adv_fft).detach().cpu().squeeze().permute(1, 2, 0).clip(0, 1)
    img_dwt = unnormalize(adv_dwt).detach().cpu().squeeze().permute(1, 2, 0).clip(0, 1)
    
    noise_fft = (img_fft - img_orig) * 10 + 0.5
    noise_dwt = (img_dwt - img_orig) * 10 + 0.5
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1); plt.imshow(img_orig); plt.title("Original")
    plt.subplot(2, 3, 2); plt.imshow(img_fft); plt.title("FFT Attack (Baseline)")
    plt.subplot(2, 3, 3); plt.imshow(img_dwt); plt.title("DWT Attack (AdvSwap)")
    
    plt.subplot(2, 3, 5); plt.imshow(noise_fft); plt.title("FFT Noise")
    plt.subplot(2, 3, 6); plt.imshow(noise_dwt); plt.title("DWT Noise")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()