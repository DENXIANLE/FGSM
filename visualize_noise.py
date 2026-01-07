from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 模型结构 (必须保持一致)
# ==========================================
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

# ==========================================
# 2. 攻击函数
# ==========================================
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def freq_attack(image, epsilon, data_grad, radius=5):
    # 梯度转频域
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    
    # 制作掩膜 (挖掉低频)
    mask = torch.ones_like(grad_freq_shifted)
    r = int(min(radius, crow, ccol))
    if r > 0:
        mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    
    # 应用掩膜
    grad_freq_high = grad_freq_shifted * mask
    grad_freq_high_ishift = torch.fft.ifftshift(grad_freq_high, dim=(-2, -1))
    perturbation = torch.fft.ifftn(grad_freq_high_ishift, dim=(-2, -1)).real
    
    # 生成样本
    perturbed_image = image + epsilon * perturbation.sign()
    return torch.clamp(perturbed_image, 0, 1)

# ==========================================
# 3. 可视化主程序
# ==========================================
def main():
    # 设置参数
    epsilon = 0.2  # 稍微大一点，方便肉眼看清噪点
    radius = 5     # 使用你实验得出的最佳半径
    
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 加载数据 (只取第一张图)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data_row', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=1, shuffle=True)

    # 加载模型
    model = NetA().to(device)
    model.load_state_dict(torch.load("./data/mnist_cnn.pt", map_location=device))
    model.eval()

    # 获取一张图片
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    data.requires_grad = True

    # ---------------------------
    # 执行攻击
    # ---------------------------
    # 1. 计算梯度
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # 原始预测
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data

    # 2. 生成 FGSM 样本
    adv_fgsm = fgsm_attack(data, epsilon, data_grad)
    output_fgsm = model(adv_fgsm)
    pred_fgsm = output_fgsm.max(1, keepdim=True)[1] # FGSM 预测

    # 3. 生成 频域 样本 (Radius=5)
    adv_freq = freq_attack(data, epsilon, data_grad, radius=radius)
    output_freq = model(adv_freq)
    pred_freq = output_freq.max(1, keepdim=True)[1] # 频域 预测

    # ---------------------------
    # 计算噪点 (Perturbation)
    # ---------------------------
    # 噪点 = 对抗样本 - 原始图片
    noise_fgsm = (adv_fgsm - data).squeeze().detach().cpu().numpy()
    noise_freq = (adv_freq - data).squeeze().detach().cpu().numpy()
    
    # 准备画图数据
    img_orig = data.squeeze().detach().cpu().numpy()
    img_fgsm = adv_fgsm.squeeze().detach().cpu().numpy()
    img_freq = adv_freq.squeeze().detach().cpu().numpy()

    # ---------------------------
    # 画图
    # ---------------------------
    plt.figure(figsize=(12, 8))

    # 第一行：FGSM
    plt.subplot(2, 3, 1)
    plt.title(f"Original (Pred: {init_pred.item()})")
    plt.imshow(img_orig, cmap="gray")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(f"FGSM Attack (Pred: {pred_fgsm.item()})")
    plt.imshow(img_fgsm, cmap="gray")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("FGSM Noise (Perturbation)")
    # 使用 seismic 颜色映射，并在0处居中，方便看正负噪点
    plt.imshow(noise_fgsm, cmap="seismic", vmin=-epsilon, vmax=epsilon)
    plt.colorbar()
    plt.axis('off')

    # 第二行：Freq Attack
    plt.subplot(2, 3, 4)
    plt.title(f"Original (Pred: {init_pred.item()})")
    plt.imshow(img_orig, cmap="gray")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title(f"Freq Attack (R={radius}) (Pred: {pred_freq.item()})")
    plt.imshow(img_freq, cmap="gray")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title(f"Freq Noise (R={radius})")
    plt.imshow(noise_freq, cmap="seismic", vmin=-epsilon, vmax=epsilon)
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()