import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 基础配置 (针对 ImageNet 高清大图)
# ==========================================
# ImageNet 标准参数
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

# 复用 DWT/IDWT 定义 (Haar 小波)
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).float()

    def forward(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        filters = self.filters.to(x.device)
        return F.conv2d(x_reshaped, filters, stride=2).view(b, c * 4, h // 2, w // 2)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).float()

    def forward(self, x):
        b, c4, h2, w2 = x.shape
        c = c4 // 4
        x_reshaped = x.view(b * c, 4, h2, w2)
        filters = self.filters.to(x.device)
        return F.conv_transpose2d(x_reshaped, filters, stride=2).view(b, c, h2 * 2, w2 * 2)

def get_hvs_mask(image):
    """计算大图的边缘掩码"""
    x_grad = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    y_grad = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    grad_mag = torch.zeros_like(image)
    grad_mag[:, :, :, :-1] += x_grad
    grad_mag[:, :, :-1, :] += y_grad
    grad_mag = grad_mag.mean(dim=1, keepdim=True)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    # 大图上边缘更稀疏，可以适当调强掩码对比度
    mask = 0.3 + 0.7 * grad_mag 
    return mask

# ==========================================
# 2. 攻击算法实现 (MI-FGSM vs MI-SSA-DWT)
# ==========================================
def mifgsm_attack(model, image, target, epsilon, num_iter=10):
    alpha = epsilon / num_iter
    momentum = torch.zeros_like(image)
    adv_image = image.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        output = model(normalize(adv_image))
        loss = F.cross_entropy(output, target)
        model.zero_grad(); loss.backward()
        grad = adv_image.grad.data
        grad = grad / (torch.norm(grad, p=1, dim=(1,2,3), keepdim=True) + 1e-10)
        momentum = 1.0 * momentum + grad
        adv_image = torch.clamp(adv_image + alpha * momentum.sign(), 0, 1).detach().requires_grad_(True)
    return adv_image.detach()

def mi_ssa_dwt_attack(model, image, target, epsilon, num_iter=20, eta=0.3):
    dwt, idwt = DWT().to(image.device), IDWT().to(image.device)
    hvs_mask = get_hvs_mask(image).to(image.device)
    alpha = epsilon / num_iter
    momentum = torch.zeros_like(image)
    adv_image = image.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        # SSA 增强
        x_freq = dwt(adv_image)
        rho = (1 - eta) + (2 * eta) * torch.rand(x_freq.shape[0], x_freq.shape[1], 1, 1).to(image.device)
        mask_freq = torch.ones_like(x_freq)
        for ch in range(image.shape[1]): mask_freq[:, ch*4+1 : ch*4+4] = rho[:, ch*4+1 : ch*4+4]
        x_aug = idwt(x_freq * mask_freq)
        
        output = model(normalize(x_aug))
        loss = F.cross_entropy(output, target)
        model.zero_grad(); loss.backward()
        grad = adv_image.grad.data
        grad = grad / (torch.norm(grad, p=1, dim=(1,2,3), keepdim=True) + 1e-10)
        momentum = 1.0 * momentum + grad
        
        # HVS 动态约束
        dynamic_eps = epsilon * hvs_mask
        adv_temp = adv_image + alpha * momentum.sign()
        delta = torch.clamp(adv_temp - image, -dynamic_eps, dynamic_eps)
        adv_image = torch.clamp(image + delta, 0, 1).detach().requires_grad_(True)
    return adv_image.detach()

# ==========================================
# 3. 运行可视化
# ==========================================
def run_high_res_demo(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型 (ImageNet 预训练)
    print("Loading ImageNet Pretrained ResNet18...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    
    # 2. 处理图片
    raw_img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)
    
    # 3. 预测原始类别
    with torch.no_grad():
        output = model(normalize(img_tensor))
        target = output.max(1)[1]
    
    # 4. 执行攻击
    EPS = 16/255 # 大图上可以用稍大的 eps 观察对比
    print("Running Attacks...")
    adv_mim = mifgsm_attack(model, img_tensor, target, EPS)
    adv_ssa = mi_ssa_dwt_attack(model, img_tensor, target, EPS)
    
    # 5. 绘图对比
    def to_np(t): return t.squeeze().cpu().permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 第一行：对抗样本原图
    axes[0, 0].imshow(to_np(img_tensor)); axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(to_np(adv_mim)); axes[0, 1].set_title("MI-FGSM Attack")
    axes[0, 2].imshow(to_np(adv_ssa)); axes[0, 2].set_title("MI-SSA-DWT (Ours)")
    
    # 第二行：扰动噪声 (放大 20 倍以便观察分布)
    noise_mim = torch.abs(adv_mim - img_tensor) * 20
    noise_ssa = torch.abs(adv_ssa - img_tensor) * 20
    
    axes[1, 0].axis('off')
    axes[1, 1].imshow(to_np(torch.clamp(noise_mim, 0, 1))); axes[1, 1].set_title("MI-FGSM Noise (x20)")
    axes[1, 2].imshow(to_np(torch.clamp(noise_ssa, 0, 1))); axes[1, 2].set_title("Our Proposed Noise (x20)")
    
    for ax in axes.flatten(): ax.axis('off')
    plt.tight_layout()
    plt.savefig("high_res_comparison.png", dpi=300)
    plt.show()
    print("Result saved as 'high_res_comparison.png'")

if __name__ == "__main__":
    run_high_res_demo("chechin.jpg")