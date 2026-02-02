import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg11_bn, mobilenet_v2, shufflenet_v2_x1_0
import numpy as np
import os

# 尝试导入 torchmetrics
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

# ==========================================
# 1. 基础工具组件 (DWT/IDWT)
# ==========================================
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        # Haar Kernels
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

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ==========================================
# 2. HVS 视觉掩码计算
# ==========================================
def get_hvs_mask(image):
    """
    计算视觉掩码：边缘区域允许大扰动(1.0)，平滑区域限制扰动(0.5)
    这能显著提升 PSNR/SSIM，且符合"不可感知"定义
    """
    b, c, h, w = image.shape
    x_grad = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    y_grad = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    
    # 填充回原尺寸
    grad_mag = torch.zeros_like(image)
    grad_mag[:, :, :, :-1] += x_grad
    grad_mag[:, :, :-1, :] += y_grad
    grad_mag = grad_mag.mean(dim=1, keepdim=True) # 转灰度
    
    # 归一化并映射到 [0.5, 1.0]
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    mask = 0.5 + 0.5 * grad_mag
    return mask

# ==========================================
# 3. 核心算法对比
# ==========================================

# A. MI-FGSM (Baseline)
def mifgsm_attack(model, image, target, epsilon, num_iter=10, decay=1.0):
    alpha = epsilon / num_iter
    momentum = torch.zeros_like(image).detach()
    adv_image = image.clone().detach()
    adv_image.requires_grad = True
    
    for i in range(num_iter):
        output = model(normalize(adv_image))
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        grad = grad / (torch.norm(grad, p=1, dim=(1,2,3), keepdim=True) + 1e-10)
        momentum = decay * momentum + grad
        
        adv_image = adv_image + alpha * momentum.sign()
        delta = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.clamp(image + delta, 0.0, 1.0).detach()
        adv_image.requires_grad = True
        
    return adv_image.detach()

# B. MI-SSA-DWT (V2 - Correct Implementation)
def mi_ssa_dwt_attack_v2(model, image, target, epsilon, num_iter=20, decay=1.0, eta=0.2):
    """
    V2 改进点:
    1. Input Transformation: SSA 应用在 forward 之前 (增强迁移性)
    2. HVS Masking: 使用边缘掩码约束 epsilon (增强 PSNR)
    """
    dwt = DWT().to(image.device)
    idwt = IDWT().to(image.device)
    
    # 计算 HVS 掩码
    hvs_mask = get_hvs_mask(image) 
    
    alpha = epsilon / num_iter
    momentum = torch.zeros_like(image).detach()
    adv_image = image.clone().detach()
    adv_image.requires_grad = True
    
    for i in range(num_iter):
        # --- 1. 输入端 SSA 变换 ---
        x_freq = dwt(adv_image)
        # 生成随机噪声
        rho = (1 - eta) + (2 * eta) * torch.rand(x_freq.shape[0], x_freq.shape[1], 1, 1).to(image.device)
        
        # 只缩放高频，LL 不变
        mask_freq = torch.ones_like(x_freq)
        for ch in range(image.shape[1]):
            mask_freq[:, ch*4+1 : ch*4+4] = rho[:, ch*4+1 : ch*4+4]
            
        x_aug = idwt(x_freq * mask_freq)
        
        # --- 2. 前向传播 ---
        output = model(normalize(x_aug))
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        grad = grad / (torch.norm(grad, p=1, dim=(1,2,3), keepdim=True) + 1e-10)
        momentum = decay * momentum + grad
        
        # --- 3. 更新与约束 (应用 HVS Mask) ---
        dynamic_epsilon = epsilon * hvs_mask
        
        adv_image = adv_image + alpha * momentum.sign()
        
        # Clamp 时使用动态 epsilon
        delta = adv_image - image
        delta = torch.max(torch.min(delta, dynamic_epsilon), -dynamic_epsilon)
        
        adv_image = torch.clamp(image + delta, 0.0, 1.0).detach()
        adv_image.requires_grad = True
        
    return adv_image.detach()


# ==========================================
# 4. 主实验逻辑 (Fixed get_model)
# ==========================================
def get_model(name, device):
    # === 修复点：强制转小写，避免 KeyError ===
    name = name.lower() 
    
    paths = {
        "resnet": ["./data/cifar_resnet18.pt", "./data/cifar_resnet.pt"],
        "vgg": ["./data/cifar_vgg11.pt", "./data/cifar_vgg.pt"],
        "mobilenet": ["./data/cifar_mobilenet.pt", "./data/cifar_mobilenet_v2.pt"],
        "shufflenet": ["./data/cifar_shufflenet.pt", "./data/cifar_shufflenet_v2.pt"]
    }
    
    if name == "resnet": 
        model = resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512,10)
    elif name == "vgg": 
        model = vgg11_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
    elif name == "mobilenet": 
        model = mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
    elif name == "shufflenet": 
        model = shufflenet_v2_x1_0(pretrained=False)
        model.conv1[0] = nn.Conv2d(3, 24, 3, 1, 1, bias=False)
        model.fc = nn.Linear(1024, 10)
    else:
        print(f"Model {name} not defined.")
        return None
    
    # 尝试加载权重
    for p in paths.get(name, []):
        if os.path.exists(p):
            try:
                model.load_state_dict(torch.load(p, map_location=device))
                return model.to(device).eval()
            except Exception as e:
                print(f"Failed loading {p}: {e}")
                
    print(f"Error: Weights for {name} not found.")
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running FINAL EXPERIMENT (V3) on {device}")
    
    if HAS_TORCHMETRICS: ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    source = get_model("resnet", device)
    # 这里即使写大写 "VGG" 也没问题了，因为 get_model 内部会处理
    target_names = ["VGG", "MobileNet", "ShuffleNet"]
    targets = {k: get_model(k, device) for k in target_names}
    
    # 移除加载失败的模型
    targets = {k: v for k, v in targets.items() if v is not None}
    
    if source is None or not targets: 
        print("Critical Error: Models not loaded. Please check ./data/ folder.")
        return

    # 实验参数
    EPS = 8/255
    SAMPLES = 100
    
    res = {"MI-FGSM": {"acc":{k:0 for k in targets}, "psnr":0, "ssim":0}, 
           "MI-SSA-DWT": {"acc":{k:0 for k in targets}, "psnr":0, "ssim":0}}
    
    count = 0
    print(f"Testing {SAMPLES} images...")
    
    for data, lbl in loader:
        if count >= SAMPLES: break
        data, lbl = data.to(device), lbl.to(device)
        
        # 仅攻击分类正确的
        if source(normalize(data)).max(1)[1] != lbl: continue
        
        # 生成
        adv_base = mifgsm_attack(source, data, lbl, EPS)
        adv_ours = mi_ssa_dwt_attack_v2(source, data, lbl, EPS, num_iter=20, eta=0.3)
        
        # 记录 PSNR/SSIM
        for m, adv in zip(["MI-FGSM", "MI-SSA-DWT"], [adv_base, adv_ours]):
            res[m]["psnr"] += calculate_psnr(data, adv).item()
            if HAS_TORCHMETRICS: res[m]["ssim"] += ssim_metric(adv, data).item()
            
            # 记录 Acc
            for name, net in targets.items():
                if net(normalize(adv)).max(1)[1] == lbl:
                    res[m]["acc"][name] += 1
                    
        count += 1
        if count % 10 == 0: print(f"Processing... {count}")
        
    print("\n" + "="*70)
    print(f"{'Method':<15} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Avg Transfer Acc (Lower=Better)':<30}")
    print("-" * 70)
    
    for m in ["MI-FGSM", "MI-SSA-DWT"]:
        psnr = res[m]["psnr"]/count
        ssim = res[m]["ssim"]/count if HAS_TORCHMETRICS else 0
        acc = sum(res[m]["acc"].values()) / (count * len(targets)) * 100
        print(f"{m:<15} | {psnr:<10.2f} | {ssim:<8.4f} | {acc:.2f}%")
        
    print("-" * 70)
    print("Verdict:")
    print("Run finished. Check if MI-SSA-DWT has higher PSNR and lower/similar Acc.")

if __name__ == '__main__':
    main()