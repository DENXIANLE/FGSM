import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg11_bn, mobilenet_v2, shufflenet_v2_x1_0
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

# ==========================================
# 1. 小波变换模块 (用于 SSA-DWT)
# ==========================================
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        # Haar 小波核
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        # 增加维度以适配卷积: [4, 1, 2, 2]
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).float()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # 将 Batch 和 Channel 合并，以便对每个 Channel 独立卷积
        x_reshaped = x.view(b * c, 1, h, w)
        filters = self.filters.to(x.device)
        # 卷积 stride=2 实现下采样
        output = F.conv2d(x_reshaped, filters, stride=2)
        # 还原形状: [B, C*4, H/2, W/2]
        return output.view(b, c * 4, h // 2, w // 2)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).float()

    def forward(self, x):
        # x: [B, C*4, H/2, W/2]
        b, c4, h2, w2 = x.shape
        c = c4 // 4
        x_reshaped = x.view(b * c, 4, h2, w2)
        filters = self.filters.to(x.device)
        # 转置卷积上采样
        output = F.conv_transpose2d(x_reshaped, filters, stride=2)
        return output.view(b, c, h2 * 2, w2 * 2)

# ==========================================
# 2. 攻击算法库
# ==========================================

# [算法 A] 基础 MI-FGSM (直接内置，无需 import)
def mifgsm_attack(model, image, target, epsilon, decay=1.0, num_iter=10):
    alpha = epsilon / num_iter
    momentum = torch.zeros_like(image).detach().to(image.device)
    adv_image = image.clone().detach()
    adv_image.requires_grad = True
    
    for i in range(num_iter):
        output = model(adv_image)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        
        data_grad = adv_image.grad.data
        # 梯度归一化 (L1 norm)
        grad_norm = torch.norm(data_grad, p=1, dim=(1,2,3), keepdim=True)
        data_grad = data_grad / (grad_norm + 1e-10)
        
        # 动量累积
        momentum = decay * momentum + data_grad
        
        # 更新
        adv_image = adv_image + alpha * momentum.sign()
        delta = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.clamp(image + delta, 0, 1).detach()
        adv_image.requires_grad = True
        
    return adv_image

# [算法 B] 你的创新算法：MI-SSA-DWT
def mi_ssa_dwt_attack(model, image, target, epsilon, num_iter=10, decay=1.0, eta=0.1):
    """
    MI-SSA-DWT: 结合动量(Momentum)与小波域频谱模拟(SSA)的迭代攻击
    eta: 频谱抖动幅度 (0.1 表示在 0.9~1.1 倍之间抖动)
    """
    dwt = DWT().to(image.device)
    idwt = IDWT().to(image.device)
    
    alpha = epsilon / num_iter
    momentum = torch.zeros_like(image).detach()
    adv_image = image.clone().detach()
    adv_image.requires_grad = True
    
    for i in range(num_iter):
        output = model(adv_image)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        
        grad = adv_image.grad.data
        
        # --- 核心创新：小波域频谱模拟 (SSA) ---
        # 1. 梯度变换到小波域
        grad_dwt = dwt(grad) 
        
        # 2. 生成随机缩放因子 rho
        # rho ~ Uniform(1-eta, 1+eta)
        rho = (1 - eta) + (2 * eta) * torch.rand(1).item()
        
        # 3. 对高频子带 (LH, HL, HH) 进行缩放，保持 LL 不变
        # 通道结构: [C1_LL, C1_LH, C1_HL, C1_HH, C2_LL, ...]
        for ch in range(image.shape[1]): # 遍历 RGB 通道
            # 索引: ch*4 是 LL (跳过)
            # 索引: ch*4+1, ch*4+2, ch*4+3 是高频 (应用 rho)
            grad_dwt[:, ch*4+1 : ch*4+4, :, :] *= rho
            
        # 4. 逆变换回像素空间，得到"鲁棒"梯度
        grad_refined = idwt(grad_dwt)
        
        # --- 动量更新 (同 MI-FGSM) ---
        grad_norm = torch.norm(grad_refined, p=1, dim=(1,2,3), keepdim=True)
        grad_refined = grad_refined / (grad_norm + 1e-10)
        momentum = decay * momentum + grad_refined
        
        # --- 步进更新 ---
        adv_image = adv_image + alpha * momentum.sign()
        delta = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.clamp(image + delta, 0, 1).detach()
        adv_image.requires_grad = True
        
    return adv_image

# ===========================
# 3. 模型加载与主测试程序
# ===========================
def get_model(name, device):
    # 定义模型结构与权重路径的映射
    model_map = {
        "resnet": (resnet18, ["./data/cifar_resnet18.pt", "./data/cifar_resnet.pt"]),
        "vgg": (vgg11_bn, ["./data/cifar_vgg11.pt", "./data/cifar_vgg.pt"]),
        "mobilenet": (mobilenet_v2, ["./data/cifar_mobilenet.pt", "./data/cifar_mobilenet_v2.pt"]),
        "shufflenet": (shufflenet_v2_x1_0, ["./data/cifar_shufflenet.pt", "./data/cifar_shufflenet_v2.pt"])
    }
    
    if name not in model_map: return None
    
    m_func, paths = model_map[name]
    model = m_func(pretrained=False)
    
    # 针对 CIFAR-10 修改首层和全连接层
    if name == "resnet":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)
    elif name == "vgg":
        model.classifier[6] = nn.Linear(4096, 10)
    elif name == "mobilenet":
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
    elif name == "shufflenet":
        out_c = model.conv1[0].out_channels
        model.conv1[0] = nn.Conv2d(3, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)

    # 加载权重
    loaded = False
    for p in paths:
        if os.path.exists(p):
            try:
                model.load_state_dict(torch.load(p, map_location=device))
                print(f"Loaded {name} from {p}")
                loaded = True
                break
            except: pass
    
    if not loaded:
        print(f"Warning: Weights not found for {name}, using random weights.")
    
    return model.to(device).eval()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 New Battle: FGSM vs MI-FGSM vs MI-SSA-DWT (Proposed)")

    # 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
        loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    except:
        print("Error: CIFAR-10 data not found. Please download it first.")
        return

    # 加载模型
    source_model = get_model("resnet", device)
    target_models = {
        "VGG-11": get_model("vgg", device),
        "MobileNet": get_model("mobilenet", device),
        "ShuffleNet": get_model("shufflenet", device)
    }
    
    # 实验参数
    EPS = 0.05
    NUM_SAMPLES = 100  # 测试 100 张图片
    
    results = {"FGSM": [], "MI-FGSM": [], "MI-SSA-DWT": []}
    labels_list = []
    adv_data = {"FGSM": [], "MI-FGSM": [], "MI-SSA-DWT": []}

    print("\nGenerating Adversarial Examples...")
    count = 0
    for data, target in loader:
        if count >= NUM_SAMPLES: break
        data, target = data.to(device), target.to(device)
        
        # 只攻击源模型预测正确的图片
        if source_model(data).max(1)[1].item() != target.item(): continue 
        
        # 1. FGSM
        data.requires_grad = True
        loss = F.cross_entropy(source_model(data), target)
        source_model.zero_grad(); loss.backward()
        adv_fgsm = (data + EPS * data.grad.data.sign()).detach()
        adv_data["FGSM"].append(adv_fgsm)
        
        # 2. MI-FGSM
        adv_mim = mifgsm_attack(source_model, data, target, EPS).detach()
        adv_data["MI-FGSM"].append(adv_mim)
        
        # 3. MI-SSA-DWT (你的新算法)
        adv_ssa = mi_ssa_dwt_attack(source_model, data, target, EPS).detach()
        adv_data["MI-SSA-DWT"].append(adv_ssa)
        
        labels_list.append(target.item())
        count += 1
        if count % 20 == 0: print(f"   Processed {count}/{NUM_SAMPLES}...")

    print("\n⚔️ Testing Transferability (Accuracy of Target Model)...")
    print("(Lower Accuracy = Better Attack Success)")
    
    final_accs = {}
    
    for m_name, model in target_models.items():
        if model is None: continue
        print(f"\nTarget: {m_name}")
        final_accs[m_name] = {}
        for method in results.keys():
            correct = 0
            for i in range(len(labels_list)):
                pred = model(adv_data[method][i]).max(1)[1].item()
                if pred == labels_list[i]:
                    correct += 1
            acc = correct / len(labels_list)
            results[method].append(acc)
            final_accs[m_name][method] = acc
            print(f"   {method}: Accuracy={acc:.2f}")

    # 画图
    model_names = list(target_models.keys())
    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, results["FGSM"], width, label='FGSM', color='#a6cee3')
    rects2 = ax.bar(x, results["MI-FGSM"], width, label='MI-FGSM (Iterative)', color='gray')
    rects3 = ax.bar(x + width, results["MI-SSA-DWT"], width, label='MI-SSA-DWT (Ours)', color='darkorange')

    ax.set_ylabel('Target Accuracy (Lower is Better)')
    ax.set_title('Improved Transferability: MI-SSA-DWT vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    # 自动标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1); autolabel(rects2); autolabel(rects3)
    
    save_path = 'improved_transfer_result.png'
    plt.savefig(save_path)
    print(f"\n✅ Result plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()