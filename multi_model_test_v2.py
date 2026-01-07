import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, 
    vgg11_bn, 
    mobilenet_v2, 
    shufflenet_v2_x1_0
)
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

# ===========================
# 1. 模型加载 (保持不变)
# ===========================
def get_model(name, device):
    model = None
    path_options = []

    if name == "resnet":
        model = resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)
        path_options = ["./data/cifar_resnet18.pt", "./data/cifar_resnet.pt"]

    elif name == "vgg":
        model = vgg11_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
        path_options = ["./data/cifar_vgg11.pt", "./data/cifar_vgg.pt"]

    elif name == "mobilenet":
        model = mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        path_options = ["./data/cifar_mobilenet.pt", "./data/cifar_mobilenet_v2.pt"]

    elif name == "shufflenet":
        model = shufflenet_v2_x1_0(pretrained=False)
        out_channels = model.conv1[0].out_channels
        model.conv1[0] = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        path_options = ["./data/cifar_shufflenet.pt", "./data/cifar_shufflenet_v2.pt"]

    model = model.to(device)
    loaded = False
    for path in path_options:
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=device))
                loaded = True
                break
            except: pass
    
    if not loaded: return None
    return model.eval()

# ===========================
# 2. 三种攻击算法
# ===========================

# [1] FGSM
def fgsm_attack(image, epsilon, data_grad):
    return image + epsilon * data_grad.sign()

# [2] MI-FGSM (新加入的强力对手)
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
        
        grad_norm = torch.norm(data_grad, p=1, dim=(1,2,3), keepdim=True)
        data_grad = data_grad / (grad_norm + 1e-10)
        momentum = decay * momentum + data_grad
        
        adv_image = adv_image + alpha * momentum.sign()
        delta = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.clamp(image + delta, 0, 1).detach()
        adv_image.requires_grad = True

    return adv_image

# [3] Freq Attack (你的方法)
def freq_attack(image, epsilon, data_grad, radius=2):
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask = torch.ones_like(grad_freq_shifted)
    r = int(min(radius, crow, ccol))
    if r > 0:
        mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    grad_freq_high = grad_freq_shifted * mask
    grad_freq_high_ishift = torch.fft.ifftshift(grad_freq_high, dim=(-2, -1))
    perturbation = torch.fft.ifftn(grad_freq_high_ishift, dim=(-2, -1)).real
    return image + epsilon * perturbation.sign()

# ===========================
# 3. 主测试逻辑
# ===========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 3-Way Battle: FGSM vs MI-FGSM vs Freq Attack")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    source_model = get_model("resnet", device)
    target_models = {
        "VGG-11": get_model("vgg", device),
        "MobileNet": get_model("mobilenet", device),
        "ShuffleNet": get_model("shufflenet", device)
    }
    
    # 实验设置
    BASE_EPS = 0.05
    BOOST_EPS = BASE_EPS * 1.5
    NUM_SAMPLES = 100 # 生成100个样本对比
    
    adv_fgsm_list = []
    adv_mim_list = []   # 存储 MI-FGSM
    adv_freq_list = []
    labels_list = []
    
    print("\nGenerating Adversarial Examples...")
    count = 0
    for data, target in loader:
        if count >= NUM_SAMPLES: break
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        output = source_model(data)
        if output.max(1)[1].item() != target.item(): continue 
        
        # 1. 计算单步梯度 (给 FGSM 和 Freq 用)
        loss = F.cross_entropy(output, target)
        source_model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        # 2. 生成 FGSM
        adv_fgsm = fgsm_attack(data, BASE_EPS, data_grad)
        
        # 3. 生成 MI-FGSM (需要传入模型进行迭代)
        # 注意：MI-FGSM 通常攻击力很强，我们给它公平的 Base Epsilon
        adv_mim = mifgsm_attack(source_model, data, target, BASE_EPS, num_iter=10)
        
        # 4. 生成 Freq Attack (你的策略)
        adv_freq = freq_attack(data, BOOST_EPS, data_grad, radius=2)
        
        adv_fgsm_list.append(adv_fgsm.detach())
        adv_mim_list.append(adv_mim.detach())
        adv_freq_list.append(adv_freq.detach())
        labels_list.append(target.item())
        count += 1
        if count % 20 == 0: print(f"   Generated {count}/{NUM_SAMPLES}...")

    # 测试迁移性
    print("\n⚔️  Testing Transferability...")
    results = {"FGSM": [], "MI-FGSM": [], "Freq (Yours)": []}
    model_names = list(target_models.keys())

    for name, model in target_models.items():
        correct_fgsm = 0
        correct_mim = 0
        correct_freq = 0
        total = len(labels_list)
        
        for i in range(total):
            l = labels_list[i]
            if model(adv_fgsm_list[i]).max(1)[1].item() == l: correct_fgsm += 1
            if model(adv_mim_list[i]).max(1)[1].item() == l: correct_mim += 1
            if model(adv_freq_list[i]).max(1)[1].item() == l: correct_freq += 1
            
        results["FGSM"].append(correct_fgsm/total)
        results["MI-FGSM"].append(correct_mim/total)
        results["Freq (Yours)"].append(correct_freq/total)
        
        print(f"[{name}] FGSM: {correct_fgsm/total:.2f} | MIM: {correct_mim/total:.2f} | Freq: {correct_freq/total:.2f}")

    # 画 3 根柱子的图
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width, results["FGSM"], width, label='FGSM (Baseline)', color='steelblue')
    r2 = ax.bar(x, results["MI-FGSM"], width, label='MI-FGSM (Strong Baseline)', color='gray')
    r3 = ax.bar(x + width, results["Freq (Yours)"], width, label='Freq Attack (Yours)', color='darkorange')

    ax.set_ylabel('Target Model Accuracy')
    ax.set_title('Transferability Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(r1)
    autolabel(r2)
    autolabel(r3)

    plt.savefig('comparison_3way.png')
    plt.show()

if __name__ == '__main__':
    main()