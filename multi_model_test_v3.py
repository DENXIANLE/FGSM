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
import time  # 引入时间库，顺便测个速！

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
# 2. 攻击算法全家桶
# ===========================

# [1] FGSM
def fgsm_attack(image, epsilon, data_grad):
    return image + epsilon * data_grad.sign()

# [2] R-FGSM (新加入：随机单步)
def r_fgsm_attack(model, image, target, epsilon, alpha=None):
    if alpha is None: alpha = epsilon / 2.0
    noise = torch.randn_like(image).to(image.device) * alpha
    image_noisy = image + noise
    image_noisy = torch.clamp(image_noisy, 0, 1).detach()
    image_noisy.requires_grad = True
    output = model(image_noisy)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = image_noisy.grad.data
    adv_image = image_noisy + (epsilon - alpha) * data_grad.sign()
    return torch.clamp(adv_image, 0, 1)

# [3] MI-FGSM (迭代)
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

# [4] Freq Attack (Yours)
def freq_attack(image, epsilon, data_grad, radius=2):
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask = torch.ones_like(grad_freq_shifted)
    r = int(min(radius, crow, ccol))
    if r > 0: mask[:, :, crow-r:crow+r, ccol-r:ccol+r] = 0
    grad_freq_high = grad_freq_shifted * mask
    grad_freq_high_ishift = torch.fft.ifftshift(grad_freq_high, dim=(-2, -1))
    perturbation = torch.fft.ifftn(grad_freq_high_ishift, dim=(-2, -1)).real
    return image + epsilon * perturbation.sign()

# ===========================
# 3. 主测试逻辑
# ===========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 4-Way Battle: FGSM vs R-FGSM vs MI-FGSM vs Freq Attack")

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
    
    BASE_EPS = 0.05
    BOOST_EPS = BASE_EPS * 1.5
    NUM_SAMPLES = 100 
    
    adv_data = {"FGSM": [], "R-FGSM": [], "MI-FGSM": [], "Freq": []}
    labels_list = []
    
    print("\nGenerating Adversarial Examples...")
    count = 0
    
    # 计时器
    time_costs = {"FGSM": 0, "R-FGSM": 0, "MI-FGSM": 0, "Freq": 0}
    
    for data, target in loader:
        if count >= NUM_SAMPLES: break
        data, target = data.to(device), target.to(device)
        
        # Check source prediction
        output = source_model(data)
        if output.max(1)[1].item() != target.item(): continue 
        
        # 1. 基础梯度 (FGSM/Freq用)
        data.requires_grad = True
        t0 = time.time()
        loss = F.cross_entropy(source_model(data), target)
        source_model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        t1 = time.time()
        base_grad_time = t1 - t0

        # --- 生成 FGSM ---
        t_start = time.time()
        adv_fgsm = fgsm_attack(data, BASE_EPS, data_grad)
        time_costs["FGSM"] += (time.time() - t_start) + base_grad_time
        
        # --- 生成 Freq (Yours) ---
        t_start = time.time()
        adv_freq = freq_attack(data, BOOST_EPS, data_grad, radius=2)
        time_costs["Freq"] += (time.time() - t_start) + base_grad_time

        # --- 生成 R-FGSM (内部算梯度) ---
        t_start = time.time()
        adv_rfgsm = r_fgsm_attack(source_model, data, target, BASE_EPS)
        time_costs["R-FGSM"] += (time.time() - t_start)

        # --- 生成 MI-FGSM (内部迭代) ---
        t_start = time.time()
        adv_mim = mifgsm_attack(source_model, data, target, BASE_EPS, num_iter=10)
        time_costs["MI-FGSM"] += (time.time() - t_start)
        
        adv_data["FGSM"].append(adv_fgsm.detach())
        adv_data["R-FGSM"].append(adv_rfgsm.detach())
        adv_data["MI-FGSM"].append(adv_mim.detach())
        adv_data["Freq"].append(adv_freq.detach())
        labels_list.append(target.item())
        count += 1
        if count % 20 == 0: print(f"   Generated {count}/{NUM_SAMPLES}...")

    # 打印平均耗时
    print("\n⏱️  Average Generation Time (per image):")
    for k, v in time_costs.items():
        print(f"   {k}: {v/NUM_SAMPLES*1000:.2f} ms")

    print("\n⚔️  Testing Transferability...")
    results = {k: [] for k in adv_data.keys()}
    model_names = list(target_models.keys())

    for name, model in target_models.items():
        print(f"Target: {name}")
        for method in adv_data.keys():
            correct = 0
            for i in range(len(labels_list)):
                if model(adv_data[method][i]).max(1)[1].item() == labels_list[i]:
                    correct += 1
            acc = correct / len(labels_list)
            results[method].append(acc)
            print(f"   {method}: {acc:.2f}")

    # 画图
    x = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 颜色：FGSM(浅蓝), R-FGSM(深蓝), MI-FGSM(灰), Freq(橙)
    rects1 = ax.bar(x - 1.5*width, results["FGSM"], width, label='FGSM', color='#a6cee3')
    rects2 = ax.bar(x - 0.5*width, results["R-FGSM"], width, label='R-FGSM', color='#1f78b4')
    rects3 = ax.bar(x + 0.5*width, results["MI-FGSM"], width, label='MI-FGSM (Iterative)', color='gray')
    rects4 = ax.bar(x + 1.5*width, results["Freq"], width, label='Freq Attack (Yours)', color='darkorange')

    ax.set_ylabel('Target Accuracy (Lower is Better)')
    ax.set_title('Comprehensive Transferability Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    autolabel(rects1); autolabel(rects2); autolabel(rects3); autolabel(rects4)
    plt.savefig('comparison_4way.png')
    plt.show()

if __name__ == '__main__':
    main()