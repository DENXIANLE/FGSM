import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, 
    vgg11_bn, 
    mobilenet_v2, 
    shufflenet_v2_x1_0, 
    densenet121
)
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

# ===========================
# 1. 模型加载工厂 (Model Factory)
# ===========================
def get_model(name, device):
    model = None
    path_options = [] # 存储可能的路径，增加容错性

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

    elif name == "densenet": # 以防万一你有这个
        model = densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(1024, 10)
        path_options = ["./data/cifar_densenet.pt"]

    # 尝试加载权重
    model = model.to(device)
    loaded = False
    for path in path_options:
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            try:
                model.load_state_dict(torch.load(path, map_location=device))
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
    
    if not loaded:
        print(f"❌ Error: Could not find weight file for {name}. Checked: {path_options}")
        # 这里不退出，而是返回 None，主程序处理
        return None
        
    return model.eval()

# ===========================
# 2. 攻击算法 (核心对比)
# ===========================
def fgsm_attack(image, epsilon, data_grad):
    # 标准 FGSM
    return image + epsilon * data_grad.sign()

def freq_attack(image, epsilon, data_grad, radius=2):
    # 你的制胜算法：频域高频增强攻击
    # 注意：这里接收的 epsilon 已经是放大过的
    grad_freq = torch.fft.fftn(data_grad, dim=(-2, -1))
    grad_freq_shifted = torch.fft.fftshift(grad_freq, dim=(-2, -1))
    
    rows, cols = data_grad.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask = torch.ones_like(grad_freq_shifted)
    
    # 策略：Radius=2 (保留更多中频攻击力)
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
    print(f"🚀 Running Transferability Test on {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    # 为了快速出结果，这里随机取样 200-500 张即可
    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    # 1. 加载源模型 (ResNet)
    print("\n--- Loading Source Model ---")
    source_model = get_model("resnet", device)
    if source_model is None: return

    # 2. 加载目标模型 (VGG, MobileNet, ShuffleNet)
    print("\n--- Loading Target Models ---")
    target_names = ["vgg", "mobilenet", "shufflenet"] # 这里列出你想测的模型
    target_models = {}
    
    for name in target_names:
        m = get_model(name, device)
        if m is not None:
            target_models[name] = m
    
    if len(target_models) == 0:
        print("❌ No target models loaded. Exiting.")
        return

    # 实验参数
    BASE_EPS = 0.05
    BOOST_EPS = BASE_EPS * 1.5  # 策略：放大 1.5 倍
    NUM_SAMPLES = 200           # 测试图片数量
    
    print(f"\n⚡ Attack Settings:")
    print(f"   - FGSM Epsilon: {BASE_EPS}")
    print(f"   - Freq Epsilon: {BOOST_EPS:.3f} (Boosted)")
    print(f"   - Freq Radius : 2")
    print(f"   - Test Samples: {NUM_SAMPLES}")
    
    # 3. 生成对抗样本
    print("\ngenerating adversarial examples from ResNet-18...")
    
    adv_fgsm_list = []
    adv_freq_list = []
    labels_list = []
    
    count = 0
    for data, target in loader:
        if count >= NUM_SAMPLES: break
        
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        # 只攻击源模型能正确分类的样本
        output = source_model(data)
        if output.max(1)[1].item() != target.item(): continue 
        
        loss = F.cross_entropy(output, target)
        source_model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        # 生成样本
        adv_fgsm = fgsm_attack(data, BASE_EPS, data_grad)
        adv_freq = freq_attack(data, BOOST_EPS, data_grad, radius=2) # 你的方法
        
        adv_fgsm_list.append(adv_fgsm.detach())
        adv_freq_list.append(adv_freq.detach())
        labels_list.append(target.item())
        
        count += 1
        if count % 50 == 0: print(f"   Generated {count}/{NUM_SAMPLES} samples...")

    # 4. 迁移攻击测试
    print(f"\n⚔️  Testing Transferability (Lower Accuracy = Better Attack)...")
    
    results_fgsm = {}
    results_freq = {}
    
    # 对每个目标模型进行测试
    for name, model in target_models.items():
        correct_fgsm = 0
        correct_freq = 0
        total = len(labels_list)
        
        for i in range(total):
            label = labels_list[i]
            
            # 测试 FGSM
            pred_fgsm = model(adv_fgsm_list[i]).max(1)[1].item()
            if pred_fgsm == label: correct_fgsm += 1
            
            # 测试 Freq
            pred_freq = model(adv_freq_list[i]).max(1)[1].item()
            if pred_freq == label: correct_freq += 1
            
        acc_fgsm = correct_fgsm / total
        acc_freq = correct_freq / total
        
        results_fgsm[name] = acc_fgsm
        results_freq[name] = acc_freq
        
        print(f"   [{name}] FGSM Acc: {acc_fgsm:.4f} | Freq Acc: {acc_freq:.4f}")
        if acc_freq < acc_fgsm:
            print(f"      ✅ Freq Attack wins! (Improved by {(acc_fgsm - acc_freq)*100:.2f}%)")
        else:
            print(f"      ⚠️ FGSM wins.")

    # 5. 画图 (Bar Chart)
    print("\n📊 Plotting results...")
    models_labels = list(target_models.keys())
    scores_fgsm = [results_fgsm[m] for m in models_labels]
    scores_freq = [results_freq[m] for m in models_labels]

    x = np.arange(len(models_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, scores_fgsm, width, label='FGSM (Baseline)', color='steelblue')
    rects2 = ax.bar(x + width/2, scores_freq, width, label='Freq Attack (Yours)', color='darkorange')

    ax.set_ylabel('Target Model Accuracy')
    ax.set_title('Transferability Attack Success (Source: ResNet-18)\nLower bar means better attack')
    ax.set_xticks(x)
    ax.set_xticklabels(models_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 标数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('./transfer_result.png') # 保存图片
    plt.show()
    print("✅ Experiment Done! Result saved as 'transfer_result.png'")

if __name__ == '__main__':
    main()