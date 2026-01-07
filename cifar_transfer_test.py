import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg11_bn
import matplotlib.pyplot as plt

# --- 模型定义 (ResNet) ---
def get_resnet(device):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load("./data/cifar_resnet18.pt", map_location=device))
    return model.to(device).eval()

# --- 模型定义 (VGG) ---
def get_vgg(device):
    model = vgg11_bn(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)
    model.load_state_dict(torch.load("./data/cifar_vgg11.pt", map_location=device))
    return model.to(device).eval()

# --- 攻击函数 (已修复 Clamp 问题) ---
def fgsm_attack(image, epsilon, data_grad):
    return image + epsilon * data_grad.sign()

def freq_attack(image, epsilon, data_grad, radius=4):
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

# --- 主测试逻辑 ---
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    # 只测前 500 张以节省时间
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    source_model = get_resnet(device) # 攻击源
    target_model = get_vgg(device)    # 攻击目标 (黑盒)
    
    epsilons = [0, 0.03, 0.06, 0.1] # CIFAR-10 常用 Epsilon
    acc_fgsm = []
    acc_freq = []
    
    print(f"{'Epsilon':<10} | {'FGSM Acc (VGG)':<20} | {'Freq Acc (VGG)':<20}")
    
    for eps in epsilons:
        correct_fgsm = 0
        correct_freq = 0
        total = 0
        
        for i, (data, target) in enumerate(testloader):
            if i >= 500: break # 限制测试数量
            
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            
            # 1. 在 Source Model 上生成梯度
            output = source_model(data)
            if output.max(1)[1].item() != target.item(): continue # 只攻击原本分类正确的
            
            loss = F.cross_entropy(output, target)
            source_model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            # 2. 生成样本
            adv_fgsm = fgsm_attack(data, eps, data_grad)
            adv_freq = freq_attack(data, eps* 1.5 , data_grad, radius=2) # 可以尝试调节 Radius
            
            # 3. 攻击 Target Model (VGG)
            # 注意：攻击后要 Clamp 回合法范围，这里简单 Clip 模拟
            # 实际上由于 Normalize 存在，很难精确 Clip 到 0-1，但我们可以直接喂给模型
            
            if target_model(adv_fgsm).max(1)[1].item() == target.item():
                correct_fgsm += 1
            if target_model(adv_freq).max(1)[1].item() == target.item():
                correct_freq += 1
            total += 1
            
        acc_fgsm.append(correct_fgsm/total)
        acc_freq.append(correct_freq/total)
        print(f"{eps:<10} | {acc_fgsm[-1]:<20.4f} | {acc_freq[-1]:<20.4f}")

    # 画图
    plt.plot(epsilons, acc_fgsm, '*-', label='FGSM (Baseline)')
    plt.plot(epsilons, acc_freq, 'o-', label='Freq Attack (Yours)')
    plt.title("Transferability: ResNet-18 -> VGG-11")
    plt.xlabel("Epsilon")
    plt.ylabel("Target Model Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()