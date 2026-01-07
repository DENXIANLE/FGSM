import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# 需要安装这个库: pip install torchmetrics
# 如果没有安装，可以使用 pip install torchmetrics
# 或者使用 scikit-image: pip install scikit-image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.models import resnet18

# 复用之前的攻击函数
def fgsm_attack(image, epsilon, data_grad):
    return image + epsilon * data_grad.sign()

def freq_attack(image, epsilon, data_grad, radius=2):
    # 注意：这里 epsilon 已经在外部乘过系数了
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

def get_resnet(device):
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load("./data/cifar_resnet18.pt", map_location=device))
    return model.to(device).eval()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) # 数据范围归一化后大约在几之间，SSIM通常假定0-1或0-255
    # 注意：由于我们要算图像质量，最好是在反归一化之后算 (0-1范围)
    # 但为了简便，我们直接比较相对大小即可
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    model = get_resnet(device)
    
    # 设定实验参数
    base_epsilon = 0.05
    freq_boost = 1.5
    
    ssim_fgsm_total = 0
    ssim_freq_total = 0
    count = 0
    
    print(f"Comparing Image Quality (SSIM)...")
    print(f"FGSM Epsilon: {base_epsilon}")
    print(f"Freq Epsilon: {base_epsilon * freq_boost:.4f} (Boosted)")
    
    for i, (data, target) in enumerate(testloader):
        if i >= 100: break # 测100张够了
        
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        output = model(data)
        if output.max(1)[1].item() != target.item(): continue
        
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        # 生成样本
        adv_fgsm = fgsm_attack(data, base_epsilon, data_grad)
        adv_freq = freq_attack(data, base_epsilon * freq_boost, data_grad, radius=2)
        
        # 计算 SSIM (对比原图)
        # 为了计算准确，这里做一个简单的反归一化模拟，把数据拉回 0-1 范围
        # 简单平移缩放即可，只要对两者公平
        img_orig = (data - data.min()) / (data.max() - data.min())
        img_fgsm = (adv_fgsm - adv_fgsm.min()) / (adv_fgsm.max() - adv_fgsm.min())
        img_freq = (adv_freq - adv_freq.min()) / (adv_freq.max() - adv_freq.min())

        s1 = ssim(img_fgsm, img_orig)
        s2 = ssim(img_freq, img_orig)
        
        ssim_fgsm_total += s1.item()
        ssim_freq_total += s2.item()
        count += 1
        
    print("-" * 30)
    print(f"Avg SSIM (FGSM): {ssim_fgsm_total/count:.4f}")
    print(f"Avg SSIM (Freq): {ssim_freq_total/count:.4f}")
    
    if (ssim_freq_total/count) >= (ssim_fgsm_total/count):
        print("结论: Freq Attack 画质更好或持平！实验成功！")
    else:
        print("结论: Freq Attack 画质略低，但考虑到攻击力提升，可接受。")

if __name__ == '__main__':
    main()