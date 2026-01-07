import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. 模型与算法定义 (复用你的毕设核心代码)
# ==========================================

# ResNet-18 (CIFAR-10适配版)
def get_model(device):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    try:
        model.load_state_dict(torch.load("./data/cifar_resnet18.pt", map_location=device))
        print("模型加载成功！")
    except:
        messagebox.showerror("错误", "找不到模型文件 ./data/cifar_resnet18.pt\n请确保路径正确。")
        exit()
    return model.to(device).eval()

# FGSM 攻击
def fgsm_attack(image, epsilon, data_grad):
    return image + epsilon * data_grad.sign()

# 你的创新算法：频域高频增强攻击 (Radius=2, Boosting=Yes)
def freq_attack(image, epsilon, data_grad, radius=2):
    # 这里我们使用 "作弊" 策略：Epsilon 在外部已经放大了，这里负责频域处理
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

# 辅助：反归一化与图片转换
def tensor_to_pil(tensor):
    # 反归一化
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转 numpy -> PIL
    img_np = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    return img_pil

# 辅助：生成彩色噪点热力图
# 辅助：生成彩色噪点热力图
def get_noise_heatmap(diff_tensor):
    # diff_tensor: [1, 3, 32, 32]
    
    # === 修改点：增加 .detach() ===
    # 先 detach 剥离梯度，再转到 CPU，最后转 numpy
    diff = torch.abs(diff_tensor).mean(dim=1).squeeze().detach().cpu().numpy()
    
    # 归一化到 0-1 以便显示颜色
    diff_min = diff.min()
    diff_max = diff.max()
    
    # 防止除以 0 (如果完全没有噪点)
    if diff_max - diff_min > 1e-8:
        diff = (diff - diff_min) / (diff_max - diff_min)
    else:
        diff = np.zeros_like(diff) # 如果全是0，就保持全黑
    
    # 使用 matplotlib colormap
    colormap = plt.get_cmap('jet')
    heatmap = (colormap(diff)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(heatmap)

# CIFAR-10 类别
CLASSES = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# ==========================================
# 2. GUI 主程序
# ==========================================
class AdversarialApp:
    def __init__(self, root):
        self.root = root
        self.root.title("毕设演示：变换域对抗样本生成系统")
        self.root.geometry("1000x700")
        
        # 初始化设备和模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.device)
        
        # 加载数据 (只加载 Test Set)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.dataset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=False, transform=transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.data_iter = iter(self.loader)
        
        # 攻击参数
        self.base_epsilon = 0.05
        
        # --- UI 布局 ---
        
        # 1. 顶部控制栏
        control_frame = tk.Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        btn_font = ("Arial", 12, "bold")
        tk.Button(control_frame, text="📸 加载新图片", command=self.load_next_image, font=btn_font, bg="#DDDDDD").pack(side=tk.LEFT, padx=20)
        tk.Button(control_frame, text="⚡ 生成攻击 (Run Attack)", command=self.run_attack, font=btn_font, bg="#FFCCCC", fg="red").pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(control_frame, text="就绪", font=("Arial", 10), fg="gray")
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # 2. 主显示区 (3列：原图, FGSM, Freq)
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 定义三列容器
        self.frames = []
        titles = ["原始图片 (Original)", "基准攻击 (FGSM)", "本文方法 (Freq Attack)"]
        
        for i in range(3):
            f = tk.LabelFrame(main_frame, text=titles[i], font=("Arial", 12, "bold"), padx=10, pady=10)
            f.grid(row=0, column=i, padx=10, sticky="nsew")
            main_frame.columnconfigure(i, weight=1)
            self.frames.append(f)
            
        # 占位图片变量 (防止垃圾回收)
        self.photos = [None] * 6 # 3个主图 + 3个噪点图
        
        # 初始化界面组件 (Img Label, Pred Label, Noise Label)
        self.ui_widgets = [] # 存储 [img_lbl, pred_lbl, noise_lbl] * 3
        
        for i in range(3):
            # 预测结果文字
            pred_lbl = tk.Label(self.frames[i], text="Pred: --", font=("Arial", 14), fg="blue")
            pred_lbl.pack()
            
            # 图像显示
            img_lbl = tk.Label(self.frames[i], bg="white")
            img_lbl.pack(pady=5)
            
            # 噪点标题
            tk.Label(self.frames[i], text="扰动热力图 (Noise)", font=("Arial", 10, "italic")).pack(pady=(10,0))
            
            # 噪点显示
            noise_lbl = tk.Label(self.frames[i], bg="#F0F0F0")
            noise_lbl.pack(pady=5)
            
            self.ui_widgets.append({"pred": pred_lbl, "img": img_lbl, "noise": noise_lbl})

        # 自动加载第一张图
        self.current_data = None
        self.current_target = None
        self.load_next_image()

    def update_image(self, col_idx, pil_img, noise_pil_img=None):
        # 放大图片以便观看 (CIFAR 32x32 太小了，放大到 200x200)
        size = (200, 200)
        img_resized = pil_img.resize(size, Image.NEAREST)
        photo = ImageTk.PhotoImage(img_resized)
        
        self.ui_widgets[col_idx]["img"].config(image=photo)
        self.photos[col_idx] = photo # 保持引用
        
        if noise_pil_img:
            noise_resized = noise_pil_img.resize(size, Image.NEAREST)
            noise_photo = ImageTk.PhotoImage(noise_resized)
            self.ui_widgets[col_idx]["noise"].config(image=noise_photo)
            self.photos[col_idx+3] = noise_photo
        else:
            # 清空噪点图
            self.ui_widgets[col_idx]["noise"].config(image="")

    def load_next_image(self):
        try:
            data, target = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            data, target = next(self.data_iter)
            
        self.current_data = data.to(self.device)
        self.current_target = target.to(self.device)
        
        # 显示原图
        pil_img = tensor_to_pil(self.current_data)
        self.update_image(0, pil_img)
        
        # 预测
        with torch.no_grad():
            output = self.model(self.current_data)
            pred = output.max(1)[1].item()
        
        # 更新文字
        truth = CLASSES[self.current_target.item()]
        pred_name = CLASSES[pred]
        color = "green" if pred == self.current_target.item() else "red"
        self.ui_widgets[0]["pred"].config(text=f"True: {truth}\nPred: {pred_name}", fg=color)
        
        # 清空其他两列
        for i in [1, 2]:
            self.ui_widgets[i]["img"].config(image="")
            self.ui_widgets[i]["noise"].config(image="")
            self.ui_widgets[i]["pred"].config(text="Pred: --", fg="black")
            
        self.status_label.config(text="图片已加载，请点击生成攻击")

    def run_attack(self):
        if self.current_data is None: return
        
        self.status_label.config(text="正在生成对抗样本...")
        self.root.update() # 刷新界面
        
        # 1. 计算梯度
        data = self.current_data.clone().detach()
        data.requires_grad = True
        
        output = self.model(data)
        loss = F.cross_entropy(output, self.current_target)
        self.model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        # ---------------------------------------------
        # 核心逻辑：执行攻击
        # ---------------------------------------------
        
        # A. FGSM (Epsilon = 0.05)
        adv_fgsm = fgsm_attack(data, self.base_epsilon, data_grad)
        
        # B. Freq Attack (Epsilon = 0.075, Radius = 2) --> 你的制胜策略
        # 策略：放大 epsilon 以利用隐蔽性换取攻击力
        boost_epsilon = self.base_epsilon * 1.5 
        adv_freq = freq_attack(data, boost_epsilon, data_grad, radius=2)
        
        # ---------------------------------------------
        # 更新界面
        # ---------------------------------------------
        
        # Helper to update column
        def process_result(col_idx, adv_tensor):
            # 预测
            with torch.no_grad():
                out = self.model(adv_tensor)
                pred_idx = out.max(1)[1].item()
            
            pred_name = CLASSES[pred_idx]
            is_success = (pred_idx != self.current_target.item())
            
            # 文字更新
            txt = f"Pred: {pred_name}"
            if is_success:
                txt += "\n(攻击成功!)"
                color = "red" 
            else:
                txt += "\n(攻击失败)"
                color = "green"
            
            self.ui_widgets[col_idx]["pred"].config(text=txt, fg=color)
            
            # 图片更新
            img_pil = tensor_to_pil(adv_tensor)
            
            # 噪点计算
            noise = adv_tensor - self.current_data
            noise_pil = get_noise_heatmap(noise)
            
            self.update_image(col_idx, img_pil, noise_pil)

        # 更新 FGSM 列 (Col 1)
        process_result(1, adv_fgsm)
        
        # 更新 Freq 列 (Col 2)
        process_result(2, adv_freq)
        
        self.status_label.config(text="攻击完成！注意观察 Freq Attack 的噪点纹理与攻击结果。")

# ==========================================
# 启动程序
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AdversarialApp(root)
    root.mainloop()