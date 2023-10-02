import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# 定义 CBAMModified 和相关类
class ChannelAttentionModified(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttentionModified, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out), out  # Return the attention weight and pre-sigmoid output


class SpatialAttentionModified(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModified, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x), x  # Return the attention weight and pre-sigmoid output


class CBAMModified(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBAMModified, self).__init__()
        self.ca = ChannelAttentionModified(in_planes, ratio)
        self.sa = SpatialAttentionModified(kernel_size)

    def forward(self, x):
        ca_weight, ca_out = self.ca(x)
        sa_weight, sa_out = self.sa(x * ca_weight)
        return ca_weight * sa_weight * x, ca_weight, sa_weight

def matplotlib_pil_save_attention_map_single(input_image, ca_weight, sa_weight, filename, attention_map_dir):
    """
    使用matplotlib colormap和PIL进行混合以保存单张图片的注意力图。
    
    Args:
    - input_image (torch.Tensor): 输入图片张量。
    - ca_weight (torch.Tensor): 通道注意力权重。
    - sa_weight (torch.Tensor): 空间注意力权重。
    - filename (str): 保存注意力图的文件名。
    - attention_map_dir (str): 保存注意力图的目录。
    """
    
    # 确保文件名有一个扩展名，默认为.png
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        filename += '.png'
    
    # 组合通道和空间注意力
    mixed_attention = ca_weight * sa_weight
    
    # 如果注意力的形状与输入的形状不匹配，则调整注意力的大小
    if mixed_attention.shape[-2:] != input_image.shape[-2:]:
        mixed_attention = F.interpolate(mixed_attention, size=input_image.shape[-2:], mode='bilinear')
    
    # 重塑注意力权重以匹配输入图片的通道
    reshaped_attention = mixed_attention.reshape(mixed_attention.shape[0] // 3, 3, *mixed_attention.shape[1:]).mean(dim=1)
    
    # 将3个通道平均以得到单通道的注意力图
    single_channel_attention = reshaped_attention.mean(dim=0, keepdim=True)
    
    # 归一化到[0, 1]
    normalized_attention = (single_channel_attention - single_channel_attention.min()) / (single_channel_attention.max() - single_channel_attention.min())
    
    # 为了平滑性应用高斯模糊
    attention_np = normalized_attention.squeeze().cpu().numpy()
    attention_np = gaussian_filter(attention_np, sigma=5)
    
    # 使用matplotlib colormap将注意力转换为颜色
    colormap = plt.cm.jet
    attention_rgb = (colormap(attention_np)[:, :, :3] * 255).astype(np.uint8)
    
    input_image_np = input_image.squeeze().cpu().numpy()
    if len(input_image_np.shape) == 3:  # 彩色图像
        input_image_np = np.transpose(input_image_np, (1, 2, 0))

    # 将numpy数组转换为PIL图像
    input_image_pil = Image.fromarray((input_image_np * 255).astype(np.uint8))
    attention_pil = Image.fromarray(attention_rgb)
    
    # 混合图像
    blended_image = Image.blend(input_image_pil, attention_pil, alpha=0.4)
    
    # 保存混合图像
    blended_image.save(os.path.join(attention_map_dir, filename))

def main():
    # 加载模型
    model = CBAMModified(in_planes=2048).eval()  # 假设输入的通道数是2048，您可能需要根据实际情况调整
    model_path = "/root/autodl-tmp/new_swin/Detection_Baseline_swin/save_weights/GAN_seris_anchor/model_29.pth"
    model.load_state_dict(torch.load(model_path))

    # 加载图像
    image_path = "/root/autodl-tmp/new_swin/Detection_Baseline_swin/data_new/coco/val/video01_clip02_step2_017.jpg"
    input_image = Image.open(image_path).convert("RGB")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据您的模型输入需求进行调整
        transforms.ToTensor(),
    ])
    input_image_tensor = transform(input_image).unsqueeze(0)  # 增加batch维度

    # 获取注意力权重
    with torch.no_grad():
        _, ca_weight, sa_weight = model(input_image_tensor)

    # 注意力可视化
    attention_map_dir = "./attention_maps"
    if not os.path.exists(attention_map_dir):
        os.makedirs(attention_map_dir)

    filename = os.path.basename(image_path).replace(".jpg", "_attention.jpg")
    matplotlib_pil_save_attention_map_single(input_image_tensor[0], ca_weight, sa_weight, filename, attention_map_dir)

if __name__ == "__main__":
    main()