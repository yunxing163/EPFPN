# EPFPN for Object Detection

EPFPN（Edge Progressive Feature Pyramid Network）是一种面向通用目标检测任务的特征增强网络，专为提升 Vision Transformer（ViT）在处理局部细节方面的能力而设计。

## 项目简介

Vision Transformer (ViT) 在建模全局特征方面具有显著优势，但在需要精细局部表示的检测任务中表现有限。为此，我们提出 **EPFPN**，一种边缘渐进特征金字塔网络，旨在通过优化特征融合策略，增强模型捕捉多尺度与细粒度特征的能力。

EPFPN 采用 **自底向上 + 自顶向下的渐进式特征融合机制**，有效缩小语义鸿沟，避免信息丢失。同时，引入创新的 **EPFusion 模块**，通过：
- **边缘级注意力（Edge-Level Attention, ELA）**
- **像素加权注意力（Pixel-wise Attention）**

显著提升了 ViT 对局部细节的建模能力，尤其适用于处理高分辨率图像中的小目标。

> 在 Cascade Mask R-CNN + ViT-B 框架中，EPFPN 带来了 +0.7% box AP 和 +0.6% mask AP 的提升，显示出其良好的通用性和增强能力。

## 特性 Features

- 支持多种目标检测器（如 Faster R-CNN、Cascade R-CNN）
- 设计兼容 ViT 主干网络
- 引入边缘感知与像素级注意力机制
- 显著提升小目标检测性能
- 易于与主流检测框架集成（如 MMDetection）

## 安装 Installation

```bash
# 克隆仓库
git clone https://github.com/yunxing163/EPFPN.git

# 安装依赖（建议使用conda）
pip install -r requirements.txt
