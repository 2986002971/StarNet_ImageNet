# StarNet Pre-training on ImageNet

这个项目包含了在ImageNet数据集上预训练StarNet的代码。StarNet是一个新的卷积神经网络架构，旨在通过star operation来提升网络性能。

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- CUDA支持（推荐11.0+）

```bash
pip install torch torchvision
```

## 数据准备

1. 下载ImageNet数据集
2. 将数据集组织成如下结构：
```
/path/to/imagenet/
  ├── train/
  │   ├── n01440764/
  │   │   ├── n01440764_10026.JPEG
  │   │   ├── ...
  │   ├── ...
  └── val/
      ├── n01440764/
      │   ├── ILSVRC2012_val_00000293.JPEG
      │   ├── ...
      ├── ...
```

## 训练

基本训练命令：
```bash
python train_starnet.py /path/to/imagenet \
    --batch-size 128 \
    --lr 0.001 \
    --epochs 300 \
    --workers 4 \
    --save-dir checkpoints
```

### 主要参数说明

- `--batch-size`: 批次大小，根据GPU显存调整
- `--lr`: 初始学习率
- `--epochs`: 训练轮数
- `--workers`: 数据加载的进程数
- `--save-dir`: 模型保存路径

### 训练策略

- 优化器：AdamW (lr=0.001, weight_decay=0.05)
- 学习率调度：Cosine Annealing
- 数据增强：
  - RandomResizedCrop(224)
  - RandomHorizontalFlip
  - ColorJitter
  - Normalize

## 模型结构

StarNet的主要特点：
- 使用star operation替代传统的加法操作
- 深度可分离卷积
- 通道注意力机制

## 预训练模型

训练完成后，模型权重将保存在：
- `checkpoints/checkpoint.pth`: 最新的checkpoint
- `checkpoints/model_best.pth`: 验证集上性能最好的模型

## 使用预训练模型

```python
from backbone.StarNet import StarNet50

# 加载预训练模型
model = StarNet50()
checkpoint = torch.load('checkpoints/model_best.pth')
model.load_state_dict(checkpoint)
```

## 性能指标

目标性能：
- Top-1 Accuracy: ~76%（接近ResNet50水平）
- Top-5 Accuracy: ~93%

## 注意事项

1. 显存使用
   - batch_size=128时约需要12GB显存
   - 如果显存不足，可以减小batch_size

2. 训练时间
   - 单卡训练300epochs约需要7-10天
   - 建议使用较大的batch_size以提高训练效率

3. 数据加载
   - 建议将数据集放在SSD上以提高加载速度
   - workers数建议设置为CPU核心数的1-2倍

## 引用

如果您使用了这个预训练模型，请引用原始论文：
```
@misc{ma2024rewritestars,
      title={Rewrite the Stars}, 
      author={Xu Ma and Xiyang Dai and Yue Bai and Yizhou Wang and Yun Fu},
      year={2024},
      eprint={2403.19967},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.19967}, 
}
```
