# StarNet Pre-training on ImageNet

这个项目包含了在 ImageNet 数据集上预训练 StarNet 的代码。StarNet 是一个新的卷积神经网络架构，旨在通过 star operation 来提升网络性能。

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- tqdm
- CUDA 支持（推荐 11.0+）

```bash
pip install torch torchvision tqdm
```

## 数据准备

1. 下载 ImageNet 数据集
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

## 性能评估

本项目提供了一个评估，可以比较不同模型在 ImageNet 验证集上的性能。当前支持的模型包括 StarNet S4 和 ResNet50。评估结果包括 Top-1 和 Top-5 准确率以及评估所需时间。

### 使用方法

1. 确保您已准备好 ImageNet 验证集，并将其路径传递给评估工具。
2. 运行以下命令以评估模型：

```bash
python comparison.py --val_dir /path/to/imagenet/val
```

如果未指定 `--val_dir` 参数，默认将使用 `../imagenet/val` 作为验证集路径。

### 引用

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