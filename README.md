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
