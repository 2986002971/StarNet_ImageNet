import argparse
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.StarNet import StarNetBackbone


class StarNet(nn.Module):
    """完整的StarNet模型，包含分类头"""

    def __init__(
        self,
        base_dim=32,
        depths=[3, 3, 12, 5],
        mlp_ratio=4,
        drop_path_rate=0.0,
        num_classes=1000,
    ):
        super().__init__()
        # backbone
        self.backbone = StarNetBackbone(base_dim, depths, mlp_ratio, drop_path_rate)

        # 分类头
        self.norm = nn.BatchNorm2d(base_dim * 2 ** (len(depths) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_dim * 2 ** (len(depths) - 1), num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]  # 取最后一个特征图
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def load_starnet():
    """加载完整的预训练StarNet S4"""
    model = StarNet(base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4)

    # 加载预训练权重
    checkpoint = torch.hub.load_state_dict_from_url(
        url="https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
        map_location="cpu",
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 处理可能的module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # 移除"module."前缀
        new_state_dict[k] = v

    # 加载权重
    model.load_state_dict(new_state_dict)
    return model


def load_resnet():
    """加载完整的预训练ResNet50"""
    import torchvision.models as models

    model = models.resnet50(pretrained=True)
    return model


class ImageNetEvaluator:
    def __init__(self, val_dir, batch_size=256, num_workers=8):
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.val_dataset = datasets.ImageFolder(val_dir, self.val_transform)

        print(
            f"Found {len(self.val_dataset)} images in {len(self.val_dataset.classes)} classes"
        )
        print(f"First few classes: {self.val_dataset.classes[:5]}")

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_model(self, model, model_name):
        model = model.to(self.device)
        model.eval()

        total = 0
        correct_1 = 0
        correct_5 = 0

        start_time = time.time()

        with torch.no_grad():
            for images, labels in tqdm(
                self.val_loader, desc=f"Evaluating {model_name}"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)

                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))

                correct_1 += correct[0].sum().item()
                correct_5 += correct[:5].sum().item()
                total += labels.size(0)

        acc_1 = 100 * correct_1 / total
        acc_5 = 100 * correct_5 / total
        eval_time = time.time() - start_time

        return {
            "Top-1 Accuracy": acc_1,
            "Top-5 Accuracy": acc_5,
            "Evaluation Time": eval_time,
        }


if __name__ == "__main__":
    # 新增参数解析
    parser = argparse.ArgumentParser(description="ImageNet Evaluator")
    parser.add_argument(
        "--val_dir",
        type=str,
        default="../imagenet/val",
        help="ImageNet validation directory",
    )
    args = parser.parse_args()

    # 使用解析的val_dir
    val_dir = args.val_dir

    # 创建评估器
    evaluator = ImageNetEvaluator(val_dir)

    # 评估StarNet S4
    print("\nEvaluating StarNet S4...")
    starnet = load_starnet()
    starnet_results = evaluator.evaluate_model(starnet, "StarNet S4")

    # 评估ResNet50
    print("\nEvaluating ResNet50...")
    resnet = load_resnet()
    resnet_results = evaluator.evaluate_model(resnet, "ResNet50")

    # 打印结果对比
    print("\n=== Results Comparison ===")
    print(f"{'Model':<15} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Time (s)':<10}")
    print("-" * 50)
    print(
        f"StarNet S4{' ':<6} {starnet_results['Top-1 Accuracy']:.2f}%{' '*4} "
        f"{starnet_results['Top-5 Accuracy']:.2f}%{' '*4} "
        f"{starnet_results['Evaluation Time']:.1f}"
    )
    print(
        f"ResNet50{' ':<7} {resnet_results['Top-1 Accuracy']:.2f}%{' '*4} "
        f"{resnet_results['Top-5 Accuracy']:.2f}%{' '*4} "
        f"{resnet_results['Evaluation Time']:.1f}"
    )
