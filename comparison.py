import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.ResNet import Backbone_ResNet50
from backbone.StarNet import Backbone_StarNetS4


class ImageNetEvaluator:
    def __init__(self, val_dir, batch_size=256, num_workers=8):
        # ImageNet验证集预处理
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

        # 加载ImageNet验证集
        self.val_dataset = datasets.ImageFolder(val_dir, self.val_transform)

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

                # Top-1 and Top-5 accuracy
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


class FullModel(nn.Module):
    """包装backbone并添加分类头"""

    def __init__(self, backbone_fn, pretrained=True, num_classes=1000):
        super().__init__()
        # 获取backbone的各个部分
        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = backbone_fn(
            pretrained=pretrained
        )

        # 获取最后一层的通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dummy = self.div_2(dummy)
            dummy = self.div_4(dummy)
            dummy = self.div_8(dummy)
            dummy = self.div_16(dummy)
            dummy = self.div_32(dummy)
            last_channels = dummy.shape[1]

        # 添加分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channels, num_classes)

        # 初始化fc层
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.div_2(x)
        x = self.div_4(x)
        x = self.div_8(x)
        x = self.div_16(x)
        x = self.div_32(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    # ImageNet验证集路径
    val_dir = "../imagenet/val"  # 请修改为实际路径

    # 创建评估器
    evaluator = ImageNetEvaluator(val_dir)

    # 评估StarNet S4
    print("\nEvaluating StarNet S4...")
    starnet = FullModel(Backbone_StarNetS4, pretrained=True)
    starnet_results = evaluator.evaluate_model(starnet, "StarNet S4")

    # 评估ResNet50
    print("\nEvaluating ResNet50...")
    resnet = FullModel(Backbone_ResNet50, pretrained=True)
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


if __name__ == "__main__":
    main()
