import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from StarNet import StarNet50


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_model(args):
    # 创建模型
    model = StarNet50().cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )

    # 数据增强
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 数据集
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # 训练循环
    best_acc1 = 0
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)

        # 验证
        acc1 = validate(val_loader, model, criterion, args)

        scheduler.step()

        # 记录最佳模型
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.save_dir,
        )


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 计算输出
        output = model(images)
        loss = criterion(output, target)

        # 计算准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 记录loss和准确率
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 计算梯度和优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 测量时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f}) "
                f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
            )


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # 计算输出
            output = model(images)
            loss = criterion(output, target)

            # 计算准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # 记录loss和准确率
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    f"Test: [{i}/{len(val_loader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc@1 {top1.val:.3f} ({top1.avg:.3f}) "
                    f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
                )

        print(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")

    return top1.avg


def save_checkpoint(state, is_best, save_dir):
    # 只保存模型权重
    filename = os.path.join(save_dir, "checkpoint.pth")
    torch.save(state["state_dict"], filename)
    if is_best:
        best_filename = os.path.join(save_dir, "model_best.pth")
        import shutil

        shutil.copyfile(filename, best_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StarNet ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, help="number of data loading workers"
    )
    parser.add_argument(
        "--epochs", default=300, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="mini-batch size"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--weight-decay", "--wd", default=0.05, type=float, help="weight decay"
    )
    parser.add_argument(
        "--print-freq", "-p", default=10, type=int, help="print frequency"
    )
    parser.add_argument(
        "--save-dir", default="checkpoints", type=str, help="path to save checkpoints"
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_model(args)
