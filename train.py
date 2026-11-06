"""Train the ImageNet-20 CNN from scratch.

Expected data layout:
    --train-root  folder with the training images (class sub-folders are fine)
    --val-root    folder with the validation images
    --train-list  text file, one "<relative path> <label>" per line
    --val-list    same for validation
Images are matched to the list by file name.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import ImageNet20Net

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_EXT = (".jpeg", ".jpg", ".png")


class ImageNet20Dataset(Dataset):
    def __init__(self, root, list_file, transform=None):
        index = {}
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith(IMG_EXT):
                    index[name] = os.path.join(dirpath, name)
        self.samples = []
        with open(list_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name = os.path.basename(parts[0])
                if name in index:
                    self.samples.append((index[name], int(parts[1])))
        self.transform = transform
        print(f"Loaded {len(self.samples)} samples from {list_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return (self.transform(img) if self.transform else img), label


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--size", type=int, default=240)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.size + 16),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_loader = DataLoader(ImageNet20Dataset(args.train_root, args.train_list, train_tf),
                              batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ImageNet20Dataset(args.val_root, args.val_list, val_tf),
                            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ImageNet20Net(20).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tl, ta, vl, va])
        flag = ""
        if va > best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            flag = " *best*"
        print(f"{epoch:3d} | train {tl:.4f} {ta*100:5.2f}% | val {vl:.4f} {va*100:5.2f}%{flag}")
    print(f"Best val acc: {best*100:.2f}%")


if __name__ == "__main__":
    main()
