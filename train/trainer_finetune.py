import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd


class FineTuneTrainer:
    """
    Trainer 2 pha:
    - Phase 1: Freeze backbone, train classifier (warm-up)
    - Phase 2: Unfreeze toàn bộ backbone, fine-tune
    """

    def __init__(self, model, train_loader, val_loader, device,
                 lr_phase1=1e-3, lr_phase2=3e-4,
                 weight_decay=1e-5,
                 warmup_epochs=10,
                 total_epochs=40,
                 save_dir="./runs/finetune_efficientnet"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_phase1 = lr_phase1
        self.lr_phase2 = lr_phase2
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()

    def freeze_backbone(self, freeze=True):
        for name, param in self.model.named_parameters():
            if "classifier" not in name:  # chỉ giữ classifier mở
                param.requires_grad = not freeze

    def train_one_epoch(self, optimizer):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(self.train_loader, desc="Train", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(self.val_loader, desc="Validate", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def fit(self):
        history = []
        best_acc = 0.0
        phase = 1

        print(f"=== Phase 1: Warm-up classifier ({self.warmup_epochs} epochs, lr={self.lr_phase1}) ===")
        self.freeze_backbone(True)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=self.lr_phase1, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        for epoch in range(1, self.total_epochs + 1):
            start = time.time()
            train_loss, train_acc = self.train_one_epoch(optimizer)
            val_loss, val_acc = self.evaluate()
            scheduler.step(val_acc)

            history.append([epoch, phase, train_loss, val_loss, train_acc, val_acc])
            print(f"Epoch {epoch:03d}/{self.total_epochs} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Time: {(time.time()-start)/60:.1f}m")

            # === Chuyển pha ===
            if epoch == self.warmup_epochs:
                print(f"\n=== Phase 2: Unfreeze backbone (lr={self.lr_phase2}) ===\n")
                phase = 2
                self.freeze_backbone(False)
                optimizer = AdamW(self.model.parameters(), lr=self.lr_phase2, weight_decay=self.weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)

            # === Lưu checkpoint tốt nhất ===
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "val_acc": val_acc
                }, os.path.join(self.save_dir, "best.mtl"))
                print(f"✅ Saved new best: val_acc={best_acc:.3f}")

        # Lưu lịch sử
        df = pd.DataFrame(history, columns=["epoch", "phase", "train_loss", "val_loss", "train_acc", "val_acc"])
        df.to_csv(os.path.join(self.save_dir, "history.csv"), index=False)
        print("📊 Training completed. History saved to history.csv")

        return df
