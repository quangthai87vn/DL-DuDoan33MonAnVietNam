# utils/trainer.py
from __future__ import annotations
import os, time, traceback
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm

# bật trace CUDA rõ ràng khi debug
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

#Name_food = {0: "Banh mi", 1: "Com tam", 2: "Hu tieu", 3: "Pho"}



Name_food = {0: 'Banh beo',
 1:'Banh bot loc',
 2:'Banh can',
 3:'Banh canh',
 4:'Banh chung',
 5:'Banh cuon',
 6:'Banh duc',
 7:'Banh gio',
 8:'Banh khot',
 9:'Banh mi',
 10:'Banh pia',
 11:'Banh tet',
 12:'Banh trang nuong',
 13:'Banh xeo',
 14:'Bun bo Hue',
 15:'Bun dau mam tom',
 16:'Bun mam',
 17:'Bun rieu',
 18:'Bun thit nuong',
 19:'Ca kho to',
 20:'Canh chua',
 21:'Cao lau',
 22:'Chao long',
 23:'Com tam',
 24:'Goi cuon',
 25:'Hu tieu',
 26:'Mi quang',
 27:'Nem chua',
 28:'Pho',
 29:'Xoi xeo'}


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0

def _device_from_model(model: nn.Module) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda" if torch.cuda.is_available() else "cpu"

def _ckpt_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, "classification_best.pt")

def init_wandb(wb: bool, model: nn.Module, criterion: nn.Module):
    if not wb: return None
    try:
        import wandb
        wandb.login()
        wandb.init(project="classifi_FoodVN",
                   name=f"miniVGG_adam_{int(time.time())}",
                   config={"batch_size":128,"learning_rate":1e-4,"epoch":50})
        try: wandb.watch(model, criterion, log="all", log_freq=10)
        except Exception: pass
        return wandb
    except Exception:
        traceback.print_exc(); return None

def _sanity_check_labels(outputs: torch.Tensor, labels: torch.Tensor):
    # gọi ở batch đầu để bắt lỗi label
    C = outputs.shape[1]
    if labels.dtype != torch.long:
        print("⚠️  labels.dtype =", labels.dtype, "→ ép về long")
        labels = labels.long()
    if labels.min().item() < 0 or labels.max().item() >= C:
        raise ValueError(f"❌ Label ngoài phạm vi [0..{C-1}]. "
                         f"Min={labels.min().item()}, Max={labels.max().item()}, C={C}")
    return labels

def train(model, train_loader, optimizer, criterion, epoch, *,
          device: Optional[str]=None, wandb=None, wb=False) -> Dict[str,float]:
    model.train()
    device = device or _device_from_model(model)
    running_loss=0.0; correct=0; total=0

    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                     leave=True, colour="blue", desc=f"Epoch {epoch}",
                     bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for i,(images,labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)

        if epoch==1 and i==0:  # sanity check batch đầu
            print("DEBUG outputs:", outputs.shape, "labels unique:", labels.unique().tolist())
            labels = _sanity_check_labels(outputs, labels)

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss/(i+1); acc = 100.0*correct/max(1,total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%", lr=f"{get_lr(optimizer):.2e}")
        if wb and wandb is not None:
            wandb.log({"train/loss": avg_loss, "train/acc": acc, "lr": get_lr(optimizer), "epoch": epoch})

    return {"loss": running_loss/max(1,len(train_loader)), "acc": 100.0*correct/max(1,total)}

@torch.no_grad()
def val(model, valid_loader, optimizer, criterion, epoch, *,
        device: Optional[str]=None, wandb=None, wb=False) -> float:
    model.eval()
    device = device or _device_from_model(model)
    running_loss=0.0; correct=0; total=0

    pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader),
                     leave=True, colour="green", desc="Val",
                     bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for i,(images,labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)

        if epoch==1 and i==0:
            labels = _sanity_check_labels(outputs, labels)

        loss = criterion(outputs, labels.long())
        running_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss/(i+1); acc = 100.0*correct/max(1,total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")

    val_loss = running_loss/max(1,len(valid_loader))
    val_acc  = 100.0*correct/max(1,total)
    if wb and wandb is not None: wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})
    print(f" Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_acc:.2f}%")
    return float(val_acc)

@torch.no_grad()
def test(model, test_loader, optimizer, criterion, epoch, *,
         device: Optional[str]=None, wandb=None, wb=False) -> Dict[str,float]:
    model.eval()
    device = device or _device_from_model(model)
    running_loss=0.0; correct=0; total=0

    pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                     leave=True, colour="cyan", desc="Test",
                     bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for i,(images,labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        running_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss/(i+1); acc = 100.0*correct/max(1,total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%" )

    test_loss = running_loss/max(1,len(test_loader))
    test_acc  = 100.0*correct/max(1,total)
    if wb and wandb is not None: wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    print("--------TESTING--------")
    print(f" Test_Loss: {test_loss:.4f}, Test_Accuracy: {test_acc:.2f}%")
    return {"loss": test_loss, "acc": test_acc}

def save_weights(model: nn.Module, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"net": model.state_dict()}, checkpoint_path)

def fit(model, train_loader, valid_loader, test_loader,
        max_epochs: int=50, max_plateau_count: int=2, wb: bool=False, device: Optional[str]=None):
    device = device or _device_from_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, min_lr=0.0)

    wandb = init_wandb(wb, model, criterion)
    ckpt_path = _ckpt_path()
    best_acc_val = -1.0; plateau_count = 0; epochs = 0

    try:
        while (plateau_count <= max_plateau_count) and (epochs < max_epochs):
            epochs += 1
            tr = train(model, train_loader, optimizer, criterion, epochs, device=device, wandb=wandb, wb=wb)
            val_acc = val(model, valid_loader, optimizer, criterion, epochs, device=device, wandb=wandb, wb=wb)

            if val_acc > best_acc_val:
                best_acc_val = val_acc; plateau_count = 0; save_weights(model, ckpt_path)
            else:
                plateau_count += 1

            loss_proxy = 100.0 - val_acc  # proxy cho val_loss
            scheduler.step(loss_proxy)
            print(f"[Epoch {epochs:03d}] train_loss={tr['loss']:.4f} train_acc={tr['acc']:.2f}% | "
                  f"val_acc={val_acc:.2f}% | lr={get_lr(optimizer):.2e} | plateau={plateau_count}/{max_plateau_count}")
    except KeyboardInterrupt:
        print("\n⛔ Dừng training theo yêu cầu người dùng."); traceback.print_exc()
    except Exception:
        traceback.print_exc()

    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["net"])
        print("Accuracy on Private Test:")
        test(model, test_loader, optimizer, criterion, epochs, device=device, wandb=wandb, wb=wb)
    except Exception:
        print("⚠️  Không tìm thấy/không load được checkpoint, test với trọng số hiện tại.")
        traceback.print_exc()
        test(model, test_loader, optimizer, criterion, epochs, device=device, wandb=wandb, wb=wb)
