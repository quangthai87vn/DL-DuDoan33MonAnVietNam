# utils/trainer.py
from __future__ import annotations
import os, time, traceback
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm

import csv, json, datetime
from pathlib import Path
import matplotlib.pyplot as plt

# bật trace CUDA rõ ràng khi debug
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False



# Thiết lập style tqdm mặc định
tqdm_params = dict(
    leave=True,
    ncols=100,
    dynamic_ncols=True,
    smoothing=0.1
)



Name_food = {
                0: 'Banh beo',
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
                29:'Xoi xeo',
                30:'banh_da_lon',
                31:'banh_tieu',
                32:'banh_trung_thu'
            }

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0

def _device_from_model(model: nn.Module) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda" if torch.cuda.is_available() else "cpu"

def _ckpt_path(model: Optional[nn.Module] = None, ext: str = ".mtl") -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_name = getattr(model, "_export_name", type(model).__name__) if model is not None else "classification"
    return os.path.join(ckpt_dir, f"{model_name}_best{ext}")



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



# ==== LOGGING & RUN UTILITIES ====
import csv, json, datetime
from pathlib import Path
import matplotlib.pyplot as plt

def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def create_run_dir(model_name: str, root="runs") -> Path:
    run_dir = Path(root) / f"{model_name}-{_now_tag()}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_dir

def init_history():
    return {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

def append_and_flush_history(history: dict, run_dir: Path):
    # JSON (overwrite)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # CSV (append)
    csv_path = run_dir / "history.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch","train_loss","val_loss","train_acc","val_acc","lr"])
        i = len(history["epoch"]) - 1
        w.writerow([
            history["epoch"][i],
            f"{history['train_loss'][i]:.6f}",
            f"{history['val_loss'][i]:.6f}",
            f"{history['train_acc'][i]:.6f}",
            f"{history['val_acc'][i]:.6f}",
            f"{history['lr'][i]:.8f}",
        ])

def save_config(run_dir: Path, config: dict):
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def plot_curves(run_dir: Path, history: dict, filename="loss_accuracy.png"):
    ep = history["epoch"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(ep, history["train_loss"], label="Train")
    axs[0].plot(ep, history["val_loss"], label="Val")
    axs[0].set_title("Loss theo Epoch"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
    axs[0].legend(); axs[0].grid(alpha=.3)

    axs[1].plot(ep, [x*100 for x in history["train_acc"]], label="Train")
    axs[1].plot(ep, [x*100 for x in history["val_acc"]], label="Val")
    axs[1].set_title("Accuracy (%) theo Epoch"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend(); axs[1].grid(alpha=.3)

    plt.tight_layout()
    out = run_dir / "images" / filename
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)



# ==== LOGGING & RUN UTILITIES ====


def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def create_run_dir(model_name: str, root="runs") -> Path:
    run_dir = Path(root) / f"{model_name}-{_now_tag()}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_dir

def init_history():
    return {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

def append_and_flush_history(history: dict, run_dir: Path):
    # JSON (overwrite)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # CSV (append)
    csv_path = run_dir / "history.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch","train_loss","val_loss","train_acc","val_acc","lr"])
        i = len(history["epoch"]) - 1
        w.writerow([
            history["epoch"][i],
            f"{history['train_loss'][i]:.6f}",
            f"{history['val_loss'][i]:.6f}",
            f"{history['train_acc'][i]:.6f}",
            f"{history['val_acc'][i]:.6f}",
            f"{history['lr'][i]:.8f}",
        ])

def save_config(run_dir: Path, config: dict):
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def plot_curves(run_dir: Path, history: dict, filename="loss_accuracy.png"):
    ep = history["epoch"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(ep, history["train_loss"], label="Train")
    axs[0].plot(ep, history["val_loss"], label="Val")
    axs[0].set_title("Loss theo Epoch"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
    axs[0].legend(); axs[0].grid(alpha=.3)

    axs[1].plot(ep, [x*100 for x in history["train_acc"]], label="Train")
    axs[1].plot(ep, [x*100 for x in history["val_acc"]], label="Val")
    axs[1].set_title("Accuracy (%) theo Epoch"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend(); axs[1].grid(alpha=.3)

    plt.tight_layout()
    out = run_dir / "images" / filename
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)



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
        max_epochs: int = 50,
        max_plateau_count: int = 2,
        wb: bool = False,
        device: Optional[str] = None,
        run_root: str = "runs",
        run_meta: Optional[dict] = None):
    """
    Train/val với early-stop theo plateau, ghi lịch sử mỗi epoch (CSV + JSON),
    lưu checkpoint (best/last), và vẽ biểu đồ learning curves.
    Tất cả nằm trong runs/<model>-<timestamp>/
    """
    device = device or _device_from_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, min_lr=0.0)

    wandb = init_wandb(wb, model, criterion)

    # === Tạo RUN DIR ===
    model_name = getattr(model, "_export_name", type(model).__name__)
    run_dir = create_run_dir(model_name, root=run_root)
    print(f"[Run dir] {run_dir}")

    # === Lưu config ban đầu (để reproducible) ===
    cfg = {
        "model": model_name,
        "epochs": max_epochs,
        "optimizer": str(optimizer),
        "scheduler": str(scheduler),
        "device": str(device)
    }
    if run_meta:
        cfg.update(run_meta)  # batch_size, img_size, num_classes, data dirs...
    save_config(run_dir, cfg)

    # === Chuẩn bị lịch sử + ckpt path ===
    history = init_history()
    best_val_acc = -1.0
    plateau_count = 0
    epochs = 0

    ckpt_last = run_dir / "checkpoints" / "last.mtl"
    ckpt_best = run_dir / "checkpoints" / "best.mtl"

    try:
        while (plateau_count <= max_plateau_count) and (epochs < max_epochs):
            epochs += 1

            tr = train(model, train_loader, optimizer, criterion, epochs,
                       device=device, wandb=wandb, wb=wb)

            val_acc = val(model, valid_loader, optimizer, criterion, epochs,
                          device=device, wandb=wandb, wb=wb)

            # === Scheduler step theo proxy (100 - val_acc) ===
            loss_proxy = 100.0 - val_acc
            scheduler.step(loss_proxy)

            # === cập nhật lịch sử ===
            history["epoch"].append(epochs)
            history["train_loss"].append(float(tr["loss"]))
            history["val_loss"].append(float(loss_proxy if isinstance(loss_proxy, (int,float)) else 0.0))  # nếu bạn có val_loss chuẩn, thay bằng biến đó
            history["train_acc"].append(float(tr["acc"]/100.0))
            history["val_acc"].append(float(val_acc/100.0))
            history["lr"].append(float(get_lr(optimizer)))
            append_and_flush_history(history, run_dir)

            # === checkpoint last ===
            torch.save({
                "net": model.state_dict(),
                "epoch": epochs,
                "best_val_acc": best_val_acc,
                "history_csv": str(run_dir / "history.csv"),
                "config_json": str(run_dir / "config.json"),
            }, ckpt_last)

            # === checkpoint best ===
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "net": model.state_dict(),
                    "epoch": epochs,
                    "best_val_acc": best_val_acc,
                    "history_csv": str(run_dir / "history.csv"),
                    "config_json": str(run_dir / "config.json"),
                }, ckpt_best)
                print(f"💾 Saved BEST checkpoint: {ckpt_best.name} (val_acc={val_acc:.2f}%)")
            else:
                plateau_count += 1

            print(f"📊 Epoch {epochs}/{max_epochs} | "
                  f"Train Loss={tr['loss']:.4f} Acc={tr['acc']:.2f}% | "
                  f"Val Acc={val_acc:.2f}% | LR={get_lr(optimizer):.2e} | "
                  f"Plateau {plateau_count}/{max_plateau_count}")

    except KeyboardInterrupt:
        print("\n⛔ Dừng training theo yêu cầu người dùng.")
    except Exception:
        traceback.print_exc()

    # === Vẽ biểu đồ cuối ===
    plot_curves(run_dir, history, filename="loss_accuracy.png")
    print("✅ Đã lưu biểu đồ:", run_dir / "images" / "loss_accuracy.png")
    print("✅ Best Val Acc:", f"{best_val_acc:.2f}%")
    print("✅ Best ckpt   :", ckpt_best)
    print("✅ Last ckpt   :", ckpt_last)
