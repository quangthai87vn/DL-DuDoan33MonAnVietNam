# train_mobilenetv4.py
import os, time, json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import timm

# ========== CẤU HÌNH ==========
SEED          = 42
DATA_ROOT     = r"C:/TRAIN/Deep Learning/vietnamese-foods/Images"  # <== Sửa lại thư mục gốc
TRAIN_DIR     = os.path.join(DATA_ROOT, "Train")
VAL_DIR       = os.path.join(DATA_ROOT, "Validate")  # Nếu không có, script sẽ tự split từ Train
TEST_DIR      = os.path.join(DATA_ROOT, "Test")

IMG_SIZE      = 224
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 5e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
PATIENCE      = 10          # EarlyStopping
CKPT_DIR      = "./checkpoints"
CKPT_BEST     = os.path.join(CKPT_DIR, "classification_best.pt")

MODEL_NAME    = "mobilenetv4_conv_small.e600_r224"   # timm model name (vd: mobilenetv4_conv_medium.e600_r224)
USE_AMP       = True          # Mixed precision

os.makedirs(CKPT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ========== TRANSFORMS ==========
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(),                   # tăng cường dữ liệu
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

# ========== DATASET / DATALOADER ==========
if os.path.isdir(VAL_DIR) and len(os.listdir(VAL_DIR))>0:
    print("✅ Dùng thư mục Validate có sẵn.")
    ds_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    ds_val   = datasets.ImageFolder(VAL_DIR,   transform=test_tf)
else:
    print("ℹ️  Không có thư mục Validate. Tách 10% từ Train làm Val.")
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    n_val = max(1, int(0.1*len(full_train)))
    n_train = len(full_train) - n_val
    ds_train, ds_val = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    # IMPORTANT: set transforms for val split (random_split keeps same transform object)
    ds_val.dataset.transform = test_tf

ds_test = datasets.ImageFolder(TEST_DIR, transform=test_tf)

CLASS_NAMES = ds_train.dataset.classes if hasattr(ds_train, "dataset") else ds_train.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Số lớp: {NUM_CLASSES} — Ví dụ: {CLASS_NAMES[:min(5,NUM_CLASSES)]}")

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Lưu class mapping
with open(os.path.join(CKPT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)

# ========== MODEL ==========
# Lấy backbone từ timm, head linear sẽ tự build theo num_classes
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model.to(device)

# ========== OPTIM / LOSS / SCHEDULER ==========
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ========== TRAIN & VALIDATE ==========
def accuracy_topk(logits, target, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / target.size(0)).item())
    return res

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct1, correct5, loss_sum = 0, 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)
        acc1, acc5 = accuracy_topk(logits, y, topk=(1,5 if NUM_CLASSES>=5 else 1))
        bs = y.size(0)
        total    += bs
        loss_sum += loss.item() * bs
        correct1 += acc1 * bs
        correct5 += (acc5 if NUM_CLASSES>=5 else acc1) * bs
    return loss_sum/total, correct1/total, correct5/total

def train_one_epoch(epoch):
    model.train()
    total, loss_sum, correct1 = 0, 0.0, 0.0
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, _ = accuracy_topk(logits, y, topk=(1,1))
        bs = y.size(0)
        total    += bs
        loss_sum += loss.item() * bs
        correct1 += acc1 * bs

    scheduler.step()
    dt = time.time()-t0
    return loss_sum/total, correct1/total, dt

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc, t = train_one_epoch(epoch)
    val_loss, val_acc1, val_acc5 = evaluate(val_loader)

    print(f"[Epoch {epoch:03d}] "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% | "
          f"val_loss={val_loss:.4f} val_top1={val_acc1*100:.2f}% val_top5={val_acc5*100:.2f}% | "
          f"time={t:.1f}s")

    # Early stopping + save best
    if val_acc1 > best_val_acc:
        best_val_acc = val_acc1
        epochs_no_improve = 0
        state = {
            "model_name": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "img_size": IMG_SIZE,
        }
        torch.save({"classification_best": state}, CKPT_BEST)
        print(f"  ✅ Saved best to {CKPT_BEST}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("  ⏹ Early stopping!")
            break

print("Done training.")
