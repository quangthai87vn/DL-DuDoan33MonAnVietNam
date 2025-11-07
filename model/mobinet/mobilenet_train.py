# mobilenet_train_enrnptys.py  — bản chống treo & log chi tiết

import os, time
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")  # tuỳ chọn: nơi cache model nếu có

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # không kẹt vì ảnh hỏng

torch.backends.cudnn.benchmark = True   # tăng tốc khi input cố định

from mobilenet_model import CustomMobileNet  # đảm bảo weights=None trong file này!

# ===============================
# HÀM TRAIN
# ===============================
def train(train_dir, num_epochs, batch_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- Index dataset + log thời gian
    print("🔎 Đang index dataset...", flush=True)
    t0 = time.time()
    dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    print(f"✅ Index xong: {len(dataset)} ảnh, {len(dataset.classes)} lớp: {dataset.classes} "
          f"in {time.time()-t0:.1f}s", flush=True)

    # ---- Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    # ---- DataLoader 'an toàn' cho Windows
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin, persistent_workers=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin, persistent_workers=False)

    # ---- Lưu label map
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open('label.txt', 'w', encoding='utf-8') as f:
        for idx, class_name in enumerate(dataset.classes):
            f.write(f'{idx}: {class_name}\n')

    # ---- Model / loss / optim
    num_classes = len(dataset.classes)
    model = CustomMobileNet(num_classes=num_classes).to(device)  # NHỚ: weights=None trong mobilenet_model.py
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- Thử 1 mini-iter để phát hiện ảnh lỗi sớm
    print("🧪 Kiểm tra đọc 1 batch đầu...")
    try:
        _x, _y = next(iter(train_loader))
        print(f"✅ Batch mẫu: {_x.shape}")
    except Exception as e:
        print("❌ Lỗi khi đọc batch đầu. Có thể do ảnh hỏng hoặc đường dẫn sai.")
        raise e

    best_acc = 0.0
    print("🚀 Bắt đầu train ...", flush=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bi, (inputs, labels) in enumerate(train_loader):
            if bi % 10 == 0:
                print(f"  ... epoch {epoch+1}/{num_epochs} | batch {bi}", flush=True)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = 100. * correct / total

        # ---- Eval
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = 100. * correct / total

        print(f"📈 Epoch [{epoch+1}/{num_epochs}] "
              f"TrainLoss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"ValLoss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path} (Val Acc: {best_acc:.2f}%)")

# ===============================
# CẤU HÌNH MẶC ĐỊNH
# ===============================
if __name__ == "__main__":
    train_dir  = r"C:\TRAIN\Deep Learning\vietnamese-foods\Images\Train"  # đổi theo máy bạn
    num_epochs = 100
    batch_size = 32
    
    
    # Lấy thư mục gốc của file đang chạy
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Bước 1: Lấy đường dẫn gốc của project (lùi ra 2 cấp: mobinet → model → project)
    PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    # Bước 2: Đặt đường dẫn model_path ra thư mục checkpoints ngoài project
    model_path = os.path.join(PROJECT_DIR, "checkpoints", "mtl-mobilenet.pth")
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    #model_path =  "../checkpoints/mtl-mobilenet.pth"  # dùng đường dẫn tương đối + tạo thư mục tự động


    print("🚀 Bắt đầu huấn luyện MobileNet...")
    print(f"📂 Dataset: {train_dir}")
    print(f"📦 Model lưu tại: {model_path}")
    print(f"🧮 Epochs: {num_epochs}, Batch size: {batch_size}")

    train(train_dir=train_dir,
          num_epochs=num_epochs,
          batch_size=batch_size,
          model_path=model_path)
