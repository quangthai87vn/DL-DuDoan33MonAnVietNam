
# classifi_main.py
from __future__ import print_function, division
import os
import argparse
import torch
from torch.utils.data import DataLoader

# ====== project imports ======
from utils.processing import *                         # transforms + Name_food + getAllDataset
from utils.vnfood_ds import *                          # FoodVNDs (giữ nguyên như cũ)
from utils.trainer import fit                          # vòng train/val/test chuẩn
from model.cnn import miniVGG                          # CNN gốc
from model.vggnet import vgg16                         # (nếu bạn đang dùng)
from model.resnet import resnet18                      # (nếu bạn đang dùng)
from model.mobilenet import mobilenet_v1               # <-- file mới tạo

def build_loaders(batch_size=32, workers=0):
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = getAllDataset()
    train_dataset = FoodVNDs(train_paths, train_labels, transform=train_transform)
    val_dataset   = FoodVNDs(val_paths,   val_labels,   transform=test_transform)
    test_dataset  = FoodVNDs(test_paths,  test_labels,  transform=test_transform)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=pin)
    valid_loader = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader  = DataLoader(test_dataset, batch_size=1,  shuffle=False, num_workers=workers, pin_memory=pin)
    return train_loader, valid_loader, test_loader

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "cnn":
        return miniVGG()                                 # đầu ra 33 lớp viết sẵn trong cnn.py :contentReference[oaicite:1]{index=1}
    elif name == "mobilenet":
        return mobilenet_v1(num_classes=num_classes)     # dùng factory ở trên
    elif name == "vgg16":
        return vgg16(pretrained=True)
    elif name == "resnet18":
        return resnet18(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn","mobilenet","vgg16","resnet18"], default="mobilenet",
                        help="Chọn mô hình để train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    NUM_CLASSES = len(Name_food)  # lấy từ utils.processing.Name_food :contentReference[oaicite:2]{index=2}
    model = build_model(args.model, NUM_CLASSES).to(device)

    train_loader, valid_loader, test_loader = build_loaders(batch_size=args.batch_size, workers=0)

    #fit(model, train_loader, valid_loader, test_loader,
    #    max_epochs=args.epochs, max_plateau_count=15, wb=False, device=device)  # dùng trainer như cũ :contentReference[oaicite:3]{index=3}
    

    name = args.model.lower()  # "cnn", "mobilenet", ...
    pretty = {"cnn":"MTL-CNN", "mobilenet":"MTL-MobiNet"}.get(name, type(model).__name__)
    setattr(model, "_export_name", pretty)   # gắn nhãn cho model trước khi train
    #fit(model, train_loader, valid_loader, test_loader,
    #    max_epochs=args.epochs, max_plateau_count=15, wb=False, device=device,
    #    ckpt_dir="checkpoints", model_name=pretty)

    fit(model, train_loader, valid_loader, test_loader,
        max_epochs=args.epochs, max_plateau_count=15, wb=False, device=device)

if __name__ == "__main__":
    import torch.multiprocessing as mp    
    mp.freeze_support()
    main()
