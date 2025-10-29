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
from model.mtl_cnn import mtl_cnn_v1
from model.mobilenet_v4 import CustomMobileNetV4
from model.efficientnet_b0 import efficientnet_b0_model

_WORKER=8 # VGA mạnh thì tăng lên
_NUM_CLASSES=33

def build_loaders(batch_size=32, workers=_WORKER):
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = getAllDataset()
    train_dataset = FoodVNDs(train_paths, train_labels, transform=train_transform)
    val_dataset   = FoodVNDs(val_paths,   val_labels,   transform=test_transform)
    test_dataset  = FoodVNDs(test_paths,  test_labels,  transform=test_transform)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=pin,persistent_workers=True)
    valid_loader = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin,persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=1,  shuffle=False, num_workers=workers, pin_memory=pin)
    return train_loader, valid_loader, test_loader

def build_model(name: str, num_classes: int):
    name = name.lower()
    #if name == "cnn":
        #return miniVGG()                              # đầu ra 33 lớp viết sẵn trong cnn.py :contentReference[oaicite:1]{index=1}
    if name == "mtl-cnn":
        return mtl_cnn_v1(num_classes=num_classes)     # dùng factory ở trên
    elif name == "vgg16":
        return vgg16(pretrained=True)
    elif name == "resnet18":
        return resnet18(pretrained=True)
    elif name == "mobilenetv4":
        return CustomMobileNetV4(num_classes=num_classes, pretrained=True, freeze_backbone=False)
    elif name == "efficientnet" or name == "efficientnet_b0":
        return efficientnet_b0_model(num_classes=_NUM_CLASSES,
                                  pretrained=True,          # fine-tune
                                  freeze_backbone=False)    # True nếu muốn warmup
    else:
        raise ValueError(f"Unknown model: {name}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mtl-cnn","mobilenet","efficientnet_b0","vgg16","resnet18","mobilenetv4"], default="mobilenet",
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

    train_loader, valid_loader, test_loader = build_loaders(batch_size=args.batch_size, workers=_WORKER)

    name = args.model.lower()  # "cnn", "mobilenet", ...
    pretty = {"mtl-cnn":"mtl-cnn","mobilenetv4":"mtl-mobilenetv4","efficientnet_b0":"mtl-efficientnet_b0","resnet18":"mtl-resnet18"}.get(name, type(model).__name__)
    setattr(model, "_export_name", pretty)   # gắn nhãn cho model trước khi train
   
    #fit(model, train_loader, valid_loader, test_loader,
    #    max_epochs=args.epochs, max_plateau_count=15, wb=False, device=device)
    

     # === thêm meta để lưu vào config.json trong runs/ ===
    run_meta = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_classes": NUM_CLASSES,
        # bạn có thể bổ sung img_size, mean/std nếu có:
        # "img_size": 224,
        # "normalize": {"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}
    }

    fit(model, train_loader, valid_loader, test_loader,
        max_epochs=args.epochs, max_plateau_count=15, wb=False, device=device,
        run_root="runs", run_meta=run_meta)



if __name__ == "__main__":
    import torch.multiprocessing as mp    
    mp.freeze_support()
    main()


   

   
