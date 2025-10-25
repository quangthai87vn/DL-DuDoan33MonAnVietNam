from __future__ import print_function, division
import os
import torch
from torch.utils.data import DataLoader

# ====== project imports ======
from utils.processing import *
from utils.vnfood_ds import *
from utils.trainer import fit
#from model.cnn import miniVGG
from model.vggnet import vgg16
from model.resnet import resnet18

def build_loaders():
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = getAllDataset()
    train_dataset = FoodVNDs(train_paths, train_labels, transform=train_transform)
    val_dataset   = FoodVNDs(val_paths,   val_labels,   transform=test_transform)
    test_dataset  = FoodVNDs(test_paths,  test_labels,  transform=test_transform)

    workers = 0  # Windows nên để 0 khi test
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=workers, pin_memory=pin)
    valid_loader = DataLoader(val_dataset,  batch_size=32, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader  = DataLoader(test_dataset, batch_size=1,  shuffle=False, num_workers=workers, pin_memory=pin)
    return train_loader, valid_loader, test_loader

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    #model = miniVGG().to(device)  
  
    #model = vgg16(pretrained = True)
    model = resnet18(pretrained = True)

    train_loader, valid_loader, test_loader = build_loaders()

    fit(model, train_loader, valid_loader, test_loader,
        max_epochs=50, max_plateau_count=15, wb=False, device=device)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    # gỡ bỏ: os.system("pip install wandb")
    main()
