# visualize_simplecnn_summary.py
import torch
from torchinfo import summary
from model.cnn import simpleCNN   # đúng đường dẫn dự án của bạn

model = simpleCNN()               # file của bạn đã cố định out=33 lớp
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Input chuẩn của mạng là 3x224x224
summary(
    model, 
    input_size=(1, 3, 224, 224),  # batch=1
    col_names=("input_size", "output_size", "num_params", "kernel_size"),
    depth=5
)


