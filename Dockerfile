FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# python + deps hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip

# CÀI TORCH CUDA (chọn đúng cu118/cu121 khớp với image)
RUN pip3 install --no-cache-dir \
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# cài các lib còn lại
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=1111
EXPOSE 1111
CMD ["streamlit", "run", "app.py", "--server.port=1111", "--server.address=0.0.0.0"]
