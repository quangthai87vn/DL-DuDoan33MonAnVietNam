FROM python:3.10-slim

WORKDIR /app

# Cài thư viện hệ thống cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Cài pip & torch GPU + streamlit
RUN pip install --upgrade pip
RUN pip install --no-cache-dir streamlit matplotlib seaborn pandas scikit-learn pillow opencv-python-headless
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Copy code của bạn vào container
COPY ../webapps /app/webapps
COPY ../model /app/model
COPY ../Jupyter /app/Jupyter

EXPOSE 6789

CMD ["streamlit", "run", "webapps/app.py", "--server.port=6789", "--server.address=0.0.0.0"]
