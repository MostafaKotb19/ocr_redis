# 1. Base Image
FROM python:3.11-slim 

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/app

# 3. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Copy requirements.txt and Install Python Packages
COPY requirements.txt .

# Install PaddlePaddle with GPU support
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Project Code
COPY src/ /app/src/

# 7. Define Entrypoint/Command
ENTRYPOINT ["python", "-m", "src.main_parser"]