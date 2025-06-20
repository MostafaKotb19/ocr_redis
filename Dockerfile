# pdf_to_redis_parser/Dockerfile

# 1. Base Image: Use python:3.9-slim as originally planned. 3.13 might be too new for stable paddle deps.
# Python 3.9 or 3.10 are generally safer bets for complex ML libraries.
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
    # libopencv-dev # Usually not needed if paddle installs its own cv2, but can be a fallback
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Copy requirements.txt and Install Python Packages
COPY requirements.txt .
# Install paddlepaddle from requirements.txt, ensure it's listed there.
# The separate `RUN python -m pip install paddlepaddle` might be redundant or cause version conflicts
# if paddlepaddle is also in requirements.txt. Best to manage all python deps via requirements.txt.
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --no-cache-dir -r requirements.txt


# 6. Copy Project Code
COPY src/ /app/src/

# 7. Define Entrypoint/Command
ENTRYPOINT ["python", "-m", "src.main_parser"]