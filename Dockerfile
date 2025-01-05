# Use a base image with CUDA 12.4 support
FROM nvidia/cuda:12.4.0-base-ubuntu20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file initially to leverage Docker cache for dependencies
COPY requirements.txt .

RUN pip3 install streamlit

# Install PyTorch and Detectron2 with CUDA 12.4 compatibility
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Detectron2 (compatible with your PyTorch and CUDA version)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt || echo "Some packages failed to install, but continuing..."

# Copy the entire application code into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
