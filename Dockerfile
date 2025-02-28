FROM python:3.9

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all application files
COPY *.py .
COPY *.html .
COPY mask_detector_model.pth .

# Expose ports for all three services
EXPOSE 8000 8001 9000

# Create a startup script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]