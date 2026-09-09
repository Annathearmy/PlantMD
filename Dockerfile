FROM tensorflow/tensorflow:2.3.1

WORKDIR /app

# Install API dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir \
    fastapi==0.61.1 \
    uvicorn==0.12.3 \
    pillow==8.0.1 \
    numpy==1.18.5 \
    python-multipart==0.0.5

# Create model directory
RUN mkdir -p /app/models

# Download the actual Git LFS model
RUN curl -L \
    "https://media.githubusercontent.com/media/Annathearmy/PlantMD/master/models/AlexNetModel.hdf5" \
    -o /app/models/AlexNetModel.hdf5

# Copy our API
COPY app.py .

# Cloud Run uses port 8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
