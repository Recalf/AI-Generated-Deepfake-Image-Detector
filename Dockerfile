
# Build from repo root: docker build -t ai-image-detector .
FROM python:3.13-slim


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app code (folder name has a space)
COPY "AI Images Detector"/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY "AI Images Detector"/inference.py .
COPY "AI Images Detector"/transforms.py .
COPY "AI Images Detector"/model ./model
COPY "AI Images Detector"/.streamlit ./.streamlit

# Checkpoints are large; create dir and mount at runtime (see README)
RUN mkdir -p checkpoints

EXPOSE 8501

# Bind to 0.0.0.0 so the app is reachable from outside the container
CMD ["streamlit", "run", "inference.py", "--server.address=0.0.0.0", "--server.port=8501"]
