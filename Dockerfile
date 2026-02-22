# Build from repo root: docker build -t ai-image-detector .
FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    curl \                                  # for healthcheck
    && rm -rf /var/lib/apt/lists/*          # remove apt cache, lighter image

COPY "AI Images Detector"/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY . . # this after copy requirements.txt so that if we change something it wont rerun pip install again when building

# checkpoints are large, we'll create a directory and mount at runtime (see README)
RUN mkdir -p checkpoints

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health # periodic healthcheck

CMD ["streamlit", "run", "inference.py", "--server.address=0.0.0.0", "--server.port=8501"] # server.address=0.0.0.0 allows connection outside of the container
