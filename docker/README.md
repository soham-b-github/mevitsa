# Docker setup
## Build and run

```bash
# Build the image
docker build -t mevitsa:latest .

# Run container with GPU support and volume mounting
# Note: You must place the .pt weights in src/trained/ before running
docker run -d -p 8501:8501 --init --rm -it --gpus=all --ipc=host \
  --user=$(id -u):$(id -g) \
  -e HOME=/app \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --name="Mevitsa_UI" --env TZ=UTC \
  --volume=$PWD:/app \
  mevitsa:latest
```

## Usage
```bash
# Frontend UI
streamlit run src/app.py

# Evaluation
python3 src/evaluation.py
