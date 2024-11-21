# AI Background Remover

A simple web application for removing backgrounds from images using AI.

## Requirements

- Docker
- Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU version)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WpythonW/rmbg2.00-inerface.git
cd https://github.com/WpythonW/rmbg2.00-inerface.git
```

2. Create a directory for models:
```bash
mkdir -p models
```

## Running

### CPU Version

```bash
# Production mode
docker compose -f docker-compose.cpu.yml up -d

# Development mode
docker compose -f docker-compose.cpu.yml up -d --build
```

### GPU Version

```bash
# Make sure NVIDIA Container Toolkit is installed
nvidia-smi

# Production mode
docker compose -f docker-compose.gpu.yml up -d

# Development mode
docker compose -f docker-compose.gpu.yml up -d --build
```

### Development Mode

For development, you can use direct launch with code mounting:

```bash
# CPU Version
docker compose -f docker-compose.cpu.yml up -d --build
docker compose -f docker-compose.cpu.yml exec rmbg-cpu bash

# GPU Version
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml exec rmbg-gpu bash
```

## Accessing the Application

After launch, the application will be available at:
```
http://localhost:7860
```

## Stopping

```bash
# CPU Version
docker compose -f docker-compose.cpu.yml down

# GPU Version
docker compose -f docker-compose.gpu.yml down
```

## Project Structure

```
.
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── rmbg.py            # Main application code
├── docker-compose.cpu.yml    # Docker Compose for CPU version
├── docker-compose.gpu.yml    # Docker Compose for GPU version
├── Dockerfile.cpu     # Dockerfile for CPU version
├── Dockerfile.gpu     # Dockerfile for GPU version
└── models/            # Directory for model cache
```

## Important Notes

1. The `models/` directory is used for caching Hugging Face models. It is mounted in the container to preserve models between restarts.

2. In development mode, you can modify the code in `rmbg.py` - changes will be reflected in the container thanks to volume mounting.

3. The GPU version requires installed NVIDIA Container Toolkit and compatible GPU.

## Troubleshooting

1. If you experience permission issues with the `models/` directory:
```bash
sudo chown -R 1000:1000 models/
```

2. To check GPU in container:
```bash
docker compose -f docker-compose.gpu.yml exec rmbg-gpu nvidia-smi
```

3. Checking logs:
```bash
# CPU Version
docker compose -f docker-compose.cpu.yml logs -f

# GPU Version
docker compose -f docker-compose.gpu.yml logs -f
```