# Triton Model Repository Configuration

This directory contains the Pixel model packaged for NVIDIA Triton Inference Server deployment.

## Structure

```
model_repository/
├── pixel/
│   ├── 1/                          # Model version 1
│   │   ├── model.pt               # PyTorch model file
│   │   └── model.onnx             # ONNX format (optional)
│   ├── 2/                          # Model version 2 (A/B testing)
│   │   ├── model.pt
│   │   └── model.onnx
│   └── config.pbtxt               # Model configuration
├── ensemble/
│   ├── 1/
│   │   └── model.pt
│   └── config.pbtxt
└── README.md
```

## Model Configuration

Each model requires a `config.pbtxt` file that specifies:
- Model platform (pytorch_libtorch, onnx_runtime, etc.)
- Input/output specifications
- Batching configuration
- Instance group (GPU/CPU placement)
- Model parameters

## Deployment

### Single Model
```bash
tritonserver --model-repository=./model_repository
```

### With Specific Version
```bash
TRITON_ARGS="--model-control-mode=explicit" tritonserver --model-repository=./model_repository
```

### With Environment Variables
```bash
export TRITON_METRICS_PORT=8002
export TRITON_HTTP_PORT=8000
export TRITON_GRPC_PORT=8001
tritonserver --model-repository=./model_repository
```

## A/B Testing Setup

Deploy multiple model versions and route traffic:
- Version 1: Current production model
- Version 2: Candidate model for evaluation

Use client-side routing or Triton ensemble to split traffic.

## Monitoring

Triton exposes metrics at:
- HTTP: `http://localhost:8000/metrics`
- Prometheus scrape endpoint: `http://localhost:8002/metrics`

## Documentation

- [Triton Model Deployment](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)
- [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)
- [Batching Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/advanced_batching.md)
