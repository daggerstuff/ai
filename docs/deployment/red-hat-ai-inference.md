# Red Hat AI Inference Setup Guide

Operator-ready setup for Red Hat AI Inference 3.4 across two paths:

- standalone Linux on a GPU host
- Kubernetes with provider-specific guidance for Civo and OVHcloud

This guide keeps vendor support boundaries explicit. Red Hat documents managed
Kubernetes installs for AKS and CoreWeave through the `rhai-on-xks` chart. Civo
and OVHcloud are not listed as supported Red Hat xKS targets, so this guide uses
upstream `llm-d` guidance for those clusters instead of presenting them as a
supported Red Hat path.

## Scope and support boundary

Use this guide when you need one of these outcomes:

1. run Red Hat AI Inference on a single Linux GPU host quickly
2. prepare a Kubernetes cluster for distributed inference patterns based on
   upstream `llm-d`
3. benchmark or validate the deployment after install

Do not use this guide to claim Red Hat support for Civo or OVHcloud managed
Kubernetes. For Red Hat-supported managed Kubernetes installation, use the Red
Hat AI Inference 3.4 xKS documentation for AKS or CoreWeave.

## Prerequisites

### Common prerequisites

- Red Hat AI Inference 3.4 access through `registry.redhat.io`
- Hugging Face token with `read` scope for gated/public model downloads, or
  `write` if you also need private-repository access
- `curl` and `jq`
- outbound network access to `registry.redhat.io` and Hugging Face

### Linux host prerequisites

- RHEL 9 or another Linux host with Podman or Docker
- NVIDIA GPU drivers installed and working
- NVIDIA container runtime installed and working
- one or more GPUs sized to the model you plan to serve

### Kubernetes prerequisites

- Kubernetes cluster with GPU worker nodes
- `kubectl`
- Helm 3.17+
- storage class for PVC-backed caching or model artifacts
- ingress or load balancer strategy for exposing the API

## Recommended starting models

Start small so validation is fast and GPU fit is obvious.

| Model | Good for | Notes |
|---|---|---|
| `RedHatAI/Llama-3.2-1B-Instruct-FP8` | smoke tests, dev bring-up | smallest recommended starting point from Red Hat examples |
| `RedHatAI/Granite-3.1-8B-Instruct` | enterprise-style instruction tuning | verify GPU memory fit before scaling concurrency |
| `mistralai/*` via supported images/workflows | general experimentation | validate support and tokenizer compatibility first |

## Path 1: Standalone Linux install

### What this path does

The scripted Linux path pulls the Red Hat AI Inference CUDA image, prepares a
cache directory, logs in to `registry.redhat.io`, launches the inference server,
waits for readiness, and runs a smoke test against the OpenAI-compatible API.

### What the script does not do

The script does not install GPU drivers or the container toolkit. Those are host
prerequisites and should be verified before you start.

### Script location

`scripts/devops/install-red-hat-ai-inference-linux.sh`

### Example usage

```bash
chmod +x scripts/devops/install-red-hat-ai-inference-linux.sh

export HF_TOKEN='<your-hugging-face-token>'

scripts/devops/install-red-hat-ai-inference-linux.sh \
  --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
  --tensor-parallel-size 1
```

### Non-interactive usage

```bash
export HF_TOKEN='<your-hugging-face-token>'

scripts/devops/install-red-hat-ai-inference-linux.sh \
  --engine podman \
  --registry-user '<registry-user>' \
  --registry-password '<registry-password-or-token>' \
  --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
  --port 8000 \
  --tensor-parallel-size 1
```

### Expected success criteria

The script is successful when all of the following are true:

- container starts without crash looping
- `GET /v1/models` answers on the local port
- the included completion smoke test returns JSON

### Manual validation

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq

curl -fsS -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello in five words.","max_tokens":16}' | jq
```

### Rollback

```bash
podman rm -f rhaii-vllm
# or
docker rm -f rhaii-vllm
```

### Linux troubleshooting

#### `nvidia-smi` is missing or shows no GPUs

Stop. Fix host driver installation first. The container path will not work until
the host sees the GPUs.

#### `401 Unauthorized` or image pull failures from `registry.redhat.io`

Re-run login and confirm your account has access to Red Hat AI Inference images.

#### Hugging Face download failures

Verify `HF_TOKEN` and ensure the token can access the selected model.

#### Podman with SELinux enabled

The script enables `container_use_devices` when SELinux is active because Red
Hat documents that requirement for device access.

## Path 2: Kubernetes setup

## Red Hat-supported Kubernetes targets

Red Hat AI Inference 3.4 documents these managed Kubernetes targets for the
`rhai-on-xks` chart:

- Azure AKS
- CoreWeave Kubernetes Service

If you need a fully supported Red Hat-managed Kubernetes install, follow the Red
Hat xKS documentation for those platforms.

## Civo and OVHcloud strategy

For Civo and OVHcloud, use this model:

1. create a Kubernetes cluster with GPU-capable node pools
2. bootstrap common prerequisites with the companion script in
   `scripts/devops/setup-rhaii-k8s-prereqs.sh`
3. install NVIDIA GPU operator using the provider-specific settings below when
   you need to validate or override the script defaults manually
4. install common `llm-d` prerequisites such as cert-manager, Gateway API
   support, and the required inference components from upstream docs
5. deploy your inference services through upstream `llm-d` patterns

That gives you a practical setup guide without overstating Red Hat support.

### Companion bootstrap script

Script location:

`scripts/devops/setup-rhaii-k8s-prereqs.sh`

What it does:

- adds Helm repos for NVIDIA and cert-manager
- creates namespaces for GPU Operator, cert-manager, and inference workloads
- installs cert-manager
- installs the NVIDIA GPU Operator with Civo or OVH-specific settings
- optionally applies a Gateway API manifest URL that you provide

What it does not do:

- deploy `llm-d`
- install KServe or application-specific inference CRDs
- expose your inference endpoint
- claim Red Hat support for Civo or OVH

### Example bootstrap commands

#### Civo

```bash
chmod +x scripts/devops/setup-rhaii-k8s-prereqs.sh

scripts/devops/setup-rhaii-k8s-prereqs.sh \
  --provider civo \
  --gateway-api-manifest-url https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml
```

#### OVHcloud

```bash
chmod +x scripts/devops/setup-rhaii-k8s-prereqs.sh

scripts/devops/setup-rhaii-k8s-prereqs.sh \
  --provider ovh \
  --ovh-driver-version 535.183.01 \
  --gateway-api-manifest-url https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml
```

### Bootstrap success criteria

Treat the bootstrap step as successful only when all of these are true:

- `gpu-operator` pods are healthy
- `cert-manager` pods are healthy
- GPU nodes show expected NVIDIA labels or allocatable GPU resources
- Gateway API CRDs install cleanly if you passed a manifest URL

## Cluster design checklist

Before applying anything, decide these items:

- GPU SKU and GPU count per node
- storage class for model cache or result persistence
- ingress or load balancer exposure pattern
- namespace layout for operators, gateways, and workloads
- autoscaling expectations
- observability stack for GPU, pod, and request metrics

## Civo Kubernetes setup

### Civo-specific notes

- Civo documents GPU-capable node pools with NVIDIA A100, H100, L40S, B200, and
  GH200 availability depending on region and SKU
- Civo GPU images already ship with the NVIDIA container toolkit, so install the
  GPU operator with `toolkit.enabled=false`
- Civo clusters default to K3s, which matters when validating operator
  compatibility and cluster features
- single-GPU H100 nodes may require the documented NVLink disable workaround

### Civo bring-up sequence

1. create the cluster and attach a GPU node pool in the Civo dashboard, CLI, or
   Terraform
2. confirm GPU nodes are present and schedulable
3. install the NVIDIA GPU operator with toolkit disabled
4. verify GPU feature discovery labels exist on GPU nodes
5. continue with upstream `llm-d` prerequisites and workloads

### Civo validation commands

```bash
kubectl get nodes -L nvidia.com/gpu.present
kubectl get pods -A
```

### Civo GPU operator install

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator \
  --set toolkit.enabled=false
```

### Civo platform caveats

- confirm quota and SKU availability before promising capacity
- prefer larger nodes because small Civo nodes have heavier reserved-resource
  overhead
- if you use H100 single-GPU nodes, follow Civo's documented NVLink workaround

## OVHcloud Managed Kubernetes setup

### OVH-specific notes

- GPU quotas are not available by default; request quota before planning the
  deployment window
- OVHcloud documents GPU flavors such as `T1-45` and `L40S-90` depending on
  region and inventory
- the default GPU operator driver line may not match CUDA versions expected by
  your inference images, so OVH guidance pins the driver version explicitly
- OVH supports autoscaling for GPU node pools; scale-to-zero may require
  additional tooling or custom node-pool automation

### OVH bring-up sequence

1. request GPU quota approval
2. create the managed Kubernetes cluster
3. add a GPU node pool from the OVHcloud GPU flavor catalog
4. install the NVIDIA GPU operator with an explicit driver version
5. verify node labels and GPU operator health
6. continue with upstream `llm-d` prerequisites and workloads

### OVH validation commands

```bash
kubectl get nodes -o wide
kubectl get pods -n gpu-operator
```

### OVH GPU operator install

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator \
  --set driver.enabled=true \
  --set driver.version="535.183.01" \
  --set toolkit.enabled=true \
  --set operator.defaultRuntime=containerd \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set gfd.enabled=true \
  --set migManager.enabled=false \
  --timeout 20m
```

### OVH platform caveats

- quota delays can block the project before the first GPU node exists
- first GPU node provisioning may take longer than a normal worker pool
- if `dcgm-exporter` image pulls fail, compare the deployed image tag to OVH's
  current guidance and override as needed

## Upstream `llm-d` prerequisites for Civo and OVH

Red Hat's OpenShift and xKS guides cover supported platforms. For Civo and OVH,
continue with the upstream `llm-d` quick start and provider docs after GPU setup.
At minimum, plan for these cluster-level dependencies:

- Gateway API-compatible ingress or service mesh
- cert-manager
- KServe and related inference CRDs where required by the chosen pattern
- LeaderWorkerSet only if you need wide expert parallelism

Follow the upstream `llm-d` project documentation for the exact current install
sequence because those components change faster than the Red Hat product docs.

## Benchmarking and validation

### GuideLLM quick benchmark

```bash
pip install guidellm[recommended]

guidellm run \
  --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

### vLLM regression benchmark

```bash
pip install vllm pandas datasets
git clone https://github.com/vllm-project/vllm.git

python vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
  --num-prompts 100 \
  --dataset-name random \
  --random-input 1024 \
  --random-output 512 \
  --port 8000
```

### Validation checklist

- [ ] image pull works from `registry.redhat.io`
- [ ] model download works with the supplied Hugging Face token
- [ ] `/v1/models` answers
- [ ] completion smoke test answers
- [ ] GPU operator pods are healthy on Kubernetes
- [ ] GPU node labels are present
- [ ] a short benchmark run completes without timeouts

## Model optimization

Red Hat AI Model Optimization Toolkit wraps LLM Compressor workflows such as AWQ,
GPTQ, FP8, SmoothQuant, and sparsity-oriented transforms.

```bash
podman pull registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.4.0

podman run --rm \
  -v "$(pwd):/opt/app-root/model-opt" \
  --device nvidia.com/gpu=all --ipc=host \
  -e HF_TOKEN='<your-hugging-face-token>' \
  registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.4.0 \
  python /opt/app-root/model-opt/llm-compressor/examples/quantization_w8a8_int8/llama3_example.py
```

## Source references

- Red Hat AI Inference 3.4 getting started docs
- Red Hat AI Inference distributed inference docs for OpenShift and xKS
- upstream `llm-d` repository and quick start
- Civo GPU Kubernetes documentation
- OVHcloud Managed Kubernetes GPU deployment documentation
- GuideLLM install and usage docs
- LLM Compressor repository and Red Hat model optimization docs
