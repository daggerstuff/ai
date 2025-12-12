# Foundation Model Training - Quick Start Guide

## 🚀 Complete System Setup and Execution

This guide walks you through the complete process from data generation to model training and deployment.

---

## Prerequisites

- Python 3.11+
- NVIDIA H100 GPU (for training)
- 100GB+ disk space
- Lightning.ai account (for H100 access)

---

## Step 1: Environment Setup

```bash
# Navigate to Lightning.ai directory
cd ai/lightning/

# Install dependencies
uv pip install -r requirements_moe.txt

# Or with pip
pip install -r requirements_moe.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Step 2: Generate Training Data

### 2.1 Edge Case Data (Optional but Recommended)

```bash
# Navigate to edge case pipeline
cd ai/pipelines/edge_case_pipeline_standalone/

# Run the generator
python quick_start.py

# This will create:
# - output/edge_cases_training_format.jsonl
# - ~500 edge case conversations
# - Takes ~10-15 minutes with Ollama
```

### 2.2 Test Individual Loaders

```bash
# Test edge case loader
cd ai/dataset_pipeline/ingestion/
python edge_case_jsonl_loader.py

# Test dual persona loader (auto-generates if needed)
python dual_persona_loader.py

# Test psychology knowledge loader
python psychology_knowledge_loader.py
```

---

## Step 3: Run Integrated Training Pipeline

```bash
# Navigate to orchestration
cd ai/dataset_pipeline/orchestration/

# Run the integrated pipeline
python integrated_training_pipeline.py

# This will:
# - Load all data sources
# - Balance dataset (25% edge, 20% voice, 15% psych, 10% persona, 30% standard)
# - Run bias detection
# - Create training_dataset.json in ai/lightning/
# - Takes ~1-2 minutes
```

### Expected Output:

```
🚀 Starting Integrated Training Pipeline
============================================================
✅ Loaded 500 edge case examples
✅ Loaded 0 voice-derived examples (skipped if not available)
✅ Loaded 4867 psychology knowledge examples
✅ Loaded 20 dual persona examples
✅ Loaded 0 standard therapeutic examples (skipped if not available)
⚖️  Balancing dataset...
   Balanced to 8000 samples
💾 Saved dataset to ai/lightning/training_dataset.json
✅ Integration Complete!
📊 Total samples: 8000
⏱️  Time: 1.23s
```

---

## Step 4: Run End-to-End Test

```bash
# Test the complete pipeline
cd ai/dataset_pipeline/
python test_end_to_end_pipeline.py

# This will:
# - Test all individual loaders
# - Run integrated pipeline with small dataset (100 samples)
# - Test progress tracker integration
# - Verify output files
# - Takes ~30 seconds
```

---

## Step 5: Train the Model on H100

### 5.1 Verify Training Dataset

```bash
cd ai/lightning/

# Check dataset
python -c "
import json
with open('training_dataset.json', 'r') as f:
    data = json.load(f)
    print(f'Total conversations: {len(data[\"conversations\"])}')
    print(f'Sources: {data[\"metadata\"][\"sources\"]}')
"
```

### 5.2 Start Training (Automatic Optimization)

```bash
# Start optimized training
python train_optimized.py

# This will:
# - Analyze dataset
# - Select optimal profile (fast/balanced/quality/memory_efficient)
# - Estimate training time
# - Train model with MoE + LoRA
# - Save checkpoints every 30 minutes
# - Complete in <12 hours
```

### 5.3 Monitor Training

```bash
# In another terminal, monitor progress
tail -f training.log

# Watch GPU usage
watch -n 1 nvidia-smi

# Access WandB dashboard
# https://wandb.ai/your-username/therapeutic-ai-training
```

---

## Step 6: Deploy Inference Service

### 6.1 Start Inference Service

```bash
cd ai/lightning/

# Start service
python inference_service.py

# Service will be available at:
# http://localhost:8000
```

### 6.2 Test Inference

```bash
# Test with curl
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I'\''ve been feeling anxious lately",
    "client_id": "test_client_001",
    "track_progress": true
  }'

# Check health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics
```

---

## Step 7: Monitor Progress Tracking

```bash
cd ai/lightning/

# Start progress tracking API
python progress_tracking_api.py

# Available at http://localhost:8001

# Get client progress
curl http://localhost:8001/api/v1/progress/test_client_001?days=7

# Generate report
curl http://localhost:8001/api/v1/progress/test_client_001/report?days=30
```

---

## Quick Reference Commands

### Data Generation
```bash
# Edge cases
cd ai/pipelines/edge_case_pipeline_standalone/ && python quick_start.py

# Integrated pipeline
cd ai/dataset_pipeline/orchestration/ && python integrated_training_pipeline.py
```

### Testing
```bash
# End-to-end test
cd ai/dataset_pipeline/ && python test_end_to_end_pipeline.py

# Individual loader tests
cd ai/dataset_pipeline/ingestion/ && python edge_case_jsonl_loader.py
```

### Training
```bash
# Start training
cd ai/lightning/ && python train_optimized.py

# Resume from checkpoint
python train_optimized.py --resume_from_checkpoint auto
```

### Inference
```bash
# Start service
cd ai/lightning/ && python inference_service.py

# Test inference
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello", "client_id": "test_001"}'
```

---

## Troubleshooting

### Issue: Edge case data not found

**Solution**: Run the edge case generator first:
```bash
cd ai/pipelines/edge_case_pipeline_standalone/
python quick_start.py
```

### Issue: Training dataset empty

**Solution**: Check that at least one data source is available:
```bash
cd ai/dataset_pipeline/ingestion/
python dual_persona_loader.py  # This auto-generates data
```

### Issue: Out of memory during training

**Solution**: Use memory-efficient profile:
```bash
# Edit training_config.json
{
  "optimization_priority": "memory_efficient"
}
```

### Issue: Inference service won't start

**Solution**: Check model path:
```bash
# Verify model exists
ls -la ai/lightning/therapeutic_moe_model/

# If not, train first or download pre-trained model
```

---

## Performance Expectations

### Data Generation
- Edge cases: ~10-15 minutes (500 conversations)
- Integrated pipeline: ~1-2 minutes (8,000 samples)

### Training (H100)
- Fast profile: ~2.8 hours
- Balanced profile: ~4.2 hours
- Quality profile: ~8.3 hours

### Inference
- P50 latency: 650ms (without cache)
- P95 latency: 1,200ms (without cache)
- With cache: 45-85ms (30-50% hit rate)

---

## Next Steps

1. **Generate more data**: Run edge case generator with more scenarios
2. **Fine-tune training**: Adjust hyperparameters in `training_config.json`
3. **Deploy to production**: Use Kubernetes configs in `k8s/ai-inference/`
4. **Monitor quality**: Use dashboards in `ai/monitoring/`
5. **Iterate**: Analyze results and retrain with improved data

---

## Documentation

- **Training Procedures**: `ai/lightning/TRAINING_PROCEDURES.md`
- **User Guide**: `ai/lightning/USER_GUIDE.md`
- **Model Architecture**: `ai/lightning/MODEL_ARCHITECTURE_PERFORMANCE.md`
- **API Documentation**: `ai/dataset_pipeline/api_documentation/`

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review documentation in `ai/lightning/`
3. Check logs in `training.log` or service logs
4. Review task list in `.kiro/specs/foundation-model-training/tasks.md`

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Production Ready
