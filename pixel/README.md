# Pixel: AI-First Training Simulation for Mental Health Professionals

## Overview

Pixel is an enhanced Qwen3-30B model with specialized emotional intelligence, clinical accuracy, and therapeutic conversation capabilities designed for training mental health professionals.

## Architecture

```
ai/pixel/
├── models/           # Enhanced model architecture
├── training/         # Multi-objective training system  
├── data/            # Psychology knowledge & conversation processing
├── validation/      # Clinical accuracy & EQ assessment
├── evaluation/      # Comprehensive evaluation framework
├── infrastructure/  # Distributed training setup
├── research/        # Advanced emotional intelligence research
├── config/          # Configuration files
├── utils/           # Supporting utilities
└── scripts/         # Setup and processing scripts
```

## Key Components

### Models
- **PixelBaseModel**: Enhanced Qwen3-30B with emotional intelligence heads
- **Clinical Intervention System**: DSM-5/PDM-2 compliance and intervention recommendations
- **Persona Classification**: Dual-mode therapy/assistant operation

### Training
- **Multi-Objective Loss**: Combines language modeling, EQ, clinical accuracy, and empathy
- **Dynamic Loss Scheduling**: Adaptive weight adjustment during training
- **Psychology Integration**: Real-time clinical knowledge retrieval and validation

### Data Processing
- **Therapeutic Conversation Schema**: Standardized format for psychology knowledge conversion
- **Psychology Knowledge Loader**: DSM-5/PDM-2 extraction and processing
- **Voice Training Pipeline**: Authentic personality modeling from YouTube transcriptions

### Validation & Evaluation
- **Clinical Accuracy Validation**: Expert review and DSM-5/PDM-2 compliance
- **Emotional Intelligence Assessment**: 5-domain EQ measurement
- **Empathy Calibration**: Human baseline comparison and progression tracking

## Getting Started

1. **Environment Setup**:
   ```bash
   cd ai/pixel
   uv add torch transformers datasets
   uv add faiss-cpu sentence-transformers
   ```

2. **Data Preparation**:
   ```python
   from ai.pixel.data import PsychologyKnowledgeLoader
   loader = PsychologyKnowledgeLoader()
   knowledge = loader.load_all_knowledge()
   ```

3. **Model Training**:
   ```python
   from ai.pixel.training import MultiObjectiveLoss
   from ai.pixel.models import PixelBaseModel
   
   model = PixelBaseModel.from_pretrained("Qwen/Qwen3-30B-A3B")
   loss_fn = MultiObjectiveLoss()
   ```

## Task Progress

See `.notes/pixel/tasks-phase-2.md` for detailed implementation progress.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- 8x A100 GPUs (minimum for full training)
- 500GB+ storage for datasets
- CUDA 11.8+

## License

MIT License - See LICENSE file for details.