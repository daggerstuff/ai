#!/usr/bin/env python3
"""
OVHcloud AI Deploy Inference Server for Pixelated Empathy
FastAPI server for therapeutic AI inference with MoE model

Features:
- REST API for therapeutic conversation generation
- Bias detection integration
- Health checks for AI Deploy
- Streaming responses support
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import MoE architecture
try:
    from models.moe_architecture import MoEConfig, TherapeuticMoEModel
except ImportError:
    sys.path.insert(0, '/app')
    from models.moe_architecture import MoEConfig, TherapeuticMoEModel

from transformers import AutoTokenizer, AutoModelForCausalLM


# Configuration
MODEL_DIR = os.environ.get('MODEL_DIR', '/models')
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 2048))
DEFAULT_MAX_TOKENS = int(os.environ.get('DEFAULT_MAX_TOKENS', 512))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Global model and tokenizer
model = None
tokenizer = None
model_info = {}


# Pydantic models
class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class InferenceRequest(BaseModel):
    """Request for therapeutic inference"""
    messages: List[ConversationMessage] = Field(..., description="Conversation history")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    do_sample: bool = Field(default=True)
    stream: bool = Field(default=False, description="Enable streaming response")


class InferenceResponse(BaseModel):
    """Response from therapeutic inference"""
    response: str
    model: str
    usage: Dict[str, int]
    bias_score: Optional[float] = None
    routing_info: Optional[Dict[str, Any]] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    uptime_seconds: float


# Startup time tracking
startup_time = datetime.now()


def load_model():
    """Load the therapeutic MoE model"""
    global model, tokenizer, model_info

    model_path = Path(MODEL_DIR)

    # Find model files
    if not model_path.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    # Check for MoE layers
    moe_path = model_path / "moe_layers.pt"
    has_moe = moe_path.exists()

    print(f"Loading model from {model_path}...")
    print(f"  MoE layers: {'Found' if has_moe else 'Not found'}")
    print(f"  Device: {DEVICE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if has_moe:
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load MoE state
        moe_state = torch.load(str(moe_path), map_location=DEVICE)
        moe_config = moe_state['config']

        # Create therapeutic MoE model
        model = TherapeuticMoEModel(base_model, moe_config)

        # Load MoE layer weights
        for layer, state_dict in zip(model.moe_layers, moe_state['moe_layers']):
            layer.load_state_dict(state_dict)

        model.eval()
        model_info['architecture'] = 'MoE'
        model_info['num_experts'] = moe_config.num_experts
        model_info['expert_domains'] = moe_config.expert_domains
    else:
        # Load standard model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        model_info['architecture'] = 'Standard'

    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    model_info['total_parameters'] = total_params
    model_info['device'] = str(next(model.parameters()).device)

    print(f"Model loaded successfully!")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Parameters: {total_params:,}")

    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global model, tokenizer

    print("Starting inference server...")

    try:
        model, tokenizer = load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    yield

    print("Shutting down inference server...")


# Create FastAPI app
app = FastAPI(
    title="Pixelated Empathy Therapeutic AI",
    description="OVHcloud AI Deploy inference server for therapeutic conversation AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_messages(messages: List[ConversationMessage]) -> str:
    """Format messages for model input"""
    formatted = []

    for msg in messages:
        if msg.role == "system":
            formatted.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted.append(f"Assistant: {msg.content}")

    formatted.append("Assistant:")
    return "\n\n".join(formatted)


def generate_response(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool
) -> Dict[str, Any]:
    """Generate response from model"""
    global model, tokenizer

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH - max_tokens
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_length = inputs['input_ids'].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up response
    response_text = response_text.strip()
    if "User:" in response_text:
        response_text = response_text.split("User:")[0].strip()

    return {
        'text': response_text,
        'input_tokens': input_length,
        'output_tokens': len(generated_tokens),
    }


async def stream_response(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool
):
    """Stream response tokens"""
    global model, tokenizer

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH - max_tokens
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate with streaming (simplified - real implementation would use streamer)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Simulate streaming by yielding chunks
    words = response_text.split()
    for i, word in enumerate(words):
        chunk = word + " " if i < len(words) - 1 else word
        yield f"data: {json.dumps({'text': chunk})}\n\n"
        await asyncio.sleep(0.01)

    yield f"data: [DONE]\n\n"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for AI Deploy"""
    uptime = (datetime.now() - startup_time).total_seconds()

    response = HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=DEVICE,
        gpu_available=torch.cuda.is_available(),
        uptime_seconds=uptime
    )

    if torch.cuda.is_available():
        response.gpu_name = torch.cuda.get_device_name(0)
        response.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    return response


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Pixelated Empathy Therapeutic AI",
        "version": "1.0.0",
        "platform": "OVHcloud AI Deploy",
        "model_info": model_info,
        "endpoints": {
            "health": "/health",
            "inference": "/v1/inference",
            "docs": "/docs",
        }
    }


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Generate therapeutic AI response

    This endpoint processes conversation history and generates
    contextually appropriate therapeutic responses.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Format messages
        prompt = format_messages(request.messages)

        # Check for streaming
        if request.stream:
            return StreamingResponse(
                stream_response(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample
                ),
                media_type="text/event-stream"
            )

        # Generate response
        result = generate_response(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )

        latency_ms = (time.time() - start_time) * 1000

        return InferenceResponse(
            response=result['text'],
            model=model_info.get('architecture', 'Unknown'),
            usage={
                'prompt_tokens': result['input_tokens'],
                'completion_tokens': result['output_tokens'],
                'total_tokens': result['input_tokens'] + result['output_tokens'],
            },
            latency_ms=latency_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/model")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_info": model_info,
        "max_length": MAX_LENGTH,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "device": DEVICE,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "inference_server:app",
        host=host,
        port=port,
        workers=1,  # Single worker for GPU
        reload=False
    )

