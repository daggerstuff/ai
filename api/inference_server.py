import os
import argparse
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from llama_cpp import Llama

app = FastAPI(title="Pixelated Empathy EI Engine - Local Inference")

# Global model instance
model = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


@app.on_event("startup")
def load_model():
    global model
    model_path = os.environ.get("MODEL_PATH", "pixelated-v1-wayfarer.Q4_K_M.gguf")

    if not os.path.exists(model_path):
        import sys

        print(f"❌ CRITICAL ERROR: Model file not found at {model_path}")
        print("Please download the GGUF model from Modal before starting the server.")
        sys.exit(1)

    print(f"🚀 Loading model: {model_path}...")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=int(os.cpu_count() or 4),
            n_gpu_layers=0,
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        import sys

        print(f"❌ CRITICAL ERROR: Failed to load model: {e}")
        sys.exit(1)


@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    # Defining as 'def' instead of 'async def' tells FastAPI
    # to run this in a thread pool, preventing it from blocking the event loop.
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format prompt for Wayfarer (matching its training/instruction style)
    # Most GGUF models respond well to the chat-completion API style directly
    # but we can fine-tune the prompt construction here if needed.

    formatted_prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            formatted_prompt += f"<<SYS>>\n{msg.content}\n<</SYS>>\n\n"
        elif msg.role == "user":
            formatted_prompt += f"[INST] {msg.content} [/INST] "
        elif msg.role == "assistant":
            formatted_prompt += f"{msg.content} "

    try:
        response = model(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["[INST]", "</s>", "<|endoftext|>"],
        )

        # Structure as OpenAI-compatible response
        return {
            "id": "chatcmpl-pixelated",
            "object": "chat.completion",
            "created": 123456789,
            "model": "pixelated-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["choices"][0]["text"].strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": response["usage"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default="pixelated-v1-wayfarer.Q4_K_M.gguf")
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)
