import argparse
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel

app = FastAPI(title="Pixelated Empathy EI Engine - Local Inference")

# Global model instance
model = None
INTERNAL_ERROR_MSG = "Internal server error"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


@app.on_event("startup")
def load_model():
    global model
    model_path = os.environ.get("MODEL_PATH", "pixelated-v1-wayfarer.Q4_K_M.gguf")

    if not os.path.exists(model_path):
        sys.exit(1)

    try:
        model = Llama(model_path=model_path, n_ctx=4096, n_threads=int(os.cpu_count() or 4), n_gpu_layers=0)
    except Exception:
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

    # Use list and join for O(n) string concatenation performance
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"<<SYS>>\n{msg.content}\n<</SYS>>\n\n")
        elif msg.role == "user":
            prompt_parts.append(f"[INST] {msg.content} [/INST] ")
        elif msg.role == "assistant":
            prompt_parts.append(f"{msg.content} ")
    formatted_prompt = "".join(prompt_parts)

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
                    "message": {"role": "assistant", "content": response["choices"][0]["text"].strip()},
                    "finish_reason": "stop",
                }
            ],
            "usage": response["usage"],
        }
    except Exception as e:
        logging.exception("Chat completion failed:")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_MSG) from e


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
