#!/usr/bin/env python3
"""
Optimized Inference Engine for Therapeutic MoE Model
Target: <2 second response time with high quality
"""

import torch
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from collections import OrderedDict
from threading import Lock

from transformers import AutoTokenizer, GenerationConfig
from moe_architecture import TherapeuticMoEModel, MoEConfig


@dataclass
class InferenceConfig:
    """Configuration for optimized inference"""
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Performance optimizations
    use_cache: bool = True
    use_flash_attention: bool = True
    compile_model: bool = True
    batch_size: int = 1
    
    # Context management
    max_context_length: int = 2048
    context_window: int = 10  # Number of previous messages
    
    # Caching
    enable_response_cache: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Expert routing
    use_expert_cache: bool = True
    expert_cache_size: int = 500


@dataclass
class InferenceMetrics:
    """Metrics for inference performance"""
    total_requests: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)
    
    def update(self, latency: float, cache_hit: bool = False):
        """Update metrics with new request"""
        self.total_requests += 1
        self.total_time += latency
        self.latencies.append(latency)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Keep only last 1000 latencies for percentile calculation
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
        
        # Update statistics
        self.avg_latency = self.total_time / self.total_requests
        
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            self.p50_latency = sorted_latencies[int(n * 0.50)]
            self.p95_latency = sorted_latencies[int(n * 0.95)]
            self.p99_latency = sorted_latencies[int(n * 0.99)]


class ResponseCache:
    """LRU cache for responses"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.lock = Lock()
    
    def _make_key(self, prompt: str, context: List[str]) -> str:
        """Create cache key from prompt and context"""
        context_str = "|".join(context[-5:])  # Last 5 messages
        return f"{prompt}::{context_str}"
    
    def get(self, prompt: str, context: List[str]) -> Optional[str]:
        """Get cached response if available and not expired"""
        key = self._make_key(prompt, context)
        
        with self.lock:
            if key in self.cache:
                response, timestamp = self.cache[key]
                
                # Check if expired
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return response
                else:
                    # Expired, remove
                    del self.cache[key]
        
        return None
    
    def set(self, prompt: str, context: List[str], response: str):
        """Cache a response"""
        key = self._make_key(prompt, context)
        
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = (response, time.time())
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size
            }


class OptimizedInferenceEngine:
    """
    Optimized inference engine for therapeutic MoE model
    Target: <2 second response time
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.device = device
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Caching
        self.response_cache = ResponseCache(
            max_size=self.config.cache_size,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_response_cache else None
        
        # Metrics
        self.metrics = InferenceMetrics()
        
        # Load model
        self._load_model()
        
        # Optimize model
        self._optimize_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"📦 Loading model from {self.model_path}...")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load MoE model
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Load MoE layers if they exist
        moe_path = Path(self.model_path) / "moe_layers.pt"
        if moe_path.exists():
            print("   Loading MoE layers...")
            self.model = TherapeuticMoEModel.from_pretrained(
                self.model_path,
                base_model=base_model
            )
        else:
            print("   Using base model (no MoE layers found)")
            self.model = base_model
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
    
    def _optimize_model(self):
        """Apply optimizations to model"""
        print("⚡ Applying optimizations...")
        
        # Enable inference mode
        torch.set_grad_enabled(False)
        
        # Compile model (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("   Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"   Warning: Could not compile model: {e}")
        
        # Enable Flash Attention if available
        if self.config.use_flash_attention:
            try:
                self.model.config.use_flash_attention_2 = True
            except:
                pass
        
        # Create generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.config.use_cache
        )
        
        print("✅ Optimizations applied")
    
    def _prepare_prompt(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Prepare prompt with context"""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history (limited by context window)
        if conversation_history:
            history = conversation_history[-self.config.context_window:]
            messages.extend(history)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Format as conversation
        prompt = self._format_conversation(messages)
        return prompt
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as conversation string"""
        formatted = []
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            formatted.append(f"{role}: {content}")
        
        formatted.append("Assistant:")
        return "\n".join(formatted)
    
    @torch.inference_mode()
    def generate(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response with optimizations
        
        Args:
            user_input: User's input text
            conversation_history: Previous conversation messages
            system_prompt: System prompt for context
            use_cache: Whether to use response cache
            
        Returns:
            Tuple of (response_text, metadata)
        """
        start_time = time.time()
        
        # Check cache
        cache_hit = False
        if use_cache and self.response_cache:
            context = [msg['content'] for msg in (conversation_history or [])]
            cached_response = self.response_cache.get(user_input, context)
            
            if cached_response:
                cache_hit = True
                latency = time.time() - start_time
                self.metrics.update(latency, cache_hit=True)
                
                return cached_response, {
                    'latency': latency,
                    'cache_hit': True,
                    'tokens_generated': len(self.tokenizer.encode(cached_response))
                }
        
        # Prepare prompt
        prompt = self._prepare_prompt(user_input, conversation_history, system_prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        ).to(self.device)
        
        # Generate
        generation_start = time.time()
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        generation_time = time.time() - generation_start
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Cache response
        if use_cache and self.response_cache:
            context = [msg['content'] for msg in (conversation_history or [])]
            self.response_cache.set(user_input, context, response)
        
        # Calculate metrics
        total_latency = time.time() - start_time
        self.metrics.update(total_latency, cache_hit=False)
        
        metadata = {
            'latency': total_latency,
            'generation_time': generation_time,
            'cache_hit': False,
            'tokens_generated': len(outputs[0]) - inputs['input_ids'].shape[1],
            'input_tokens': inputs['input_ids'].shape[1]
        }
        
        return response, metadata
    
    async def generate_async(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Async version of generate"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            user_input,
            conversation_history,
            system_prompt,
            use_cache
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics"""
        cache_stats = self.response_cache.stats() if self.response_cache else {}
        
        return {
            'total_requests': self.metrics.total_requests,
            'avg_latency': self.metrics.avg_latency,
            'p50_latency': self.metrics.p50_latency,
            'p95_latency': self.metrics.p95_latency,
            'p99_latency': self.metrics.p99_latency,
            'cache_hit_rate': self.metrics.cache_hits / max(self.metrics.total_requests, 1),
            'cache_stats': cache_stats,
            'meets_sla': self.metrics.p95_latency < 2.0  # <2s target
        }
    
    def warmup(self, num_requests: int = 10):
        """Warmup the model with dummy requests"""
        print(f"🔥 Warming up model with {num_requests} requests...")
        
        dummy_prompts = [
            "How are you feeling today?",
            "Tell me about your anxiety.",
            "What brings you here?",
            "Can you describe your symptoms?",
            "How has your week been?"
        ]
        
        for i in range(num_requests):
            prompt = dummy_prompts[i % len(dummy_prompts)]
            _, metadata = self.generate(prompt, use_cache=False)
            print(f"   Request {i+1}/{num_requests}: {metadata['latency']:.3f}s")
        
        print("✅ Warmup complete")
        print(f"   Avg latency: {self.metrics.avg_latency:.3f}s")


def create_optimized_engine(
    model_path: str,
    device: str = "cuda",
    enable_cache: bool = True,
    compile_model: bool = True
) -> OptimizedInferenceEngine:
    """
    Create an optimized inference engine
    
    Args:
        model_path: Path to trained model
        device: Device to run on ('cuda' or 'cpu')
        enable_cache: Enable response caching
        compile_model: Compile model with torch.compile
        
    Returns:
        OptimizedInferenceEngine instance
    """
    config = InferenceConfig(
        enable_response_cache=enable_cache,
        compile_model=compile_model,
        use_flash_attention=True,
        use_cache=True
    )
    
    engine = OptimizedInferenceEngine(
        model_path=model_path,
        config=config,
        device=device
    )
    
    # Warmup
    engine.warmup(num_requests=5)
    
    return engine


if __name__ == "__main__":
    # Example usage
    print("🚀 Optimized Inference Engine")
    print("=" * 60)
    
    # Create engine
    engine = create_optimized_engine(
        model_path="./therapeutic_moe_model",
        device="cuda",
        enable_cache=True,
        compile_model=True
    )
    
    # Test inference
    print("\n📝 Testing inference...")
    
    test_inputs = [
        "I've been feeling really anxious lately.",
        "Can you help me with my depression?",
        "I'm having trouble sleeping."
    ]
    
    for user_input in test_inputs:
        response, metadata = engine.generate(user_input)
        
        print(f"\nUser: {user_input}")
        print(f"Assistant: {response}")
        print(f"Latency: {metadata['latency']:.3f}s")
        print(f"Cache hit: {metadata['cache_hit']}")
    
    # Show metrics
    print("\n📊 Performance Metrics:")
    metrics = engine.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
