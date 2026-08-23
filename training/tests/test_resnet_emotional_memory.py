#!/usr/bin/env python3
"""
Unit tests for ResNet Emotional Memory (PIX-003).

Tests cover:
    - Model initialization and configuration
    - Forward pass shapes and outputs
    - ResidualBlock skip connections
    - EmotionalMemoryPool operations
    - HumanContextLayer functionality
    - VAD and emotion output ranges
"""

import pytest
import torch
from torch import nn

from models.base.resnet_emotional_memory import (
    EmotionalMemoryPool,
    HumanContextLayer,
    ResidualBlock,
    ResNetEmotionalMemory,
    ResNetEmotionalMemoryConfig,
    StochasticDepth,
    create_resnet_emotional_memory,
)


class TestResNetEmotionalMemoryConfig:
    """Tests for ResNetEmotionalMemoryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ResNetEmotionalMemoryConfig()
        assert config.vocab_size == 30522
        assert config.hidden_size == 512
        assert config.num_resnet_blocks == 6
        assert config.memory_size == 64
        assert config.dropout == 0.15
        assert config.use_stochastic_depth is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ResNetEmotionalMemoryConfig(
            vocab_size=10000,
            hidden_size=256,
            num_resnet_blocks=4,
            memory_size=32,
            dropout=0.2,
        )
        assert config.vocab_size == 10000
        assert config.hidden_size == 256
        assert config.num_resnet_blocks == 4
        assert config.memory_size == 32
        assert config.dropout == 0.2

    def test_invalid_num_blocks(self):
        """Test that num_resnet_blocks < 1 raises ValueError."""
        with pytest.raises(ValueError):
            ResNetEmotionalMemoryConfig(num_resnet_blocks=0).__post_init__()

    def test_invalid_memory_size(self):
        """Test that memory_size < 1 raises ValueError."""
        with pytest.raises(ValueError):
            ResNetEmotionalMemoryConfig(memory_size=0).__post_init__()


class TestResidualBlock:
    """Tests for ResidualBlock module."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        block = ResidualBlock(hidden_size=256, intermediate_size=512)
        x = torch.randn(2, 32, 256)
        output = block(x)
        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connection is applied."""
        block = ResidualBlock(hidden_size=128, intermediate_size=256)
        block.eval()

        x = torch.randn(2, 16, 128)
        output = block(x)

        assert output.shape == x.shape

    def test_different_configs(self):
        """Test with different configurations."""
        configs = [
            (128, 256),
            (256, 512),
            (512, 1024),
        ]
        for hidden_size, intermediate_size in configs:
            block = ResidualBlock(hidden_size, intermediate_size)
            x = torch.randn(2, 16, hidden_size)
            output = block(x)
            assert output.shape == x.shape


class TestStochasticDepth:
    """Tests for StochasticDepth module."""

    def test_training_mode(self):
        """Test stochastic depth during training."""
        sd = StochasticDepth(drop_prob=0.5)
        sd.train()

        x = torch.randn(2, 16, 128)
        residual = torch.randn(2, 16, 128)
        output = sd(x, residual)

        assert output.shape == x.shape

    def test_eval_mode(self):
        """Test that stochastic depth is identity during eval."""
        sd = StochasticDepth(drop_prob=0.5)
        sd.eval()

        x = torch.randn(2, 16, 128)
        residual = torch.randn(2, 16, 128)
        output = sd(x, residual)

        expected = x + residual
        assert torch.allclose(output, expected)

    def test_zero_drop_prob(self):
        """Test that zero drop prob is identity."""
        sd = StochasticDepth(drop_prob=0.0)
        sd.train()

        x = torch.randn(2, 16, 128)
        residual = torch.randn(2, 16, 128)
        output = sd(x, residual)

        expected = x + residual
        assert torch.allclose(output, expected)


class TestEmotionalMemoryPool:
    """Tests for EmotionalMemoryPool module."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        memory = EmotionalMemoryPool(
            hidden_size=256,
            memory_size=32,
            memory_heads=4,
        )

        x = torch.randn(2, 16, 256)
        output, memory_state = memory(x)

        assert output.shape == x.shape
        assert memory_state.shape == (2, 32, 256)

    def test_different_memory_sizes(self):
        """Test with different memory sizes."""
        configs = [(32, 4), (64, 8), (128, 4)]
        for memory_size, memory_heads in configs:
            memory = EmotionalMemoryPool(
                hidden_size=128,
                memory_size=memory_size,
                memory_heads=memory_heads,
            )
            x = torch.randn(2, 8, 128)
            output, state = memory(x)
            assert output.shape == x.shape
            assert state.shape == (2, memory_size, 128)

    def test_with_attention_mask(self):
        """Test with attention mask."""
        memory = EmotionalMemoryPool(hidden_size=128, memory_size=32)

        x = torch.randn(2, 16, 128)
        mask = torch.ones(2, 16)
        mask[0, 8:] = 0

        output, _state = memory(x, mask)
        assert output.shape == x.shape


class TestHumanContextLayer:
    """Tests for HumanContextLayer module."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        layer = HumanContextLayer(
            hidden_size=256,
            num_attention_heads=8,
            memory_size=32,
            memory_heads=4,
            context_window_size=16,
        )

        x = torch.randn(2, 16, 256)
        output, memory_state = layer(x)

        assert output.shape == x.shape
        assert memory_state.shape == (2, 32, 256)

    def test_with_attention_mask(self):
        """Test with attention mask."""
        layer = HumanContextLayer(
            hidden_size=128,
            num_attention_heads=4,
            memory_size=16,
            memory_heads=2,
            context_window_size=8,
        )

        x = torch.randn(2, 16, 128)
        mask = torch.ones(2, 16)
        mask[0, 8:] = 0

        output, _state = layer(x, mask)
        assert output.shape == x.shape


class TestResNetEmotionalMemory:
    """Tests for the main ResNetEmotionalMemory model."""

    def test_model_initialization(self):
        """Test model can be initialized with default config."""
        model = ResNetEmotionalMemory(ResNetEmotionalMemoryConfig())
        assert isinstance(model, nn.Module)

    def test_forward_output_keys(self):
        """Test forward pass returns expected keys."""
        model = ResNetEmotionalMemory(ResNetEmotionalMemoryConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 32))
        outputs = model(input_ids)

        assert "vad" in outputs
        assert "emotion_logits" in outputs
        assert "hidden_states" in outputs
        assert "memory_state" in outputs

    def test_vad_output_range(self):
        """Test VAD outputs are in valid range [-1, 1]."""
        model = ResNetEmotionalMemory(ResNetEmotionalMemoryConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (4, 64))
        with torch.no_grad():
            outputs = model(input_ids)

        vad = outputs["vad"]
        assert vad.shape == (4, 3)
        assert torch.all(vad >= -1.0)
        assert torch.all(vad <= 1.0)

    def test_emotion_logits_shape(self):
        """Test emotion logits shape."""
        config = ResNetEmotionalMemoryConfig(num_emotion_classes=8)
        model = ResNetEmotionalMemory(config)
        model.eval()

        batch_size = 3
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["emotion_logits"].shape == (batch_size, 8)

    def test_hidden_states_shape(self):
        """Test hidden states output shape."""
        config = ResNetEmotionalMemoryConfig(hidden_size=512)
        model = ResNetEmotionalMemory(config)
        model.eval()

        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["hidden_states"].shape == (batch_size, 512)

    def test_memory_state_shape(self):
        """Test memory state output shape."""
        config = ResNetEmotionalMemoryConfig(hidden_size=256, memory_size=64)
        model = ResNetEmotionalMemory(config)
        model.eval()

        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["memory_state"].shape == (batch_size, 64, 256)

    def test_with_attention_mask(self):
        """Test forward pass with attention mask."""
        model = ResNetEmotionalMemory(ResNetEmotionalMemoryConfig())
        model.eval()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 16:] = 0

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        assert outputs["vad"].shape == (batch_size, 3)

    def test_without_stochastic_depth(self):
        """Test model without stochastic depth."""
        config = ResNetEmotionalMemoryConfig(
            use_stochastic_depth=False,
            num_resnet_blocks=3,
        )
        model = ResNetEmotionalMemory(config)
        assert model.stochastic_depths is None

        model.eval()
        input_ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["vad"].shape == (2, 3)

    def test_return_dict_false(self):
        """Test return_dict=False returns tuple."""
        model = ResNetEmotionalMemory(ResNetEmotionalMemoryConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(input_ids, return_dict=False)

        assert isinstance(outputs, tuple)
        assert len(outputs) == 3

    def test_different_batch_and_seq_sizes(self):
        """Test with various batch and sequence sizes."""
        model = ResNetEmotionalMemory(ResNetEmotionalMemoryConfig())
        model.eval()

        for batch_size in [1, 2, 4]:
            for seq_len in [16, 32, 64, 128]:
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))
                with torch.no_grad():
                    outputs = model(input_ids)
                assert outputs["vad"].shape == (batch_size, 3)


class TestCreateResNetEmotionalMemory:
    """Tests for the factory function."""

    def test_default_creation(self):
        """Test creating model with defaults."""
        model = create_resnet_emotional_memory()
        assert isinstance(model, ResNetEmotionalMemory)

    def test_custom_params(self):
        """Test creating model with custom parameters."""
        model = create_resnet_emotional_memory(
            vocab_size=5000,
            hidden_size=256,
            num_resnet_blocks=4,
            memory_size=32,
            dropout=0.2,
        )
        assert isinstance(model, ResNetEmotionalMemory)
        assert model.config.vocab_size == 5000
        assert model.config.hidden_size == 256
        assert model.config.num_resnet_blocks == 4
        assert model.config.memory_size == 32
        assert model.config.dropout == 0.2


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradients_flow(self):
        """Test that gradients flow through all parameters."""
        config = ResNetEmotionalMemoryConfig(num_resnet_blocks=2)
        model = ResNetEmotionalMemory(config)
        model.train()

        input_ids = torch.randint(0, 1000, (2, 32))
        outputs = model(input_ids)

        loss = outputs["vad"].sum() + outputs["emotion_logits"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestInferenceSpeed:
    """Tests for inference performance."""

    def test_cpu_inference_speed(self):
        """Test that CPU inference is reasonably fast."""
        import time

        config = ResNetEmotionalMemoryConfig(
            hidden_size=256,
            num_resnet_blocks=3,
            memory_size=32,
        )
        model = ResNetEmotionalMemory(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 128))

        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids)

        start = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                _ = model(input_ids)
        elapsed = (time.perf_counter() - start) / 20

        assert elapsed < 0.5, f"Inference too slow: {elapsed * 1000:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
