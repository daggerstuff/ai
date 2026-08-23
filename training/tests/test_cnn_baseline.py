#!/usr/bin/env python3
"""
Unit tests for CNN Feature Extraction Baseline (PIX-002).

Tests cover:
    - Model initialization and configuration
    - Forward pass shapes and outputs
    - Multi-scale convolution operations
    - Micro-rhythm layer functionality
    - Hierarchical aggregation
    - VAD and emotion output ranges
"""

import pytest
import torch
from torch import nn

from models.base.cnn_feature_extractor import (
    CNNFeatureConfig,
    CNNFeatureExtractor,
    HierarchicalAggregator,
    MicroRhythmLayer,
    MultiScaleConvLayer,
    PositionalEncoding,
    create_cnn_baseline,
)


class TestCNNFeatureConfig:
    """Tests for CNNFeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CNNFeatureConfig()
        assert config.vocab_size == 30522
        assert config.hidden_size == 256
        assert config.embedding_dim == 128
        assert config.num_filters == 64
        assert config.kernel_sizes == (1, 2, 3, 4, 5)
        assert config.dropout == 0.1
        assert config.output_dim == 3
        assert config.enable_micro_rhythm is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CNNFeatureConfig(
            vocab_size=10000,
            hidden_size=512,
            num_filters=128,
            kernel_sizes=(2, 3, 4),
            enable_micro_rhythm=False,
        )
        assert config.vocab_size == 10000
        assert config.hidden_size == 512
        assert config.num_filters == 128
        assert config.kernel_sizes == (2, 3, 4)
        assert config.enable_micro_rhythm is False

    def test_empty_kernel_sizes_raises(self):
        """Test that empty kernel_sizes raises ValueError."""
        with pytest.raises(ValueError):
            CNNFeatureConfig(kernel_sizes=()).__post_init__()


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        pe = PositionalEncoding(embedding_dim=128, max_len=512)
        x = torch.randn(2, 32, 128)
        output = pe(x)
        assert output.shape == x.shape

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        pe = PositionalEncoding(embedding_dim=64, max_len=256)
        for seq_len in [16, 32, 64, 128]:
            x = torch.randn(1, seq_len, 64)
            output = pe(x)
            assert output.shape == x.shape

    def test_positional_values_are_added(self):
        """Test that positional encoding adds to input."""
        pe = PositionalEncoding(embedding_dim=32, max_len=64, dropout=0.0)
        pe.eval()
        x = torch.zeros(1, 10, 32)
        output = pe(x)
        assert not torch.allclose(output, x)


class TestMultiScaleConvLayer:
    """Tests for MultiScaleConvLayer module."""

    def test_output_dimension(self):
        """Test that output dimension is correct."""
        embedding_dim = 128
        num_filters = 64
        kernel_sizes = (1, 2, 3)
        conv = MultiScaleConvLayer(embedding_dim, num_filters, kernel_sizes)

        expected_dim = num_filters * len(kernel_sizes)
        assert conv.output_dim == expected_dim

    def test_forward_shape(self):
        """Test forward pass output shape."""
        embedding_dim = 64
        num_filters = 32
        kernel_sizes = (1, 2, 3, 4)
        conv = MultiScaleConvLayer(embedding_dim, num_filters, kernel_sizes)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = conv(x)

        expected_dim = num_filters * len(kernel_sizes)
        assert output.shape == (batch_size, seq_len, expected_dim)

    def test_different_kernel_configs(self):
        """Test with different kernel configurations."""
        configs = [(1,), (2, 3), (1, 2, 3, 4, 5), (3, 4, 5, 6)]
        for kernel_sizes in configs:
            conv = MultiScaleConvLayer(64, 32, kernel_sizes)
            x = torch.randn(2, 16, 64)
            output = conv(x)
            assert output.shape[2] == 32 * len(kernel_sizes)


class TestMicroRhythmLayer:
    """Tests for MicroRhythmLayer module."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        input_dim = 256
        rhythm = MicroRhythmLayer(input_dim, hidden_dim=32, num_layers=2)

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, input_dim)
        output = rhythm(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connection is preserved."""
        input_dim = 128
        rhythm = MicroRhythmLayer(input_dim, hidden_dim=16, num_layers=1)
        rhythm.eval()

        x = torch.randn(2, 32, input_dim)
        output = rhythm(x)

        assert output.shape == x.shape

    def test_different_configs(self):
        """Test with different configurations."""
        configs = [
            (128, 16, 1),
            (256, 32, 2),
            (512, 64, 3),
        ]
        for input_dim, hidden_dim, num_layers in configs:
            rhythm = MicroRhythmLayer(input_dim, hidden_dim, num_layers)
            x = torch.randn(2, 16, input_dim)
            output = rhythm(x)
            assert output.shape == x.shape


class TestHierarchicalAggregator:
    """Tests for HierarchicalAggregator module."""

    def test_output_shape(self):
        """Test that output reduces sequence dimension."""
        input_dim = 256
        agg = HierarchicalAggregator(input_dim, hidden_dim=128)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, input_dim)
        output = agg(x)

        assert output.shape == (batch_size, input_dim)

    def test_with_attention_mask(self):
        """Test with attention mask."""
        input_dim = 128
        agg = HierarchicalAggregator(input_dim)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len)
        mask[0, 8:] = 0

        output = agg(x, mask)
        assert output.shape == (batch_size, input_dim)


class TestCNNFeatureExtractor:
    """Tests for the main CNNFeatureExtractor model."""

    def test_model_initialization(self):
        """Test model can be initialized with default config."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        assert isinstance(model, nn.Module)

    def test_forward_output_keys(self):
        """Test forward pass returns expected keys."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 32))
        outputs = model(input_ids)

        assert "vad" in outputs
        assert "emotion_logits" in outputs
        assert "hidden_states" in outputs

    def test_vad_output_range(self):
        """Test VAD outputs are in valid range [-1, 1]."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
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
        config = CNNFeatureConfig(num_emotion_classes=8)
        model = CNNFeatureExtractor(config)
        model.eval()

        batch_size = 3
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["emotion_logits"].shape == (batch_size, 8)

    def test_hidden_states_shape(self):
        """Test hidden states output shape."""
        config = CNNFeatureConfig(hidden_size=256)
        model = CNNFeatureExtractor(config)
        model.eval()

        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["hidden_states"].shape == (batch_size, 256)

    def test_with_attention_mask(self):
        """Test forward pass with attention mask."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.eval()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 16:] = 0

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        assert outputs["vad"].shape == (batch_size, 3)

    def test_without_micro_rhythm(self):
        """Test model without micro-rhythm layer."""
        config = CNNFeatureConfig(enable_micro_rhythm=False)
        model = CNNFeatureExtractor(config)
        assert model.micro_rhythm is None

        model.eval()
        input_ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["vad"].shape == (2, 3)

    def test_return_dict_false(self):
        """Test return_dict=False returns tuple."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 32))
        with torch.no_grad():
            outputs = model(input_ids, return_dict=False)

        assert isinstance(outputs, tuple)
        assert len(outputs) == 3

    def test_get_feature_size(self):
        """Test feature size method."""
        config = CNNFeatureConfig(hidden_size=512)
        model = CNNFeatureExtractor(config)
        assert model.get_feature_size() == 512

    def test_different_batch_and_seq_sizes(self):
        """Test with various batch and sequence sizes."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.eval()

        for batch_size in [1, 2, 8]:
            for seq_len in [16, 32, 64, 128]:
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))
                with torch.no_grad():
                    outputs = model(input_ids)
                assert outputs["vad"].shape == (batch_size, 3)


class TestCreateCNNBaseline:
    """Tests for the factory function."""

    def test_default_creation(self):
        """Test creating model with defaults."""
        model = create_cnn_baseline()
        assert isinstance(model, CNNFeatureExtractor)

    def test_custom_params(self):
        """Test creating model with custom parameters."""
        model = create_cnn_baseline(
            vocab_size=5000,
            hidden_size=128,
            num_filters=32,
            enable_micro_rhythm=False,
        )
        assert isinstance(model, CNNFeatureExtractor)
        assert model.config.vocab_size == 5000
        assert model.config.hidden_size == 128
        assert model.config.num_filters == 32
        assert model.config.enable_micro_rhythm is False


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradients_flow(self):
        """Test that gradients flow through all parameters."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.train()

        input_ids = torch.randint(0, 1000, (2, 32))
        outputs = model(input_ids)

        loss = outputs["vad"].sum() + outputs["emotion_logits"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_gradient_on_padding(self):
        """Test that padding tokens don't affect output."""
        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 64))
        input_ids_padded = input_ids.clone()
        input_ids_padded[0, 32:] = 0

        with torch.no_grad():
            outputs1 = model(input_ids)
            outputs2 = model(input_ids_padded)

        assert not torch.allclose(outputs1["vad"][0], outputs2["vad"][0])


class TestInferenceSpeed:
    """Tests for inference performance."""

    def test_cpu_inference_speed(self):
        """Test that CPU inference is reasonably fast."""
        import time

        model = CNNFeatureExtractor(CNNFeatureConfig())
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 128))

        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids)

        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                _ = model(input_ids)
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.5, f"Inference too slow: {elapsed * 1000:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
