"""
emotion_encoding.test.py

Unit tests for EmotionEncoder covering feature extraction, vector encoding, and emotion-aware attention.
"""

import numpy as np

from ai.pixel.utils.emotion_encoding import EmotionEncoder


class TestEmotionEncoder:
    def setup_method(self):
        self.encoder = EmotionEncoder()

    def test_extract_text_features(self):
        features = self.encoder._extract_text_features("I feel happy today.")
        assert isinstance(features, np.ndarray)
        assert features.shape == (128,)

    def test_extract_audio_features(self):
        features = self.encoder._extract_audio_features(None)
        assert isinstance(features, np.ndarray)
        assert features.shape == (128,)

    def test_extract_visual_features(self):
        features = self.encoder._extract_visual_features(None)
        assert isinstance(features, np.ndarray)
        assert features.shape == (128,)

    def test_extract_features_multimodal(self):
        input_data = {"text": "I feel sad.", "audio": None, "visual": None}
        features = self.encoder.extract_features(input_data)
        assert "text" in features
        assert "audio" in features
        assert "visual" in features

    def test_encode_vector(self):
        features = {"text": np.ones(128), "audio": np.ones(128), "visual": np.ones(128)}
        vector = self.encoder.encode_vector(features)
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (384,)

    def test_emotion_aware_attention(self):
        input_tensor = np.ones(128)
        emotion_vector = np.ones(128)
        output = self.encoder.emotion_aware_attention(input_tensor, emotion_vector)
        assert np.allclose(output, np.ones(128))
