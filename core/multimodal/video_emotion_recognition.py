"""
Video Emotion Recognition for Pixel Multimodal Integration.

Detects emotional states from facial expressions in video using VAD dimensions.
Supports edge processing for privacy-preserving emotion detection.
Integrates with Pixel's EQ scoring system for multimodal emotion tracking.

Features:
  - Facial expression detection from video frames
  - Valence detection (negative to positive)
  - Arousal detection (calm to excited)
  - Dominance detection (controlled to dominant)
  - Frame-by-frame emotion tracking over time
  - Privacy-first edge processing (no cloud calls)
  - Integration with audio/text emotion for fusion

Example:
    >>> recognizer = VideoEmotionRecognizer()
    >>> emotions = await recognizer.detect_emotions("session_001", video_path)
    >>> print(f"Valence: {emotions.overall_emotion.valence:.2f}")
    >>> print(f"Arousal: {emotions.overall_emotion.arousal:.2f}")
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Facial expression to VAD mapping (based on psychological research)
FACIAL_EXPRESSION_TO_VAD = {
    "neutral": (0.5, 0.5, 0.5),
    "happy": (0.9, 0.7, 0.6),
    "sad": (0.2, 0.3, 0.3),
    "angry": (0.2, 0.9, 0.8),
    "fearful": (0.2, 0.8, 0.2),
    "surprised": (0.7, 0.8, 0.5),
    "disgusted": (0.2, 0.5, 0.6),
    "contempt": (0.3, 0.4, 0.7),
}

# Action Unit to emotion mapping (simplified FACS)
ACTION_UNIT_EMOTIONS = {
    # Upper face
    "AU01": "sad",  # Inner brow raiser
    "AU02": "surprised",  # Outer brow raiser
    "AU04": "angry",  # Brow lowerer
    "AU05": "fearful",  # Upper lid raiser
    "AU06": "happy",  # Cheek raiser
    "AU07": "angry",  # Lid tightener
    # Lower face
    "AU09": "disgusted",  # Nose wrinkler
    "AU10": "disgusted",  # Upper lip raiser
    "AU12": "happy",  # Lip corner puller
    "AU15": "sad",  # Lip corner depressor
    "AU17": "sad",  # Chin raiser
    "AU20": "fearful",  # Lip stretcher
    "AU23": "angry",  # Lip tightener
    "AU24": "angry",  # Lip pressor
    "AU25": "neutral",  # Lips part
    "AU26": "surprised",  # Jaw drop
    "AU27": "surprised",  # Mouth stretch
    "AU43": "neutral",  # Eyes closed
    "AU45": "neutral",  # Blink
}


@dataclass
class FacialLandmark:
    """Single facial landmark point."""

    x: float
    y: float
    confidence: float = 1.0


@dataclass
class FaceDetection:
    """Detected face in a frame."""

    bbox: Tuple[float, float, float, float]  # x, y, width, height
    landmarks: List[FacialLandmark]
    confidence: float
    face_id: int = 0


@dataclass
class FrameEmotion:
    """Emotion detected in a single frame."""

    frame_number: int
    timestamp_ms: float
    emotion: str
    valence: float
    arousal: float
    dominance: float
    confidence: float
    emotion_probabilities: Dict[str, float]
    face_detection: Optional[FaceDetection] = None


@dataclass
class VideoEmotionResult:
    """Complete video emotion detection result."""

    session_id: str
    video_path: str
    overall_emotion: "EmotionalState"
    frame_emotions: List[FrameEmotion]
    trajectory: List["EmotionalState"]
    face_detections_count: int
    frames_processed: int
    processing_time_ms: float
    video_duration_s: float
    model_name: str
    edge_processed: bool
    error: Optional[str] = None


@dataclass
class EmotionalState:
    """Emotional state representation using VAD model."""

    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # -1.0 (calm) to 1.0 (excited)
    dominance: float  # -1.0 (submissive) to 1.0 (dominant)
    confidence: float  # 0.0-1.0 confidence
    primary_emotion: str  # closest emotion label
    emotion_probabilities: Dict[str, float] = field(default_factory=dict)


class VideoEmotionRecognizer:
    """Detect emotions from facial expressions in video.

    Supports edge processing for privacy-preserving emotion detection.
    Uses lightweight models for real-time processing.
    """

    def __init__(
        self,
        model_type: str = "edge",
        device: str = "cpu",
        edge_processing: bool = True,
        frame_sample_rate: int = 5,
    ):
        """Initialize video emotion recognizer.

        Args:
            model_type: Model type ('edge' for lightweight, 'accurate' for heavy)
            device: Device for inference ('cpu' or 'cuda')
            edge_processing: If True, ensures local-only processing
            frame_sample_rate: Process every Nth frame (1 = all frames)
        """
        self.model_type = model_type
        self.device = device
        self.edge_processing = edge_processing
        self.frame_sample_rate = frame_sample_rate
        self.model_name = f"pixel-video-emotion-{model_type}"

        self._face_detector = None
        self._emotion_classifier = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize models on first use."""
        if self._initialized:
            return

        logger.info(
            f"Initializing video emotion recognizer (edge={self.edge_processing})"
        )

        if self.edge_processing:
            self._init_edge_models()
        else:
            self._init_full_models()

        self._initialized = True

    def _init_edge_models(self) -> None:
        """Initialize lightweight edge models for local processing."""
        try:
            import cv2

            # Use OpenCV's DNN face detector for edge processing
            self._face_detector = "opencv_dnn"
            self._emotion_classifier = "rule_based"
            logger.info(
                "Initialized edge models: OpenCV DNN face detector + rule-based classifier"
            )
        except ImportError:
            logger.warning("OpenCV not available, using fallback detection")
            self._face_detector = "fallback"
            self._emotion_classifier = "fallback"

    def _init_full_models(self) -> None:
        """Initialize full models for accurate processing."""
        try:
            import torch
            from transformers import AutoModelForImageClassification, AutoImageProcessor

            model_id = "trpakov/vit-facial-expression-recognition"
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._emotion_classifier = AutoModelForImageClassification.from_pretrained(
                model_id
            )
            self._emotion_classifier.to(self.device)
            self._emotion_classifier.eval()
            self._face_detector = "transformers"
            logger.info(f"Initialized full models: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to load full models, falling back to edge: {e}")
            self._init_edge_models()

    async def detect_emotions(
        self,
        session_id: str,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> VideoEmotionResult:
        """Detect emotions from video file.

        Args:
            session_id: Session identifier
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)

        Returns:
            VideoEmotionResult with VAD scores and frame-by-frame analysis
        """
        start_time = time.time()
        self._ensure_initialized()

        try:
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Open video
            import cv2

            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frame_emotions = []
            frame_count = 0
            processed_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Sample frames based on rate
                if frame_count % self.frame_sample_rate != 0:
                    continue

                if max_frames and processed_count >= max_frames:
                    break

                # Detect emotion in frame
                timestamp_ms = (frame_count / fps) * 1000 if fps > 0 else 0
                frame_emotion = await self._detect_frame_emotion(
                    frame, frame_count, timestamp_ms
                )
                if frame_emotion:
                    frame_emotions.append(frame_emotion)
                    processed_count += 1

            cap.release()

            # Aggregate emotions
            overall_emotion = self._aggregate_frame_emotions(frame_emotions)
            trajectory = [
                EmotionalState(
                    valence=e.valence,
                    arousal=e.arousal,
                    dominance=e.dominance,
                    confidence=e.confidence,
                    primary_emotion=e.emotion,
                    emotion_probabilities=e.emotion_probabilities,
                )
                for e in frame_emotions
            ]

            processing_time = (time.time() - start_time) * 1000

            return VideoEmotionResult(
                session_id=session_id,
                video_path=str(video_file),
                overall_emotion=overall_emotion,
                frame_emotions=frame_emotions,
                trajectory=trajectory,
                face_detections_count=len(
                    [e for e in frame_emotions if e.face_detection]
                ),
                frames_processed=processed_count,
                processing_time_ms=processing_time,
                video_duration_s=duration,
                model_name=self.model_name,
                edge_processed=self.edge_processing,
            )

        except Exception as e:
            logger.error(f"Video emotion detection failed: {str(e)}")
            processing_time = (time.time() - start_time) * 1000
            return VideoEmotionResult(
                session_id=session_id,
                video_path=video_path,
                overall_emotion=EmotionalState(
                    valence=0.0,
                    arousal=0.0,
                    dominance=0.0,
                    confidence=0.0,
                    primary_emotion="neutral",
                    emotion_probabilities={},
                ),
                frame_emotions=[],
                trajectory=[],
                face_detections_count=0,
                frames_processed=0,
                processing_time_ms=processing_time,
                video_duration_s=0.0,
                model_name=self.model_name,
                edge_processed=self.edge_processing,
                error=str(e),
            )

    async def _detect_frame_emotion(
        self,
        frame: "np.ndarray",
        frame_number: int,
        timestamp_ms: float,
    ) -> Optional[FrameEmotion]:
        """Detect emotion in a single frame.

        Args:
            frame: Video frame as numpy array (BGR)
            frame_number: Frame index
            timestamp_ms: Frame timestamp in milliseconds

        Returns:
            FrameEmotion or None if no face detected
        """
        try:
            import cv2

            # Detect faces
            faces = self._detect_faces(frame)
            if not faces:
                return None

            # Use the largest/most confident face
            primary_face = max(
                faces, key=lambda f: f.confidence * f.bbox[2] * f.bbox[3]
            )

            # Extract face region
            x, y, w, h = [int(v) for v in primary_face.bbox]
            face_region = frame[y : y + h, x : x + w]

            if face_region.size == 0:
                return None

            # Classify emotion
            emotion, probabilities, confidence = await self._classify_emotion(
                face_region
            )

            # Map to VAD
            vad = FACIAL_EXPRESSION_TO_VAD.get(emotion, (0.5, 0.5, 0.5))
            valence, arousal, dominance = vad

            return FrameEmotion(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                emotion=emotion,
                valence=valence * 2 - 1,  # Convert to -1 to 1
                arousal=arousal * 2 - 1,
                dominance=dominance * 2 - 1,
                confidence=confidence,
                emotion_probabilities=probabilities,
                face_detection=primary_face,
            )

        except Exception as e:
            logger.debug(f"Frame {frame_number} emotion detection failed: {e}")
            return None

    def _detect_faces(self, frame: "np.ndarray") -> List[FaceDetection]:
        """Detect faces in frame.

        Args:
            frame: Video frame as numpy array

        Returns:
            List of detected faces
        """
        import cv2

        faces = []

        if self._face_detector == "opencv_dnn":
            # Use OpenCV DNN face detector
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            # Note: Full DNN implementation would load model weights
            # For edge processing, we use Haar cascades as fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            detected = cascade.detectMultiScale(gray, 1.1, 4)

            for i, (x, y, fw, fh) in enumerate(detected):
                faces.append(
                    FaceDetection(
                        bbox=(float(x), float(y), float(fw), float(fh)),
                        landmarks=[],
                        confidence=0.8,
                        face_id=i,
                    )
                )

        elif self._face_detector == "fallback":
            # Simple fallback using center of frame
            h, w = frame.shape[:2]
            faces.append(
                FaceDetection(
                    bbox=(w * 0.25, h * 0.15, w * 0.5, h * 0.7),
                    landmarks=[],
                    confidence=0.5,
                    face_id=0,
                )
            )

        return faces

    async def _classify_emotion(
        self, face_region: "np.ndarray"
    ) -> Tuple[str, Dict[str, float], float]:
        """Classify emotion from face region.

        Args:
            face_region: Cropped face image

        Returns:
            Tuple of (emotion_label, probabilities, confidence)
        """
        if (
            self._emotion_classifier == "rule_based"
            or self._emotion_classifier == "fallback"
        ):
            return self._rule_based_emotion(face_region)
        else:
            return await self._transformer_emotion(face_region)

    def _rule_based_emotion(
        self, face_region: "np.ndarray"
    ) -> Tuple[str, Dict[str, float], float]:
        """Rule-based emotion classification using facial features.

        Uses simple image statistics for lightweight edge processing.

        Args:
            face_region: Cropped face image

        Returns:
            Tuple of (emotion_label, probabilities, confidence)
        """
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Compute simple features
        h, w = gray.shape

        # Upper/lower face intensity ratio
        upper = gray[: h // 2, :].mean()
        lower = gray[h // 2 :, :].mean()
        ul_ratio = upper / (lower + 1e-6)

        # Horizontal asymmetry
        left = gray[:, : w // 2].mean()
        right = gray[:, w // 2 :].mean()
        asymmetry = abs(left - right) / 255.0

        # Edge density (expression intensity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (h * w * 255)

        # Simple rule-based classification
        probabilities = {
            "neutral": 0.3,
            "happy": 0.1,
            "sad": 0.1,
            "angry": 0.1,
            "fearful": 0.1,
            "surprised": 0.1,
            "disgusted": 0.1,
            "contempt": 0.1,
        }

        # Adjust based on features
        if ul_ratio > 1.1:
            probabilities["surprised"] += 0.2
            probabilities["fearful"] += 0.1
        elif ul_ratio < 0.9:
            probabilities["sad"] += 0.2
            probabilities["angry"] += 0.1

        if edge_density > 0.15:
            probabilities["happy"] += 0.2
            probabilities["surprised"] += 0.1
        elif edge_density < 0.05:
            probabilities["neutral"] += 0.2

        if asymmetry > 0.05:
            probabilities["contempt"] += 0.15

        # Normalize
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}

        # Get primary emotion
        primary = max(probabilities.items(), key=lambda x: x[1])

        return primary[0], probabilities, primary[1]

    async def _transformer_emotion(
        self, face_region: "np.ndarray"
    ) -> Tuple[str, Dict[str, float], float]:
        """Transformer-based emotion classification.

        Args:
            face_region: Cropped face image

        Returns:
            Tuple of (emotion_label, probabilities, confidence)
        """
        import torch

        # Preprocess
        import cv2

        rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb_face, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._emotion_classifier(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]

        # Label mapping for vit-facial-expression-recognition
        labels = [
            "angry",
            "disgusted",
            "fearful",
            "happy",
            "neutral",
            "sad",
            "surprised",
        ]
        probabilities = {
            label: float(prob) for label, prob in zip(labels, probs.cpu().numpy())
        }

        # Add contempt if not in labels
        if "contempt" not in probabilities:
            probabilities["contempt"] = 0.0

        primary_idx = probs.argmax().item()
        primary_emotion = (
            labels[primary_idx] if primary_idx < len(labels) else "neutral"
        )
        confidence = float(probs[primary_idx])

        return primary_emotion, probabilities, confidence

    def _aggregate_frame_emotions(
        self, frame_emotions: List[FrameEmotion]
    ) -> EmotionalState:
        """Aggregate emotions across frames.

        Args:
            frame_emotions: List of frame emotions

        Returns:
            Aggregated emotional state
        """
        if not frame_emotions:
            return EmotionalState(
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
                confidence=0.0,
                primary_emotion="neutral",
                emotion_probabilities={},
            )

        # Weighted average by confidence
        total_weight = sum(e.confidence for e in frame_emotions)

        if total_weight == 0:
            total_weight = 1.0

        avg_valence = (
            sum(e.valence * e.confidence for e in frame_emotions) / total_weight
        )
        avg_arousal = (
            sum(e.arousal * e.confidence for e in frame_emotions) / total_weight
        )
        avg_dominance = (
            sum(e.dominance * e.confidence for e in frame_emotions) / total_weight
        )
        avg_confidence = total_weight / len(frame_emotions)

        # Average probabilities
        all_emotions = set()
        for e in frame_emotions:
            all_emotions.update(e.emotion_probabilities.keys())

        avg_probs = {}
        for emotion in all_emotions:
            total_prob = sum(
                e.emotion_probabilities.get(emotion, 0.0) * e.confidence
                for e in frame_emotions
            )
            avg_probs[emotion] = total_prob / total_weight

        # Primary emotion
        primary = (
            max(avg_probs.items(), key=lambda x: x[1])[0] if avg_probs else "neutral"
        )

        return EmotionalState(
            valence=avg_valence,
            arousal=avg_arousal,
            dominance=avg_dominance,
            confidence=avg_confidence,
            primary_emotion=primary,
            emotion_probabilities=avg_probs,
        )


class VideoEmotionTrajectory:
    """Track emotion changes across video frames."""

    def __init__(self, window_size: int = 30):
        """Initialize trajectory tracker.

        Args:
            window_size: Number of frames for trend calculation
        """
        self.window_size = window_size
        self.frame_emotions: List[Tuple[float, FrameEmotion]] = []

    def add_frame(self, timestamp_ms: float, emotion: FrameEmotion) -> None:
        """Add frame emotion sample."""
        self.frame_emotions.append((timestamp_ms, emotion))

    def get_trend(self) -> Dict[str, float]:
        """Calculate emotion trends.

        Returns:
            Trends for valence, arousal, dominance
        """
        if len(self.frame_emotions) < 2:
            return {"valence_trend": 0.0, "arousal_trend": 0.0, "dominance_trend": 0.0}

        window = self.frame_emotions[-self.window_size :]

        if len(window) < 2:
            return {"valence_trend": 0.0, "arousal_trend": 0.0, "dominance_trend": 0.0}

        timestamps = np.array([e[0] for e in window])
        valences = np.array([e[1].valence for e in window])
        arousals = np.array([e[1].arousal for e in window])
        dominances = np.array([e[1].dominance for e in window])

        valence_trend = float(np.polyfit(timestamps, valences, 1)[0])
        arousal_trend = float(np.polyfit(timestamps, arousals, 1)[0])
        dominance_trend = float(np.polyfit(timestamps, dominances, 1)[0])

        return {
            "valence_trend": valence_trend,
            "arousal_trend": arousal_trend,
            "dominance_trend": dominance_trend,
        }

    def get_stats(self) -> Dict[str, float]:
        """Calculate emotion statistics."""
        if not self.frame_emotions:
            return {}

        valences = [e[1].valence for e in self.frame_emotions]
        arousals = [e[1].arousal for e in self.frame_emotions]

        return {
            "mean_valence": float(np.mean(valences)),
            "std_valence": float(np.std(valences)),
            "mean_arousal": float(np.mean(arousals)),
            "std_arousal": float(np.std(arousals)),
            "min_valence": float(np.min(valences)),
            "max_valence": float(np.max(valences)),
            "min_arousal": float(np.min(arousals)),
            "max_arousal": float(np.max(arousals)),
        }
