from pydantic import BaseModel
from typing import List, Dict, Any

class PipelineInput(BaseModel):
    text: str

class EmotionFeatures(BaseModel):
    features: List[float]

class ContextualEmotions(BaseModel):
    context_vectors: List[List[float]]

class FlowDynamics(BaseModel):
    velocity: List[List[float]]
    acceleration: List[List[float]]
    momentum: List[List[float]]

class MetaIntelligence(BaseModel):
    deviation: float
    reflection_score: float

class FullPipelineOutput(BaseModel):
    emotion_features: EmotionFeatures
    contextual_emotions: ContextualEmotions
    flow_dynamics: FlowDynamics
    meta_intelligence: MetaIntelligence
