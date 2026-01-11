import json
import os

import torch
import yaml
from ai.pixel.pipeline.schemas import (
    ContextualEmotions,
    EmotionFeatures,
    FlowDynamics,
    FullPipelineOutput,
    MetaIntelligence,
    PipelineInput,
)
from ai.pixel.research.emotional_cnn_layer import (
    EmotionalCNNConfig,
    EmotionalCNNTextEncoder,
)
from ai.pixel.research.emotional_flow_dynamics import EmotionalFlowDynamics
from ai.pixel.research.emotional_resnet_memory import EmotionalResNetMemory
from ai.pixel.research.meta_emotional_intelligence import MetaEmotionalIntelligence
from ai.pixel.utils.path_utils import get_project_root
from torch import nn


class EmotionalPipeline(nn.Module):
    def __init__(self, config_path: str = "config/emotional_pipeline.yaml"):
        super().__init__()
        project_root = get_project_root()
        absolute_config_path = project_root / config_path
        with open(absolute_config_path) as f:
            self.config = yaml.safe_load(f)

        self.modules_map = nn.ModuleDict()
        for module_config in self.config.get("modules", []):
            if module_config.get("enabled", False):
                name = module_config["name"]
                if name == "emotional_cnn_layer":
                    # Dummy values for vocab_size, will need to be configured properly
                    self.modules_map[name] = EmotionalCNNTextEncoder(
                        vocab_size=1000, config=EmotionalCNNConfig()
                    )
                elif name == "emotional_resnet_memory":
                    # Dummy values, will need to be configured
                    self.modules_map[name] = EmotionalResNetMemory(
                        input_dim=32, hidden_dim=64
                    )
                elif name == "emotional_flow_dynamics":
                    self.modules_map[name] = EmotionalFlowDynamics(input_dim=32)
                elif name == "meta_emotional_intelligence":
                    self.modules_map[name] = MetaEmotionalIntelligence(input_dim=32)

    def forward(self, text: str) -> FullPipelineOutput:
        # This is a simplified forward pass. A real implementation would need proper tokenization and tensor conversion.
        PipelineInput(text=text)

        # 1. CNN Layer
        if "emotional_cnn_layer" in self.modules_map:
            cnn_module = self.modules_map["emotional_cnn_layer"]
            # Dummy tokenization and tensor creation
            dummy_input_ids = torch.randint(0, 1000, (1, 20))
            features_tensor = cnn_module(dummy_input_ids)
            emotion_features = EmotionFeatures(features=features_tensor.squeeze().tolist())
        else:
            # Dummy data if module is disabled
            emotion_features = EmotionFeatures(features=[0.0] * 32)

        # For subsequent modules, we need sequence data. We'll simulate a sequence of 1 for now.
        context_input_tensor = (
            torch.tensor([emotion_features.features]).unsqueeze(0)
        )  # (Batch, Seq, Dim)

        # 2. ResNet Memory
        if "emotional_resnet_memory" in self.modules_map:
            resnet_module = self.modules_map["emotional_resnet_memory"]
            context_tensor = resnet_module(context_input_tensor)
            contextual_emotions = ContextualEmotions(
                context_vectors=context_tensor.squeeze(0).tolist()
            )
        else:
            contextual_emotions = ContextualEmotions(context_vectors=[[0.0] * 32])

        # Use the output of ResNet (or the raw features) for the next two modules
        final_context_tensor = torch.tensor(
            contextual_emotions.context_vectors
        ).unsqueeze(0)

        # 3. Flow Dynamics
        if "emotional_flow_dynamics" in self.modules_map:
            flow_module = self.modules_map["emotional_flow_dynamics"]
            vel, acc, mom = flow_module(final_context_tensor)
            flow_dynamics = FlowDynamics(
                velocity=vel.squeeze(0).tolist(),
                acceleration=acc.squeeze(0).tolist(),
                momentum=mom.squeeze(0).tolist(),
            )
        else:
            flow_dynamics = FlowDynamics(
                velocity=[[0.0] * 32],
                acceleration=[[0.0] * 32],
                momentum=[[0.0] * 32],
            )

        # 4. Meta Intelligence
        if "meta_emotional_intelligence" in self.modules_map:
            meta_module = self.modules_map["meta_emotional_intelligence"]
            meta_metrics = meta_module(final_context_tensor)
            meta_intelligence = MetaIntelligence(
                deviation=meta_metrics["deviation"].item(),
                reflection_score=meta_metrics["reflection_score"].item(),
            )
        else:
            meta_intelligence = MetaIntelligence(deviation=0.0, reflection_score=0.0)

        output = FullPipelineOutput(
            emotion_features=emotion_features,
            contextual_emotions=contextual_emotions,
            flow_dynamics=flow_dynamics,
            meta_intelligence=meta_intelligence,
        )

        self._log_run(text, output)

        return output

    def _log_run(self, text: str, output: FullPipelineOutput):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "pipeline_runs.jsonl")

        log_record = {
            "config": self.config,
            "input": {"text": text},
            "output": output.dict(),
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(log_record) + "\n")

