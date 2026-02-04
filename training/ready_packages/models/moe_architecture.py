#!/usr/bin/env python3
"""
Multi-expert Mixture of Experts (MoE) Architecture for Therapeutic AI
Implements domain-specific expert routing for psychology, mental health, and bias detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class MoEConfig:
    """Configuration for MoE architecture"""
    num_experts: int = 4
    expert_domains: List[str] = None
    hidden_size: int = 4096
    expert_capacity: int = 2  # Number of experts per token
    load_balancing_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Extended context
    max_position_embeddings: int = 8192
    
    def __post_init__(self):
        if self.expert_domains is None:
            self.expert_domains = [
                "psychology",
                "mental_health", 
                "bias_detection",
                "general_therapeutic"
            ]
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class ExpertRouter(nn.Module):
    """
    Routes inputs to appropriate domain experts based on content classification
    Uses learned routing with load balancing
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        
        # Router network
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        
        # Domain classifiers for interpretability
        self.domain_classifiers = nn.ModuleDict({
            domain: nn.Linear(config.hidden_size, 1)
            for domain in config.expert_domains
        })
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_routing_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Route tokens to experts
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            return_routing_weights: Whether to return routing weights for analysis
            
        Returns:
            expert_indices: [batch_size, seq_len, expert_capacity]
            routing_weights: [batch_size, seq_len, expert_capacity]
            routing_info: Optional dict with routing statistics
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]
        
        # Apply softmax to get routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        routing_weights, expert_indices = torch.topk(
            routing_probs, 
            k=self.expert_capacity, 
            dim=-1
        )
        
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        routing_info = None
        if return_routing_weights:
            # Compute domain-specific scores for interpretability
            domain_scores = {
                domain: torch.sigmoid(classifier(hidden_states))
                for domain, classifier in self.domain_classifiers.items()
            }
            
            # Compute load balancing metrics
            expert_usage = torch.zeros(self.num_experts, device=hidden_states.device)
            for i in range(self.num_experts):
                expert_usage[i] = (expert_indices == i).float().sum()
            
            routing_info = {
                'domain_scores': domain_scores,
                'expert_usage': expert_usage,
                'routing_entropy': -(routing_probs * torch.log(routing_probs + 1e-10)).sum(dim=-1).mean()
            }
        
        return expert_indices, routing_weights, routing_info
    
    def compute_load_balancing_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage
        """
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Average probability of routing to each expert
        expert_probs = routing_probs.mean(dim=[0, 1])  # [num_experts]
        
        # Ideal uniform distribution
        uniform_probs = torch.ones_like(expert_probs) / self.num_experts
        
        # KL divergence from uniform distribution
        load_balancing_loss = F.kl_div(
            expert_probs.log(),
            uniform_probs,
            reduction='batchmean'
        )
        
        return load_balancing_loss


class DomainExpert(nn.Module):
    """
    Individual expert specialized for a specific domain
    """
    
    def __init__(self, config: MoEConfig, domain: str):
        super().__init__()
        self.config = config
        self.domain = domain
        
        # Expert-specific feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process hidden states through expert network
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            expert_output: [batch_size, seq_len, hidden_size]
        """
        # Residual connection with expert FFN
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        expert_output = self.ffn(hidden_states)
        return residual + expert_output


class MoELayer(nn.Module):
    """
    Complete MoE layer with routing and expert processing
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Router
        self.router = ExpertRouter(config)
        
        # Domain experts
        self.experts = nn.ModuleList([
            DomainExpert(config, domain)
            for domain in config.expert_domains
        ])
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through MoE layer
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            return_routing_info: Whether to return routing information
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            routing_info: Optional routing information
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route to experts
        expert_indices, routing_weights, routing_info = self.router(
            hidden_states,
            return_routing_weights=return_routing_info
        )
        
        # Process through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Mask for tokens routed to this expert
            expert_mask = (expert_indices == i).any(dim=-1)  # [batch, seq_len]
            
            if expert_mask.any():
                # Process tokens assigned to this expert
                expert_output = expert(hidden_states)
                expert_outputs.append(expert_output)
            else:
                # No tokens for this expert
                expert_outputs.append(torch.zeros_like(hidden_states))
        
        # Combine expert outputs using routing weights
        output = torch.zeros_like(hidden_states)
        for i in range(self.config.expert_capacity):
            expert_idx = expert_indices[:, :, i]  # [batch, seq_len]
            weights = routing_weights[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            
            for expert_id in range(self.config.num_experts):
                mask = (expert_idx == expert_id).unsqueeze(-1)  # [batch, seq_len, 1]
                output += mask * weights * expert_outputs[expert_id]
        
        return output, routing_info


class TherapeuticMoEModel(nn.Module):
    """
    Complete therapeutic AI model with MoE architecture and LoRA fine-tuning
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: MoEConfig
    ):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Add MoE layers
        self.moe_layers = nn.ModuleList([
            MoELayer(config)
            for _ in range(4)  # Add MoE to 4 transformer layers
        ])
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none"
        )
        
        # Apply LoRA to base model
        self.model = get_peft_model(base_model, lora_config)
        
        # Extend context length if needed
        if hasattr(self.model.config, 'max_position_embeddings'):
            self.model.config.max_position_embeddings = config.max_position_embeddings
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
        **kwargs
    ):
        """
        Forward pass through therapeutic MoE model
        """
        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        # Apply MoE layers to hidden states
        hidden_states = outputs.hidden_states[-1]
        routing_infos = []
        
        for moe_layer in self.moe_layers:
            hidden_states, routing_info = moe_layer(
                hidden_states,
                return_routing_info=return_routing_info
            )
            if routing_info:
                routing_infos.append(routing_info)
        
        # Compute final logits
        logits = self.model.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add load balancing loss
            if routing_infos:
                load_balancing_loss = sum(
                    info.get('load_balancing_loss', 0)
                    for info in routing_infos
                ) / len(routing_infos)
                loss = loss + self.config.load_balancing_weight * load_balancing_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'routing_info': routing_infos if return_routing_info else None
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration"""
        self.model.save_pretrained(save_directory)
        
        # Save MoE layers separately
        moe_state = {
            'config': self.config,
            'moe_layers': [layer.state_dict() for layer in self.moe_layers]
        }
        torch.save(moe_state, f"{save_directory}/moe_layers.pt")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, base_model: PreTrainedModel):
        """Load model from directory"""
        # Load MoE state
        moe_state = torch.load(f"{load_directory}/moe_layers.pt")
        config = moe_state['config']
        
        # Create model
        model = cls(base_model, config)
        
        # Load MoE layer weights
        for layer, state_dict in zip(model.moe_layers, moe_state['moe_layers']):
            layer.load_state_dict(state_dict)
        
        return model


def create_therapeutic_moe_model(
    base_model_name: str,
    moe_config: Optional[MoEConfig] = None,
    device: str = "auto"
) -> TherapeuticMoEModel:
    """
    Create a therapeutic MoE model from a base model
    
    Args:
        base_model_name: HuggingFace model name
        moe_config: MoE configuration (uses defaults if None)
        device: Device to load model on
        
    Returns:
        TherapeuticMoEModel instance
    """
    from transformers import AutoModelForCausalLM
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Create MoE config if not provided
    if moe_config is None:
        moe_config = MoEConfig()
    
    # Create therapeutic MoE model
    model = TherapeuticMoEModel(base_model, moe_config)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("🧠 Therapeutic MoE Architecture")
    print("=" * 50)
    
    config = MoEConfig(
        num_experts=4,
        expert_domains=["psychology", "mental_health", "bias_detection", "general_therapeutic"],
        lora_r=16,
        lora_alpha=32,
        max_position_embeddings=8192
    )
    
    print(f"✅ Configuration:")
    print(f"   - Experts: {config.num_experts}")
    print(f"   - Domains: {', '.join(config.expert_domains)}")
    print(f"   - LoRA rank: {config.lora_r}")
    print(f"   - Context length: {config.max_position_embeddings}")
    print(f"   - Expert capacity: {config.expert_capacity}")
    
    print("\n📊 Model ready for training!")
