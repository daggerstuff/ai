from torch import serialization
from torch import _C

def from_pretrained(cls, load_directory, base_model):
    """Load model from directory"""
    # Load MoE state
    import torch
    torch_version = torch.__version__
    with torch.serialization.safe_globals([MoEConfig]):
        moe_state = torch.load(f"{load_directory}/moe_layers.pt", weights_only=True)

    # Create model
    model = cls(base_model, moe_state['config'])

    # Load MoE layer weights
    for layer, state_dict in zip(model.moe_layers, moe_state['moe_layers']):
        layer.load_state_dict(state_dict)

    return model