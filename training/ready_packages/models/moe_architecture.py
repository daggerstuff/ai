from torch import serialization
from torch import _C

def from_pretrained(cls, load_directory, base_model):
    """Load model from directory"""
    # Load MoE state
    torch.serialization.add_safe_globals([_C._get_torch_version])  # <--- FIX: Add safe globals for PyTorch 2.1–2.3
    moe_state = torch.load(f"{load_directory}/moe_layers.pt", map_location=torch.device('cpu'))  # <--- FIX: Use map_location to ensure compatibility

    # Create model
    model = cls(base_model, moe_state['config'])

    # Load MoE layer weights
    for layer, state_dict in zip(model.moe_layers, moe_state['moe_layers']):
        layer.load_state_dict(state_dict)

    return model