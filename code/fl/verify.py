from fluctlight.model import FluctlightTransformer

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Detailed breakdown by layer
    layer_params = {}
    for name, param in model.named_parameters():
        layer_params[name] = param.numel()
    
    # Sort by name for better readability
    sorted_layers = sorted(layer_params.items())
    
    return total_params, trainable_params, sorted_layers

# Create model with the exact architecture described
model = FluctlightTransformer(
    vocab_size=256,
    d_model=4,
    n_heads=2,
    n_layers=2,
    d_ff=8
)

total, trainable, layers = count_parameters(model)

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print("\nParameter breakdown by layer:")
for name, count in layers:
    print(f"{name}: {count:,}")
