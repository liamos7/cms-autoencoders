import torch

# Load the checkpoint (will work on GPU)
model_path = "/scratch/network/lo8603/thesis/fast-ad/outputs/run2/model_best.pkl"
checkpoint = torch.load(model_path)

# Recursively move all tensors to CPU
def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    else:
        return obj

checkpoint_cpu = to_cpu(checkpoint)

# Save with CPU tensors
output_path = "/scratch/network/lo8603/thesis/fast-ad/outputs/run2/best_cpu.pkl"
torch.save(checkpoint_cpu, output_path)
print(f"Converted model saved to {output_path}")
