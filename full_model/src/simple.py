import os
os.environ['TORCH_USE_HIP_DSA'] = '1'

import torch

# Check if ROCm is available
print("Is ROCm available?", torch.cuda.is_available())

# List all available devices
print("Available devices:")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Create a tensor and move it to the GPU
device = torch.device('cuda:0')
print(f"Using device: {torch.cuda.get_device_name(device)}")
x = torch.randn(5, 3, device=device)
print("Tensor on GPU:", x)

# Perform a simple operation
y = x * 2
print("Result tensor on GPU:", y)
