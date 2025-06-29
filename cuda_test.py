import torch

print(f"Torch Version: {torch.__version__}")
print(torch.version.cuda)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA device
    print("CUDA is available! Using GPU.")
    print("GPU device name:", torch.cuda.get_device_name(0)) #prints the name of the GPU
else:
    device = torch.device("cpu")  # Use CPU device
    print("CUDA is NOT available. Using CPU.")

# Example tensor creation (on the chosen device)
tensor = torch.randn(3, 3).to(device)
print(tensor)