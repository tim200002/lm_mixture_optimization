import torch

def allocate_memory_on_gpu(size_in_gb: float, device):
    """
    Allocates a tensor on the GPU to block memory.
    
    Args:
    size_in_gb (float): Size of the tensor to allocate in GB.
    
    Returns:
    torch.Tensor: The allocated tensor.
    """
    size_in_bytes = size_in_gb * 1024 ** 3
    size_in_floats = int(size_in_bytes / 4)  # Each float32 takes 4 bytes
    tensor = torch.ones(size_in_floats, device=device)
    return tensor

def deallocate_memory_on_gpu(tensor):
    """
    Deallocates the tensor to free memory on the GPU.
    
    Args:
    tensor (torch.Tensor): The tensor to deallocate.
    """
    del tensor
    torch.cuda.empty_cache()