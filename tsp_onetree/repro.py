import os
import random

# CUDA reads this setting during initialization.  Establish the default before
# importing torch so module execution receives the same protection as the
# historical wrapper entry point.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    seed = int(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def make_torch_generator(seed: int, device: str | torch.device = 'cpu') -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen

def sample_batch_permutations(batch_size: int, n: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.stack([torch.randperm(n, generator=generator, device=device) for _ in range(batch_size)], dim=0)
