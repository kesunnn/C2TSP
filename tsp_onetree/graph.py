import torch

def build_candidate_mask(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Symmetric kNN candidate graph. Returns bool mask of shape (B, n, n).
    Includes reverse-kNN closure and excludes self loops.
    """
    B, n, _ = dist_matrix.shape
    device = dist_matrix.device
    full_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    if k >= n - 1:
        return full_mask

    d = dist_matrix.clone()
    d = d + torch.eye(n, device=device).unsqueeze(0) * 1e9
    knn_idx = torch.topk(d, k=k, largest=False, dim=-1).indices
    cand = torch.zeros(B, n, n, device=device, dtype=torch.bool)
    cand.scatter_(dim=-1, index=knn_idx, value=True)
    cand = cand | cand.transpose(-2, -1)
    cand = cand & full_mask
    return cand
