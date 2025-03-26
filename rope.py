from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    shape = [1] * ndim
    shape[1] = x.shape[1]  # Match sequence length
    shape[-1] = freqs_cis.shape[-1]  # Match frequency tensor's last dimension
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    _, seqlen, _, _ = query.shape
    device = query.device

    # Separate real and imaginary parts
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Generate sinusoidal frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seqlen, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex representation

    # Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, query)

    # Apply rotary embeddings
    query_complex = torch.view_as_complex(torch.stack((query_real, query_imag), dim=-1))
    key_complex = torch.view_as_complex(torch.stack((key_real, key_imag), dim=-1))

    query_out = torch.view_as_real(query_complex * freqs_cis)
    key_out = torch.view_as_real(key_complex * freqs_cis)

    # Reshape back to original dimensions
    query_out = query_out.flatten(-2)
    key_out = key_out.flatten(-2)

    return query_out, key_out