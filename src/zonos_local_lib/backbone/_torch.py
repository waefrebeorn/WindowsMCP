# Based on gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/095b2229ee3a40e379c11f05b94bd6923db63b4b/model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..config import BackboneConfig, InferenceParams # Adjusted import


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def _update_kv_cache(
    k: torch.Tensor, v: torch.Tensor, inference_params: InferenceParams, layer_idx: int
) -> torch.Tensor:
    """k/v: (batch_size, seqlen, nheads, head_dim) or (batch_size, 1, nheads, head_dim)"""
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + k.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + k.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, 0, ...] = k
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, 1, ...] = v
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class TorchZonosBackbone(nn.Module):
    supported_architectures = ["transformer"]
    freqs_cis: torch.Tensor

    def __init__(self, config: BackboneConfig):
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layer))
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        head_dim = self.config.d_model // self.config.attn_cfg["num_heads"]

        # Determine the target device from an existing parameter of the module
        module_device = self.norm_f.weight.device

        # Compute freqs_cis on CPU then move to target device if not already there or on correct device
        # Check if it needs recomputation or moving (e.g. if device changed or not initialized)
        if not hasattr(self, 'freqs_cis') or self.freqs_cis.device != module_device or self.freqs_cis.shape[0] < 16384 : # check size too
            # Max sequence length for RoPE is often fixed (e.g., 16384 or a large value from config)
            cpu_freqs_cis = precompute_freqs_cis(16384, head_dim) # This is on CPU
            self.freqs_cis = cpu_freqs_cis.to(module_device)

        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, device=module_device) # Pass device
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
        current_device = hidden_states.device
        input_pos = torch.arange(0, hidden_states.shape[1], device=current_device)

        # Ensure lengths_per_sample is on the same device as input_pos for the addition
        lengths_per_sample_on_device = inference_params.lengths_per_sample.to(current_device)
        input_pos = input_pos + lengths_per_sample_on_device.unsqueeze(-1)

        # self.freqs_cis should now be on current_device due to changes in allocate_inference_cache
        # However, input_pos might still be on a different device if hidden_states.device was different
        # from self.freqs_cis.device at the time of freqs_cis creation.
        # Safest is to ensure input_pos is on the same device as self.freqs_cis OR input_pos is CPU.
        # Since self.freqs_cis is now moved to module_device, input_pos should also be on module_device.
        # hidden_states.device (current_device) should be module_device.

        # If self.freqs_cis is guaranteed to be on current_device:
        freqs_cis_for_indexing = self.freqs_cis
        indices_for_indexing = input_pos

        # If there's a chance self.freqs_cis is on CPU and input_pos on CUDA (or vice-versa in error):
        if freqs_cis_for_indexing.device != indices_for_indexing.device:
            # This case should ideally not happen if allocate_inference_cache sets freqs_cis to current_device
            # and current_device is consistent. But as a safeguard:
            indices_for_indexing = indices_for_indexing.to(freqs_cis_for_indexing.device)

        selected_freqs_cis = freqs_cis_for_indexing[indices_for_indexing]
        freqs_cis = selected_freqs_cis.expand(hidden_states.shape[0], -1, -1, -1)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, inference_params, freqs_cis)
        return self.norm_f(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = Attention(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // config.attn_cfg["num_heads"]

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = None):
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim, dtype=dtype, device=device), None

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        self.layer_idx = layer_idx

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, _ = x.shape

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        q, k, v = self.in_proj(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(batch_size, seqlen, self.num_heads, self.head_dim)
        k = k.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)
        v = v.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        kv = _update_kv_cache(k, v, inference_params, self.layer_idx)
        k, v = kv.unbind(dim=-3)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        y = F.scaled_dot_product_attention(q, k, v, is_causal=seqlen > 1, enable_gqa=True)

        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, q_size)

        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))
