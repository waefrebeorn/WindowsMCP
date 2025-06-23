import torch
import torch.nn as nn
import torch.nn.functional as F


def find_multiple(n: int, k: int) -> int:
    if k == 0: # Avoid division by zero if k is 0
        return n
    if n % k == 0:
        return n
    return n + k - (n % k)


def pad_weight_(w: nn.Embedding | nn.Linear, multiple: int):
    """Pad the weight of an embedding or linear layer to a multiple of `multiple`.
    This function modifies the layer's weight in-place.
    """
    if multiple == 0: # No padding if multiple is 0
        return

    if isinstance(w, nn.Embedding):
        # Pad vocabulary dimension (num_embeddings, which is dim 0 of weight matrix)
        current_vocab_size = w.weight.shape[0]
        padded_vocab_size = find_multiple(current_vocab_size, multiple)
        if padded_vocab_size == current_vocab_size:
            return

        padding_rows = padded_vocab_size - current_vocab_size
        # Pad the weight tensor (dim 0)
        # Parameter data should be modified, not the Parameter object replaced for nn.Module
        padding_tensor = torch.zeros(padding_rows, w.weight.shape[1], device=w.weight.device, dtype=w.weight.dtype)
        w.weight.data = torch.cat([w.weight.data, padding_tensor], dim=0)
        w.num_embeddings = padded_vocab_size # Update the layer's attribute

    elif isinstance(w, nn.Linear):
        # Pad output dimension (out_features, which is dim 0 of weight matrix)
        current_out_features = w.weight.shape[0]
        padded_out_features = find_multiple(current_out_features, multiple)
        if padded_out_features == current_out_features:
            return

        padding_rows = padded_out_features - current_out_features
        # Pad the weight tensor (dim 0)
        padding_weight_tensor = torch.zeros(padding_rows, w.weight.shape[1], device=w.weight.device, dtype=w.weight.dtype)
        w.weight.data = torch.cat([w.weight.data, padding_weight_tensor], dim=0)

        # Pad bias if it exists (dim 0)
        if w.bias is not None:
            padding_bias_tensor = torch.zeros(padding_rows, device=w.bias.device, dtype=w.bias.dtype)
            w.bias.data = torch.cat([w.bias.data, padding_bias_tensor], dim=0)

        w.out_features = padded_out_features # Update the layer's attribute
    else:
        raise ValueError(f"Unsupported layer type for pad_weight_: {type(w)}. Expected nn.Embedding or nn.Linear.")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # return torch.device(torch.cuda.current_device()) # Returns index, e.g. "cuda:0"
        return torch.device("cuda") # Prefer generic "cuda" to let PyTorch manage current device
    # MPS (Apple Silicon GPU) check:
    # Original code commented this out.
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


DEFAULT_DEVICE = get_device()
