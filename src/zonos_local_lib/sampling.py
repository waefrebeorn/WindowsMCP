import torch


def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        # Add a small epsilon to input to prevent division by zero if input contains zeros,
        # though for probabilities, this should ideally not happen if properly normalized.
        # However, q can be zero if exponential_(1) produces a very large number, making 1/q zero.
        # A safer way is to use log-space sampling or ensure input probabilities are > 0.
        # For now, let's assume input is well-behaved (all non-negative, sums to 1).
        # If input can be zero, input/q can be NaN or Inf.
        # Gumbel-max trick: argmax(log(probs) + G) where G ~ Gumbel(0,1)
        # Or simpler: argmax(probs / random_exponential)
        # If input has zeros, those will remain zero. If q is zero for a non-zero input, it's Inf.
        # Let's ensure q is not zero where input is not zero.
        # q = torch.where(input > 0, q.clamp_min(1e-20), torch.inf) # Avoid division by zero only for valid inputs

        # A simpler approach if `input` contains probabilities (non-negative, sums to 1):
        # if torch.any(input < 0) or not torch.allclose(input.sum(dim=-1), torch.tensor(1.0, device=input.device, dtype=input.dtype)):
            # print("Warning: Input to multinomial may not be normalized probabilities.")
            # input = torch.softmax(input, dim=-1) # Force normalization if concerned

        # Original gumbel-like sampling:
        # For stability with potential zeros in `input` from softmax (e.g. from -inf logits)
        # or if `q` from exponential_ can be problematic.
        # Alternative: Gumbel-Max trick uses `logits + gumbel_noise`.
        # Here we have `probs`. `log(probs) / q` would be `log(probs) - log(q)`.
        # `input / q` is fine if `input` is probs and `q` is positive.
        # `torch.empty_like(input).exponential_(1)` generates positive values.

        # Check for NaN/Inf in `input/q` before argmax if there's a concern.
        # val = input / q
        # if torch.isinf(val).any() or torch.isnan(val).any():
        #     print("Warning: inf/nan encountered in multinomial sampling. Clamping might occur.")
        #     # Handle by replacing inf with large number, nan with small, before argmax
        #     val = torch.nan_to_num(val, nan=-torch.inf, posinf=torch.finfo(val.dtype).max, neginf=-torch.finfo(val.dtype).max)

        return torch.argmax(input / q.clamp_min(1e-30), dim=-1, keepdim=True).to(torch.int64)


    input_ = input.reshape(-1, input.shape[-1])
    # Ensure input_ to torch.multinomial is valid (non-negative, sum > 0 per row)
    # If a row sums to 0 (e.g. all probs are 0), torch.multinomial errors.
    # This can happen if top_k/top_p filters everything.
    # Add a check and potentially add uniform noise if a row is all zero.
    row_sums = input_.sum(dim=-1)
    if torch.any(row_sums == 0):
        # print("Warning: Zero-sum row found in multinomial input. Adding epsilon uniform distribution.")
        # This should ideally be handled by the caller (e.g., top_k/p ensuring valid distribution)
        # For now, if it happens, multinomial will error.
        # A simple fix: add a tiny value to all probabilities in zero-sum rows.
        # This is a bit of a hack; the samplers (top_p etc.) should ensure this.
        zero_sum_rows = row_sums == 0
        # input_[zero_sum_rows, :] = 1.0 / input_.shape[-1] # Make uniform for these rows
        # This might not be what's intended if everything was filtered out.
        # For now, let torch.multinomial handle it (it will error if a row sums to 0).
        pass


    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def apply_unified(probs: torch.Tensor, linear: float, conf: float, quad: float) -> torch.Tensor:
    """Sample next token using unified sampling approach that combines linear scaling, confidence, and quadratic terms.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        linear (float): Linear scaling factor applied to log probabilities.
        conf (float): Confidence factor that scales the entropy term.
        quad (float): Quadratic penalty factor applied to squared log probabilities.
    Returns:
        torch.Tensor: Modified probability distribution after applying unified sampling.
    """
    logprobs = torch.log(probs.clamp_min(1e-20)) # clamp_min to avoid log(0)
    entropy = -torch.sum(probs * logprobs, dim=-1, keepdim=True)
    raw = logprobs * (linear + entropy * conf) - logprobs**2 * quad
    return raw.softmax(dim=-1)

def apply_top_k(
    probs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    if k <= 0 or k >= probs.size(-1): # if k is non-positive or covers all, no change
        return probs

    v, _ = torch.topk(probs, min(k, probs.size(-1))) # Ensure k is not out of bounds
    pivot = v.select(-1, -1).unsqueeze(-1)
    probs_clone = probs.clone() # Avoid modifying original probs if it's passed around
    probs_clone[probs_clone < pivot] = 0.0 # Zero out elements not in top-k

    # Normalize
    sum_probs = probs_clone.sum(dim=-1, keepdim=True)
    # Avoid division by zero if sum_probs is zero (e.g., if k=1 and top prob was tiny and got zeroed somehow, or all probs were < pivot)
    # This can happen if k is too small or probs are too flat.
    # If sum_probs is 0, it means all probabilities were filtered. This is an issue.
    # A robust way: if sum is 0, fall back to original probs or uniform for those rows.
    # For now, let's assume sum_probs > 0. If not, division by zero will lead to NaNs.
    # To prevent NaNs:
    probs_clone = torch.where(sum_probs > 1e-9, probs_clone / sum_probs, torch.full_like(probs_clone, 1.0 / probs_clone.shape[-1])) # fallback to uniform if sum is zero
    return probs_clone


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    if p <= 0 or p >= 1: # if p covers none or all, no change (p=0 means only top1 if strictly > p, p=1 means all)
        if p <= 0: return probs # Or handle as "only top-1 if p=0" like some implementations
        if p >= 1: return probs

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p # Elements to remove

    probs_sort_clone = probs_sort.clone() # Avoid in-place modification issues
    probs_sort_clone[mask] = 0.0 # Zero out elements not in top-p nucleus

    # Re-scatter to original order
    probs_out = torch.zeros_like(probs)
    probs_out.scatter_(-1, probs_idx, probs_sort_clone)

    # Normalize
    sum_probs_out = probs_out.sum(dim=-1, keepdim=True)
    probs_out = torch.where(sum_probs_out > 1e-9, probs_out / sum_probs_out, torch.full_like(probs_out, 1.0 / probs_out.shape[-1])) # fallback to uniform
    return probs_out


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Sample next token using min-p sampling.

    Args:
        scores (torch.FloatTensor): Input logits with token candidates on the last dimension.
        min_p (float): Minimum token probability, scaled by the probability of the most likely token.
                       Must be between 0 and 1. Typical values are in the 0.01-0.2 range.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    if min_p <= 0:
        return probs

    top_probs, _ = probs.max(dim=-1, keepdim=True)
    tokens_to_remove = probs < (min_p * top_probs)

    probs_clone = probs.clone()
    probs_clone[tokens_to_remove] = 0.0

    sum_probs_clone = probs_clone.sum(dim=-1, keepdim=True)
    probs_clone = torch.where(sum_probs_clone > 1e-9, probs_clone / sum_probs_clone, torch.full_like(probs_clone, 1.0 / probs_clone.shape[-1]))
    return probs_clone


def modify_logit_for_repetition_penalty(
    logits: torch.Tensor, # [batch_size, n_codebooks, vocab_size]
    generated_tokens: torch.Tensor, # [batch_size, n_codebooks, seq_len]
    repetition_penalty: float,
    repetition_penalty_window: int,
):
    """See https://arxiv.org/abs/1909.05858
    Apply repetition penalty over a sliding window of the last `repetition_penalty_window` tokens.
    """
    if repetition_penalty == 1.0 or repetition_penalty_window <= 0:
        return logits

    # Consider only the last `repetition_penalty_window` tokens
    # Ensure generated_tokens has enough history; if not, use what's available.
    window = min(generated_tokens.shape[-1], repetition_penalty_window)
    if window == 0: return logits # No history to penalize

    tokens_in_window = generated_tokens[..., -window:] # [B, N_q, W]

    # Clamp token IDs to be valid indices for logits
    # vocab_size is logits.shape[-1]
    vocab_size = logits.shape[-1]
    tokens_in_window = tokens_in_window.clamp(0, vocab_size - 1)

    # Create factors for penalty: initialize with 1s
    # For each token in logits, if it appeared in `tokens_in_window`, its factor becomes `repetition_penalty`
    # This is not quite right. We need to iterate over generated tokens and apply penalty.
    # The original paper's formula:
    # For positive logits, divide by penalty. For negative, multiply by penalty.

    # Create a mask for tokens that appeared in the window
    # logits_clone = logits.clone() # Work on a clone

    # For each batch and codebook
    for b_idx in range(logits.shape[0]):
        for q_idx in range(logits.shape[1]):
            # Get unique tokens that appeared in the window for this batch/codebook
            unique_tokens_in_window = torch.unique(tokens_in_window[b_idx, q_idx, :])

            for token_id in unique_tokens_in_window:
                # Apply penalty to the logit of this token_id
                current_logit = logits[b_idx, q_idx, token_id]
                if current_logit > 0:
                    logits[b_idx, q_idx, token_id] = current_logit / repetition_penalty
                else:
                    logits[b_idx, q_idx, token_id] = current_logit * repetition_penalty
    return logits


def sample_from_logits(
    logits: torch.Tensor, # [batch_size, num_codebooks, vocab_size]
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    linear: float = 0.0, # For unified sampler
    conf: float = 0.0,   # For unified sampler
    quad: float = 0.0,   # For unified sampler
    generated_tokens: torch.Tensor | None = None, # [batch_size, num_codebooks, seq_len_generated]
    repetition_penalty: float = 1.0, # Changed default from 3.0 to 1.0 (no penalty)
    repetition_penalty_window: int = 2,
) -> torch.Tensor:
    """Sample next token from logits using either top_k/p/min_p OR using NovelAI's Unified Sampler.

    Args:
        logits (torch.Tensor): Input logits with token candidates on the last dimension.

        temperature (float): Randomness of the sampling. Lower temperature results in more deterministic samples.
            To disable sampling entirely, set it to 0. For NovelAI's Unified Sampler, set it to 1.0

        top_p (float): Only sample from the most probable tokens whose cumulative probability is less than p.
            This is called nucleus sampling. Must be between 0 and 1. Typical values are in the 0.1-0.9 range.
            Set to 0 to disable.

        top_k (int): Only sample from the top k most probable tokens. Set to 0 to disable.

        min_p (float): Minimum token probability, scaled by the probability of the most likely token.
                       Must be between 0 and 1. Typical values are in the 0.01-0.2 range.
                       If too high, no token might be sampled leading to silence (?)

        linear (float): NovelAI's Unified Sampler -> 0.0 to 1.0, default from gradio 0.5
            Set Linear between 0 and 1 according to how unusual you want tokens to be.
            Lower numbers will produce more unusual/creative outputs,
            but you will have to reroll or edit more.

        conf (float): Confidence - Low values make random outputs more random. -> -2.0 * Quad to 2.0, default from gradio 0.4
            As a starting point, set Quad = 1/3 - Linear * 4 / 15, and Conf = -Quad / 2.

        quad (float): Quadratic - High values make low probablities much lower. -> -2.0 to 2.0, default from gradio 0.0

    Returns:
        torch.Tensor: Sampled tokens of shape [batch_size, num_codebooks, 1]
    """

    # Apply repetition penalty to logits first
    if repetition_penalty != 1.0 and generated_tokens is not None and repetition_penalty_window > 0:
        logits = modify_logit_for_repetition_penalty(
            logits, generated_tokens, repetition_penalty, repetition_penalty_window
        )

    if temperature == 0: # Greedy sampling
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        # Apply temperature
        probs = torch.softmax(logits / temperature, dim=-1)

        # Apply sampling strategies
        if linear > 0.0: # NovelAI's Unified Sampler (assumes temp=1 for this path)
            if abs(temperature - 1.0) > 1e-3 : # Check if temperature is not ~1.0
                 print("Warning: Unified Sampler (linear > 0) is typically used with temperature=1.0.", file=sys.stderr)
            probs = apply_unified(probs, linear, conf, quad)

        # These can be chained: top_k, then top_p, then min_p
        if top_k > 0:
            probs = apply_top_k(probs, top_k)
        if top_p > 0: # top_p is applied *after* top_k if both are set
            probs = apply_top_p(probs, top_p)
        if min_p > 0: # min_p is applied *after* top_p/top_k
            probs = apply_min_p(probs, min_p)

        # Ensure probabilities are still valid after filtering (e.g., sum to 1, no NaNs)
        # This is crucial if any filter zeroed out all probabilities for a row.
        # If a row in `probs` sums to 0, multinomial will error.
        # A simple check: if sum is near zero, make it uniform for that row.
        # This should ideally be handled within each apply_X function.
        # For example, if top_k(1) is used and the top_1 prob is 0, then sum is 0.
        row_sums = probs.sum(dim=-1, keepdim=True)
        # If any row has all probs zeroed out by filters, fall back to uniform for that row.
        # This prevents errors in `multinomial` and ensures sampling can proceed.
        probs = torch.where(row_sums < 1e-9, torch.full_like(probs, 1.0 / probs.shape[-1]), probs / row_sums.clamp_min(1e-9))

        next_token = multinomial(probs, num_samples=1)

    return next_token # [batch_size, num_codebooks, 1]
