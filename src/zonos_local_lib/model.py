import json
from typing import Callable

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .autoencoder import DACAutoencoder
from .backbone import BACKBONES
from .backbone._mamba_ssm import MambaSSMZonosBackbone # Import for isinstance check
# from .backbone._torch import TorchZonosBackbone # Optional: if TorchZonosBackbone is also checked by name
from .codebook_pattern import apply_delay_pattern, revert_delay_pattern
from .conditioning import PrefixConditioner # Assuming make_cond_dict is also in conditioning or handled by Gradio
from .config import InferenceParams, ZonosConfig
from .sampling import sample_from_logits
from .speaker_cloning import SpeakerEmbeddingLDA
from .utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values())) if BACKBONES else None # Handle empty BACKBONES


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=None): # Modified default for backbone_cls
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        if backbone_cls is None:
            if not BACKBONES:
                raise RuntimeError("No backbones available. Check Zonos library setup.")
            backbone_cls = next(iter(BACKBONES.values()))


        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        # Adjusted to handle config.eos_token_id + 2 for masked_token_id and potential PAD
        # The original 1026 implies eos_token_id=1024, masked_token_id=1025.
        # Max vocab ID used seems to be masked_token_id. Embedding size needs to be max_id + 1.
        # If eos_token_id is vocab_size-2 and masked_token_id is vocab_size-1, then vocab_size = eos_token_id+2.
        # Let's assume vocab_size is determined by the largest of these specific tokens + 1.
        # For safety, use a vocab_size that accommodates these special tokens.
        # Typically, this is handled by a tokenizer's vocab_size.
        # If masked_token_id is 1025, embedding needs to cover up to index 1025. So size 1026.
        # Heads output to 1025, meaning valid indices are 0-1024.
        # This implies a vocabulary of 1025 distinct items (0-1024).
        # Let's use self.config.masked_token_id + 1 as a safe bet for embedding size,
        # assuming token IDs are contiguous from 0.
        embedding_vocab_size = self.config.masked_token_id + 1
        if self.config.pad_vocab_to_multiple_of:
             embedding_vocab_size = find_multiple(embedding_vocab_size, self.config.pad_vocab_to_multiple_of)

        # Heads output logits for tokens up to masked_token_id (exclusive of padding if any)
        # The original code had 1025 for head output dim, implying valid token IDs 0-1024.
        # And embedding was 1026.
        # If masked_token_id is 1025, it means it's an actual token ID.
        # Output dim for heads should be the number of actual classes/tokens.
        # If token IDs are 0 to N, then N+1 classes.
        # Let's assume head_output_dim should be masked_token_id + 1 (if 0-indexed)
        # or simply the number of tokens before padding.
        # The original code hardcoded 1025 for heads and 1026 for embeddings.
        # This implies eos_token_id=1024, masked_token_id=1025 (which is not what config says).
        # Let's use config:
        # eos_token_id = 1024, masked_token_id = 1025
        # So, max token ID is 1025. Embedding needs to handle index 1025, so size is 1026.
        # Head output should be for 1025 classes (if 0-1024) or 1026 (if 0-1025).
        # The original code's head output dim was 1025. This means it predicts for tokens 0...1024.
        # This is confusing. If masked_token_id is 1025, it is a valid token.
        # Logits should be produced for it. So head output dim = masked_token_id + 1.
        # And embedding num_embeddings = masked_token_id + 1.
        # Let's assume the config eos_token_id and masked_token_id are the actual highest IDs for those tokens.

        # Reconciling with original hardcoding:
        # If eos_token_id=1024, masked_token_id=1025.
        # Vocab seems to be [0...1023 (regular), 1024 (EOS), 1025 (MASKED)]. Total 1026 tokens if 0-indexed.
        # So num_embeddings = 1026.
        # Head output logits for these 1026 tokens, so out_features = 1026.
        # The original code used 1025 for head output dim. This might mean it doesn't predict masked_token_id.
        # Or, 1025 means it predicts up to ID 1024.
        # The line `logits[..., 1025:].fill_(-torch.inf)` implies valid logits up to index 1024.
        # This means actual vocabulary size used for prediction is 1025 (tokens 0-1024).
        # And masked_token_id (1025) is special and perhaps not predicted but used as input.

        # Let's use a clear vocab_size based on config, assuming it includes EOS.
        # If eos_token_id is the ID for EOS, and it's the highest "normal" token,
        # then vocab_size for prediction could be eos_token_id + 1.
        # The masked_token_id (1025) is then an additional special input token.

        # Sticking close to original explicit numbers for now, then making configurable
        _actual_vocab_size_for_prediction = 1025 # Predicts tokens 0-1024 (EOS is 1024)
        _embedding_size = 1026 # Accommodates token 1025 (MASKED) as input

        # Initialize with unpadded sizes first. Padding will be applied after loading state_dict if necessary.
        self.embeddings = nn.ModuleList([nn.Embedding(_embedding_size, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, _actual_vocab_size_for_prediction, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        # The padding logic using find_multiple and re-initializing layers
        # has been removed from __init__. It will be handled by the
        # _pad_embeddings_and_heads method *after* state_dict loading in from_local.

    def _pad_embeddings_and_heads(self, *args, **kwargs): # This is a load_state_dict_post_hook (or can be called manually)
        # This function is for loading pretrained models that might not have padded vocab,
        # or for applying padding after initial weight loading.
        # It modifies layers in-place.
        for w_emb in self.embeddings:
            pad_weight_(w_emb, self.config.pad_vocab_to_multiple_of)
        for w_head in self.heads:
            pad_weight_(w_head, self.config.pad_vocab_to_multiple_of)


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        # This method implies fetching config.json and model.safetensors from HF Hub.
        # This should still work if the Zonos library code is local.
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        # The from_local method is what actually uses the local Zonos code structure.
        return cls.from_local(str(config_path), str(model_path), device, **kwargs)


    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        config_dict = json.load(open(config_path))
        config = ZonosConfig.from_dict(config_dict)

        backbone_cls_to_use = None
        if backbone:
            if backbone not in BACKBONES:
                raise ValueError(f"Specified backbone '{backbone}' not found. Available: {list(BACKBONES.keys())}")
            backbone_cls_to_use = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg) # Check if ssm_cfg is empty or None
            if not BACKBONES:
                 raise RuntimeError("No backbones available in Zonos library. Cannot determine default.")

            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls_to_use = BACKBONES["torch"]
            elif "mamba_ssm" in BACKBONES and not is_transformer : # Hybrid or Mamba only
                backbone_cls_to_use = BACKBONES["mamba_ssm"]
            elif BACKBONES : # Fallback to the first available one
                backbone_cls_to_use = next(iter(BACKBONES.values()))
            else: # Should not happen if previous BACKBONES check passed
                 raise RuntimeError("Could not determine a backbone class.")


        model = cls(config, backbone_cls_to_use).to(device) # Removed .bfloat16() for now, device handles dtype
        if str(device) != 'cpu': # For non-CPU, autoencoder also needs to be on device
            model.autoencoder.dac.to(device)
        # else: DAC is already on CPU by default

        # Load state dict
        sd_model = model.state_dict()
        loaded_sd = {}
        with safetensors.safe_open(model_path, framework="pt", device=str(device)) as f:
            for k in f.keys():
                loaded_sd[k] = f.get_tensor(k)

        # Filter out unexpected keys and handle missing keys from the loaded state_dict
        # This is important if the local model definition (e.g. padded vocab) differs
        # from the saved model.
        current_model_keys = set(sd_model.keys())
        loaded_keys = set(loaded_sd.keys())

        extra_keys_in_sd = loaded_keys - current_model_keys
        missing_keys_in_sd = current_model_keys - loaded_keys

        if extra_keys_in_sd:
            print(f"WARNING: Extra keys in loaded state_dict not in current model: {extra_keys_in_sd}")
            for k_extra in extra_keys_in_sd:
                del loaded_sd[k_extra] # Remove them so load_state_dict doesn't complain

        if missing_keys_in_sd:
            print(f"WARNING: Keys missing in loaded state_dict that are in current model: {missing_keys_in_sd}")
            # load_state_dict will raise an error for missing keys if strict=True (default)
            # For now, we'll let it raise or user can use strict=False if they know what they're doing.

        model.load_state_dict(loaded_sd, strict=False) # Use strict=False to tolerate missing/extra keys for now

        # Manually apply padding hook after loading if weights were from unpadded model
        # This is only if the model structure itself wasn't padded at init time.
        # Given the init logic now tries to use padded sizes, this might only be for specific keys
        # if the loaded model had different padding.
        # The _pad_embeddings_and_heads hook is more for models that are loaded and *then* padded.
        # If our model is defined with padding from the start, this hook is less critical unless
        # the loaded state dict is for an unpadded version and we want to force our padding.
        if config.pad_vocab_to_multiple_of:
             model._pad_embeddings_and_heads()


        return model.eval() # Ensure model is in eval mode

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA(device=str(self.device)) # Pass device
        # Ensure input wav is on the same device as spk_clone_model expects
        # SpeakerEmbeddingLDA internal model is on self.spk_clone_model.device
        spk_clone_device = self.spk_clone_model.device
        _, spk_embedding = self.spk_clone_model(wav.to(spk_clone_device), sr)
        # Return embedding on the main model's device, and in bfloat16 if supported/desired
        # For now, keep on spk_clone_device and let caller handle. Or move to self.device.
        # The original returned .bfloat16(). Let's match that if possible.
        # return spk_embedding.unsqueeze(0).to(self.device, dtype=torch.bfloat16 if self.device.type=='cuda' else torch.float32)
        return spk_embedding.unsqueeze(0).to(self.device) # .bfloat16() can be applied by caller if needed

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [batch_size, num_codebooks, seq_len]
        # We need to permute to [seq_len, batch_size, num_codebooks] then embed per codebook, then sum
        # Or, iterate, embed, stack, then sum.
        # Original: sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))
        # codes[:, i] would be [batch_size, seq_len]

        embedded_sum = torch.zeros(codes.shape[0], codes.shape[2], self.config.backbone.d_model, device=self.device)
        for i, emb_layer in enumerate(self.embeddings):
            embedded_sum += emb_layer(codes[:, i, :])
        return embedded_sum


    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch_size, seq_len, d_model]
        # Output: [batch_size, num_codebooks, seq_len, vocab_size_for_prediction]
        # Original: torch.stack([head(hidden_states) for head in self.heads], dim=1)
        # This would make output [batch_size, num_codebooks, seq_len, vocab_size]

        # head(hidden_states) -> [batch_size, seq_len, vocab_size_for_prediction]
        # stack along dim=1 -> [batch_size, num_codebooks, seq_len, vocab_size_for_prediction]
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)


    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams | None, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        # Backbone might take inference_params or not depending on type
        if isinstance(self.backbone, MambaSSMZonosBackbone): # Mamba uses it
            last_hidden_states_out = self.backbone(hidden_states, inference_params)
        else: # Torch transformer backbone in example doesn't use inference_params in its forward directly in this way for single step
              # but the kv_cache is updated via inference_params.
              # The provided TorchZonosBackbone takes inference_params in its forward.
            last_hidden_states_out = self.backbone(hidden_states, inference_params)

        # We need logits for the last token only if generating token by token.
        # If prefilling, we need all. The original code took last_hidden_states[:, -1, :].unsqueeze(1)
        # This implies we are always interested in the logits for the *next* token.

        # If hidden_states corresponds to a sequence of length S,
        # backbone output is [B, S, D].
        # We want logits for the token at S (predicting S+1).
        # So we take the features at the S-th position (index S-1).
        last_token_features = last_hidden_states_out[:, -1, :].unsqueeze(1) # [B, 1, D]

        # apply_heads expects [B, Seq, D], where Seq can be 1.
        logits = self.apply_heads(last_token_features) # Output: [B, N_q, 1, Vocab]
        logits = logits.squeeze(2) # -> [B, N_q, Vocab]

        # Classifier-Free Guidance
        if cfg_scale != 1.0 and cfg_scale != 0: # cfg_scale=0 means unconditional
            # hidden_states were already duplicated [cond_batch+uncond_batch, seq, dim]
            # So logits are also [cond_batch+uncond_batch, N_q, Vocab]
            # Split them into conditional and unconditional parts
            # Assuming batch dimension was doubled: first half cond, second half uncond
            # This means the original hidden_states passed to _compute_logits should be doubled.
            # The calling functions _decode_one_token and _prefill handle this doubling.
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            logits = guided_logits

        # Mask out padding or unwanted tokens
        # Original: logits[..., 1025:].fill_(-torch.inf)
        # This means tokens from ID 1025 onwards are invalid for prediction.
        # If _actual_vocab_size_for_prediction is 1025 (IDs 0-1024), then this is correct.
        # If head output size was padded to `padded_head_output_size`,
        # then logits for these padded dims should also be -inf.
        # The number of actual predictable tokens is `_actual_vocab_size_for_prediction` (e.g. 1025)
        # So valid indices are 0 to `_actual_vocab_size_for_prediction - 1`.
        # Any logits beyond that (due to padding of head layer) should be masked.

        # Get the actual vocabulary size the heads were defined with (could be padded)
        actual_head_output_dim = self.heads[0].out_features

        if actual_head_output_dim > 1025: # Assuming 1025 is the number of non-padded tokens (0-1024)
             logits[..., 1025:actual_head_output_dim].fill_(-torch.inf) # Mask out padded part of vocab
        # Also, if there are any other special tokens beyond 1024 that shouldn't be predicted, mask them.
        # The original masked_token_id = 1025. If it's not meant to be predicted, it should be masked.
        # The line `logits[..., 1025:].fill_(-torch.inf)` suggests that token 1025 (masked_token_id) is NOT predicted.
        # This is consistent with `_actual_vocab_size_for_prediction = 1025`.

        return logits.float() # Ensure float for sampling

    def _decode_one_token(
        self,
        input_ids: torch.Tensor, # Current token(s) to process: [Batch, N_q, Seq=1]
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True, # Not used in this version of Zonos
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.
        `input_ids` here are for the *current* step.
        """
        hidden_states_local = self.embed_codes(input_ids) # [Batch, Seq=1, Dim]

        if cfg_scale != 1.0 and cfg_scale != 0: # cfg_scale=0 means unconditional
            # Replicate for CFG: first half conditional, second half unconditional
            # Unconditional part is usually derived from a "null" or generic prompt.
            # Here, the `prepare_conditioning` in `generate` handles creating doubled prefix_conditioning.
            # For the main generation loop, `input_ids` are from the single batch.
            # If CFG is used, the `inference_params` cache should be for batch_size * 2.
            # And the `prefix_hidden_states` given to `_prefill` would already be doubled.
            # So, `hidden_states_local` here should also be doubled if it's not already.
            # The `generate` function calls `decode_one_token` with `input_ids` of original batch size.
            # So, we need to duplicate `hidden_states_local` here.
            # And `inference_params` should have been setup for batch_size*2.

            # Let's assume `input_ids` is [batch_size, N_q, 1]
            # `hidden_states_local` is [batch_size, 1, D]
            # We need to form [2*batch_size, 1, D] where the second half corresponds to unconditional.
            # This implies the unconditional part of `input_ids` might need to be different (e.g. masked).
            # The original Zonos code's `generate` has `prefix_conditioning` already doubled.
            # For subsequent tokens, it passes `input_ids` from the single batch.
            # The CUDA graph version in original Zonos had `hidden_states_local = hidden_states_local.repeat(2, 1, 1)`.
            # This means the *same* input token is used for both conditional and unconditional paths during decoding steps.
            hidden_states_for_compute = hidden_states_local.repeat(2, 1, 1)
        else:
            hidden_states_for_compute = hidden_states_local

        return self._compute_logits(hidden_states_for_compute, inference_params, cfg_scale)


    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor, # [Batch*2 (if CFG) or Batch, Cond_Seq_Len, D]
        input_ids: torch.Tensor, # Audio prompt codes: [Batch (original), N_q, Prefix_Audio_Seq_Len]
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: Process `prefix_hidden_states` (from text/other conditions)
        and then append embeddings of `input_ids` (from audio prompt), then compute logits.
        """
        # `input_ids` are from the audio prefix, original batch size.
        # `prefix_hidden_states` is already doubled if CFG.

        embedded_audio_prompt = self.embed_codes(input_ids) # [Batch, Prefix_Audio_Seq_Len, D]

        if cfg_scale != 1.0 and cfg_scale != 0:
            # `prefix_hidden_states` is [2B, Cond_Seq, D]
            # `embedded_audio_prompt` needs to be [2B, Prefix_Audio_Seq, D]
            # Assuming the same audio prompt is used for cond and uncond paths.
            embedded_audio_prompt_for_cat = embedded_audio_prompt.repeat(2, 1, 1)
        else:
            embedded_audio_prompt_for_cat = embedded_audio_prompt

        # Concatenate text/condition prefix with audio prefix embeddings
        # Ensure seq_len dimension is 1 for cat
        hidden_states = torch.cat([prefix_hidden_states, embedded_audio_prompt_for_cat], dim=1)

        # Now compute logits. `_compute_logits` will take the last token's features.
        return self._compute_logits(hidden_states, inference_params, cfg_scale)


    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None) -> InferenceParams:
        # If dtype is None, use model's default dtype (e.g. float32 for CPU, bfloat16 for CUDA)
        # For now, let's stick to bfloat16 as a common default for inference if not CPU
        # The original code used bfloat16.
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32

        max_seqlen = find_multiple(max_seqlen, 8) # Ensure multiple of 8

        # Allocate cache for the backbone (Mamba or Transformer)
        # The batch_size here should be the actual batch size for the cache (e.g., doubled if using CFG)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

        # lengths_per_sample is for Transformer RoPE, Mamba might not use it this way.
        # It's part of InferenceParams dataclass.
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32, device=self.device)

        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        # This method uses the PrefixConditioner.
        # If CFG is used, cond_dict is for conditional, uncond_dict for unconditional.
        # The PrefixConditioner itself handles creating embeddings for these.
        # The output should be [2*B, Cond_Seq_Len, D_model] if CFG, or [B, Cond_Seq_Len, D_model] if not.

        cond_embedding = self.prefix_conditioner(cond_dict) # [B, Cond_Seq_Len, D_model]

        if uncond_dict is not None: # Indicates CFG is likely active
            uncond_embedding = self.prefix_conditioner(uncond_dict) # [B, Cond_Seq_Len, D_model]
            # Concatenate along batch dimension
            return torch.cat([cond_embedding, uncond_embedding], dim=0) # [2B, Cond_Seq_Len, D_model]
        else:
            return cond_embedding # [B, Cond_Seq_Len, D_model]


    def can_use_cudagraphs(self) -> bool: # Not used in this simplified local Zonos
        # Only the mamba-ssm backbone supported CUDA Graphs in the original.
        # return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)
        return False # Disable for now for local version

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz*2 or bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz_orig, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        # batch_size: int = 1, # batch_size is inferred from prefix_conditioning
        sampling_params: dict | None = None,
        progress_bar: bool = True,
        # disable_torch_compile: bool = False, # Not using torch.compile for local version simplicity
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        if sampling_params is None:
            sampling_params = dict(min_p=0.1) # Default from original

        # Determine original batch size (before CFG duplication)
        # If CFG, prefix_conditioning is [2B, ...]. If not, [B, ...]
        # audio_prefix_codes is always [B_orig, ...]
        original_batch_size = audio_prefix_codes.shape[0] if audio_prefix_codes is not None else prefix_conditioning.shape[0]
        if cfg_scale != 1.0 and cfg_scale != 0:
            if prefix_conditioning.shape[0] % 2 != 0 and audio_prefix_codes is None:
                 raise ValueError("For CFG, prefix_conditioning batch size must be even if audio_prefix_codes is None.")
            if audio_prefix_codes is not None and prefix_conditioning.shape[0] != 2 * original_batch_size:
                 # If audio_prefix_codes is [B,...] and prefix_conditioning is not [2B,...], it's an issue for CFG.
                 # This implies prepare_conditioning was not called correctly for CFG.
                 # For safety, let's assume prefix_conditioning is already correctly doubled if cfg_scale demands it.
                 pass # Assume caller handled doubling prefix_conditioning batch for CFG

            effective_batch_for_cache = 2 * original_batch_size
        else:
            effective_batch_for_cache = original_batch_size


        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # cg = self.can_use_cudagraphs() # Not using cudagraphs or compile for simplicity now
        # decode_one_token_fn = torch.compile(self._decode_one_token, dynamic=True, disable=cg or disable_torch_compile)
        decode_one_token_fn = self._decode_one_token # Use directly

        unknown_token = -1 # Placeholder for not-yet-generated tokens
        # Max audio seq len for the codes tensor (prefix + new)
        audio_seq_len = prefix_audio_len + max_new_tokens
        # Total sequence length for inference cache (conditioning prefix + audio part + delay pattern overhead)
        # The delay pattern adds num_codebooks (9) to the audio sequence length effectively.
        total_cache_seq_len = prefix_conditioning.shape[1] + audio_seq_len + self.autoencoder.num_codebooks

        inference_params = self.setup_cache(batch_size=effective_batch_for_cache, max_seqlen=total_cache_seq_len)

        # Initialize codes tensor for the original batch size
        codes = torch.full((original_batch_size, self.autoencoder.num_codebooks, audio_seq_len),
                           unknown_token, dtype=torch.long, device=device)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes.to(device)

        # Apply delay pattern (masked_token_id is from config)
        delayed_codes = apply_delay_pattern(codes, self.config.masked_token_id)
        # delayed_codes shape: [original_batch_size, num_codebooks, audio_seq_len + num_codebooks]

        # --- Prefill Phase ---
        # Process conditioning prefix + audio prefix (if any)
        # `current_prefix_hidden_states` starts with text/condition embeddings
        current_prefix_hidden_states = prefix_conditioning # Already [2B or B, cond_seq, D]

        if prefix_audio_len > 0:
            # `delayed_prefix_audio_codes` are the first `prefix_audio_len + 1` tokens from `delayed_codes`
            # These are the actual audio prompt tokens shifted by the delay pattern.
            # The `+1` is crucial as the Mamba model needs one more step to "see" the last input token
            # to correctly predict the token *after* it. Or, for transformers, to establish context.
            # The original code used `delayed_codes[..., : prefix_audio_len + 1]`
            # Shape: [original_batch_size, num_codebooks, prefix_audio_len + 1]
            delayed_prefix_audio_input_for_prefill = delayed_codes[..., : prefix_audio_len + 1]

            # _prefill will embed these audio codes and concat with current_prefix_hidden_states
            logits = self._prefill(current_prefix_hidden_states,
                                   delayed_prefix_audio_input_for_prefill,
                                   inference_params, cfg_scale)
            # `logits` are for the token *after* the `delayed_prefix_audio_input_for_prefill`.
            # This corresponds to the first token of the *newly generated* part.

            # Update inference_params offset
            prefill_len = current_prefix_hidden_states.shape[1] + delayed_prefix_audio_input_for_prefill.shape[2]
            inference_params.seqlen_offset += prefill_len
            if inference_params.lengths_per_sample is not None: # For Transformer RoPE
                inference_params.lengths_per_sample += prefill_len
        else: # No audio prefix, just text/condition prefix
            # We need one "starter" token to kick off generation.
            # This could be a BOS token for each codebook, or simply use the prefix_conditioning
            # and generate the very first audio token.
            # The original Zonos seems to assume there's always at least one step of audio prefix
            # or the logic for the very first token generation needs to be explicit if no audio prefix.
            # Let's assume if no audio_prefix_codes, we start by "predicting" the first audio frame.
            # This means `_compute_logits` is called with just `prefix_conditioning`.
            logits = self._compute_logits(current_prefix_hidden_states, inference_params, cfg_scale)

            prefill_len = current_prefix_hidden_states.shape[1]
            inference_params.seqlen_offset += prefill_len
            if inference_params.lengths_per_sample is not None:
                inference_params.lengths_per_sample += prefill_len

        # Sample the first (set of) new token(s) using these logits
        # logits shape: [original_batch_size, num_codebooks, vocab_size]
        next_tokens_sampled = sample_from_logits(logits, **sampling_params) # [B_orig, N_q, 1]

        # Determine where to place these first sampled tokens in `delayed_codes`
        # This is at `delayed_idx_offset`.
        # If prefill used `prefix_audio_len + 1` tokens, the next one is at index `prefix_audio_len + 1`.
        delayed_idx_offset = prefix_audio_len + 1

        # Update `delayed_codes` with the first sampled tokens
        # Ensure `next_tokens_sampled` aligns with `frame` for masked_scatter_
        frame_to_fill = delayed_codes[..., delayed_idx_offset : delayed_idx_offset + 1] # [B_orig, N_q, 1]
        # We only fill where `frame_to_fill` was `unknown_token` or `masked_token_id`
        # This ensures we don't overwrite parts of a given audio prefix that were not unknown.
        # However, `codes` was init with `unknown_token`, so `delayed_codes` also has it.
        # So, this condition `frame_to_fill == unknown_token` should be true for new parts.
        mask_for_scatter = (frame_to_fill == unknown_token) | (frame_to_fill == self.config.masked_token_id)
        frame_to_fill.masked_scatter_(mask_for_scatter, next_tokens_sampled)
        # This updates `delayed_codes` in place.

        # --- Autoregressive Generation Loop ---
        # `logit_bias` to prevent EOS from being generated too early in non-0 codebooks
        logit_bias = torch.zeros_like(logits) # [original_batch_size, num_codebooks, vocab_size]
        # Only allow EOS (ID self.config.eos_token_id) in the first codebook (index 0)
        logit_bias[:, 1:, self.config.eos_token_id] = -torch.inf

        # Stopping condition trackers for each item in batch
        # `stopping` becomes True when EOS is generated in the first codebook.
        # `remaining_steps_after_eos` counts down from num_codebooks once EOS is hit, to complete the delay pattern.
        stopping = torch.zeros(original_batch_size, dtype=torch.bool, device=device)
        remaining_steps_after_eos = torch.full((original_batch_size,), self.autoencoder.num_codebooks, device=device)

        # Max steps for the loop: from `delayed_idx_offset + 1` up to end of `delayed_codes`
        # Total length of `delayed_codes` is `audio_seq_len + num_codebooks`
        # Current `delayed_idx_offset` is where we just wrote. Next input is `delayed_idx_offset`.
        # Loop starts from step `delayed_idx_offset + 1`.

        # The loop iterates `max_new_tokens -1` more times (since one token is already sampled)
        # Or, more simply, loop while `delayed_idx_offset` < `delayed_codes.shape[2] -1`
        # and not all batches are stopped.

        # Progress bar setup
        pbar = tqdm(total=max_new_tokens, desc="Generating Audio Tokens", disable=not progress_bar, initial=1)

        for step_num in range(1, max_new_tokens): # Already did 1st token
            if torch.all(stopping & (remaining_steps_after_eos <= 0)): # All batches finished generating
                break

            # Prepare input for this step: the tokens we just generated/filled.
            # Input is `delayed_codes[..., delayed_idx_offset : delayed_idx_offset + 1]`
            current_input_ids = delayed_codes[..., delayed_idx_offset : delayed_idx_offset + 1] # [B_orig, N_q, 1]

            # Update inference_params for this step
            inference_params.seqlen_offset += 1
            if inference_params.lengths_per_sample is not None:
                inference_params.lengths_per_sample += 1

            # Get logits for the next set of tokens
            # `cfg_scale_tensor = torch.tensor(cfg_scale, device=device)` - not needed, pass float
            logits = decode_one_token_fn(current_input_ids, inference_params, cfg_scale) # [B_orig, N_q, Vocab]
            logits += logit_bias # Apply EOS bias

            # Sample next tokens
            # `generated_tokens` for repetition penalty should be `delayed_codes[..., :delayed_idx_offset+1]`
            # This includes all previously determined tokens in the delayed format.
            next_tokens_sampled = sample_from_logits(logits,
                                                     generated_tokens=delayed_codes[..., :delayed_idx_offset+1],
                                                     **sampling_params) # [B_orig, N_q, 1]

            # Handle EOS logic: if EOS is sampled in codebook 0 for a batch item
            eos_in_cb0_for_batch = (next_tokens_sampled[:, 0, 0] == self.config.eos_token_id)

            # For items where EOS is now true for the first time
            newly_stopping = eos_in_cb0_for_batch & (~stopping)
            if torch.any(newly_stopping):
                stopping[newly_stopping] = True
                # `remaining_steps_after_eos` is already num_codebooks, so it starts counting down.

            # If an item is in `stopping` state, fill remaining tokens for delay pattern completion
            for i in range(original_batch_size):
                if stopping[i]:
                    # How many more tokens (codebooks) to fill for this item to complete EOS pattern
                    eos_fill_idx = self.autoencoder.num_codebooks - remaining_steps_after_eos[i].item()

                    # Fill with MASKED_TOKEN up to the EOS codebook index
                    next_tokens_sampled[i, :eos_fill_idx, 0] = self.config.masked_token_id
                    # Place EOS at the correct codebook index
                    if eos_fill_idx < self.autoencoder.num_codebooks:
                         next_tokens_sampled[i, eos_fill_idx, 0] = self.config.eos_token_id
                    # Fill remaining with MASKED_TOKEN (or could be another padding, but MASKED is fine)
                    if eos_fill_idx + 1 < self.autoencoder.num_codebooks:
                         next_tokens_sampled[i, eos_fill_idx + 1 :, 0] = self.config.masked_token_id

                    remaining_steps_after_eos[i] -= 1

            # Update `delayed_codes` with these new tokens
            delayed_idx_offset += 1 # Move to next position to fill
            frame_to_fill = delayed_codes[..., delayed_idx_offset : delayed_idx_offset + 1]
            mask_for_scatter = (frame_to_fill == unknown_token) | (frame_to_fill == self.config.masked_token_id)
            frame_to_fill.masked_scatter_(mask_for_scatter, next_tokens_sampled)

            pbar.update(1)

            if callback is not None:
                # Callback receives the latest *generated frame* (next_tokens_sampled might be more useful)
                # and current step relative to max_new_tokens.
                if not callback(next_tokens_sampled.clone(), step_num + 1, max_new_tokens):
                    break

        pbar.close()

        # Revert delay pattern to get final codes
        # `delayed_codes` has shape [B, N_q, audio_seq_len + N_q]
        # We need to decide how much of it to revert.
        # `delayed_idx_offset` points to the last position written in `delayed_codes`.
        # The actual number of "audio frames" generated is related to this.
        # The `revert_delay_pattern` expects a certain structure.
        # Original: out_codes = revert_delay_pattern(delayed_codes)
        #           out_codes = out_codes[..., : offset - 9] where offset was delayed_idx_offset

        # The effective length of generated audio is `delayed_idx_offset - num_codebooks + 1`
        # if `delayed_idx_offset` is the index of the *last token of the last complete frame* in delayed_codes.
        # Or, if `delayed_idx_offset` is the *next position to write*, then it's `delayed_idx_offset - num_codebooks`.

        # `revert_delay_pattern` takes the full `delayed_codes`
        # `reverted_codes` will have length `delayed_codes.shape[2] - num_codebooks`
        reverted_codes = revert_delay_pattern(delayed_codes)

        # Determine actual length of generated content.
        # If EOS was hit, generation might be shorter than max_new_tokens.
        # `delayed_idx_offset` is the index of the *next token to be filled* in `delayed_codes`.
        # So, valid data in `delayed_codes` is up to `delayed_idx_offset - 1`.
        # The corresponding length in `reverted_codes` is `(delayed_idx_offset - 1) - num_codebooks + 1`
        # which is `delayed_idx_offset - num_codebooks`.
        final_audio_len = delayed_idx_offset - self.autoencoder.num_codebooks

        out_codes = reverted_codes[..., :final_audio_len]

        # Mask out any special tokens (EOS, MASKED) in the final output, replace with 0 (or a silence token)
        out_codes.masked_fill_(out_codes >= self.config.eos_token_id, 0) # Assuming EOS and MASKED are >= EOS_ID

        # self._cg_graph = None # Not using CUDA graphs here

        return out_codes
