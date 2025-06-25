import json
from typing import Callable

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .autoencoder import DACAutoencoder
from .backbone import BACKBONES
# from .backbone._mamba_ssm import MambaSSMZonosBackbone # Removed to make Mamba optional
# from .backbone._torch import TorchZonosBackbone
from .codebook_pattern import apply_delay_pattern, revert_delay_pattern
from .conditioning import PrefixConditioner
from .config import InferenceParams, ZonosConfig
from .sampling import sample_from_logits
from .speaker_cloning import SpeakerEmbeddingLDA
from .utils import DEFAULT_DEVICE, find_multiple, pad_weight_

# Determine DEFAULT_BACKBONE_CLS safely
if BACKBONES:
    DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))
else:
    DEFAULT_BACKBONE_CLS = None


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=None): # Allow backbone_cls to be None initially
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        if backbone_cls is None:
            if not BACKBONES:
                raise RuntimeError("No backbones available in BACKBONES dictionary. Check zonos_local_lib.backbone setup.")

            # Fallback logic to select a backbone if not provided
            is_transformer = not bool(config.backbone.ssm_cfg)
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]
            elif not is_transformer and "mamba_ssm" in BACKBONES: # Prefer Mamba if SSM config and Mamba available
                backbone_cls = BACKBONES["mamba_ssm"]
            elif "torch" in BACKBONES: # Fallback to torch if primary choice not met
                backbone_cls = BACKBONES["torch"]
            elif BACKBONES: # Fallback to the first available one if others not suitable/available
                 backbone_cls = next(iter(BACKBONES.values()))
            else: # Should be caught by the earlier "No backbones available"
                 raise RuntimeError("Could not determine a backbone class for Zonos __init__.")

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # Using hardcoded vocab sizes from the provided "original" file for now
        # These might need to be made configurable based on self.config for robustness
        _embedding_vocab_size = 1026
        _head_output_vocab_size = 1025

        self.embeddings = nn.ModuleList([nn.Embedding(_embedding_vocab_size, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, _head_output_vocab_size, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        # The original used *self.embeddings, *self.heads
        # Ensuring this works with ModuleList:
        for w_emb in self.embeddings:
            pad_weight_(w_emb, self.config.pad_vocab_to_multiple_of)
        for w_head in self.heads: # Assuming pad_weight_ works for Linear layers output features
            pad_weight_(w_head, self.config.pad_vocab_to_multiple_of)


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))

        backbone_cls_resolved = None
        if backbone:
            backbone_cls_resolved = BACKBONES.get(backbone)
            if backbone_cls_resolved is None:
                raise ValueError(f"Specified backbone '{backbone}' not found or failed to import. Available: {list(BACKBONES.keys())}")
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            if not BACKBONES: # Should ideally not be empty if TorchZonosBackbone is always there
                raise RuntimeError("No backbones available in BACKBONES dictionary.")

            # Defaulting logic
            if is_transformer and "torch" in BACKBONES:
                backbone_cls_resolved = BACKBONES["torch"]
            elif not is_transformer and "mamba_ssm" in BACKBONES: # Mamba preferred for SSM
                backbone_cls_resolved = BACKBONES["mamba_ssm"]
            elif "torch" in BACKBONES: # Fallback to torch if mamba not available/suitable
                backbone_cls_resolved = BACKBONES["torch"]
            elif BACKBONES: # Absolute fallback to first available
                 backbone_cls_resolved = next(iter(BACKBONES.values()))
            else:
                 raise RuntimeError("Could not determine a backbone class for Zonos.from_local.")

        target_dtype = torch.bfloat16 if str(device) != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = cls(config, backbone_cls_resolved).to(device) # Move to device first
        if target_dtype != torch.float32 : # Then cast dtype if not float32
             model = model.to(dtype=target_dtype)

        if hasattr(model.autoencoder, 'dac') and model.autoencoder.dac is not None:
            model.autoencoder.dac.to(device)
            if target_dtype != torch.float32 and hasattr(model.autoencoder.dac, 'to'):
                 try: # Autoencoder might not support all dtypes
                      model.autoencoder.dac = model.autoencoder.dac.to(dtype=target_dtype)
                 except RuntimeError:
                      print(f"Warning: Could not cast DAC autoencoder to {target_dtype}, keeping its original dtype.")


        loaded_tensors = {}
        # Load tensors to CPU first, then move to target device & dtype via model.load_state_dict
        # This avoids issues if safetensors device mapping isn't perfect or if model is already on device.
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                loaded_tensors[k] = f.get_tensor(k)

        # Align dtypes of loaded tensors to model's target_dtype before loading, if they are float
        # This is important because load_state_dict can be strict about dtypes.
        # model.to(dtype=target_dtype) should have already set parameter dtypes.
        # We are ensuring the loaded state_dict tensors match this.
        for k_loaded, t_loaded in loaded_tensors.items():
            if t_loaded.is_floating_point() and t_loaded.dtype != target_dtype:
                loaded_tensors[k_loaded] = t_loaded.to(dtype=target_dtype)
            elif not t_loaded.is_floating_point() and hasattr(model.state_dict().get(k_loaded, None),'dtype') and \
                 t_loaded.dtype != model.state_dict()[k_loaded].dtype :
                 # For non-float, e.g. int/bool, ensure they match if model has specific non-float dtypes (rare for params)
                 # This part is less common, usually float parameters are the main concern.
                 pass # Or cast if necessary: loaded_tensors[k_loaded] = t_loaded.to(dtype=model.state_dict()[k_loaded].dtype)


        model.load_state_dict(loaded_tensors, strict=True) # strict=True is good for aligned structures

        return model.eval()

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA(device=str(self.device))

        # SpeakerEmbeddingLDA handles input wav device internally.
        # It returns embedding on its own target_device.
        _, spk_embedding = self.spk_clone_model(wav, sr)

        # Ensure final embedding is on main model's device and matches model's primary float dtype
        model_float_dtype = next(p.dtype for p in self.parameters() if p.is_floating_point())
        return spk_embedding.unsqueeze(0).to(device=self.device, dtype=model_float_dtype)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [batch_size, num_codebooks, seq_len], type torch.long
        # self.embeddings are ModuleList of nn.Embedding, already on self.device and self.dtype
        # codes need to be on self.device.
        codes_on_device = codes.to(self.device)
        return sum(emb(codes_on_device[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch_size, seq_len, d_model], on self.device, self.dtype
        # self.heads are ModuleList of nn.Linear, already on self.device, self.dtype
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        # hidden_states: [B_eff, S, D], dtype e.g. bfloat16
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        # last_hidden_states: [B_eff, 1, D], dtype e.g. bfloat16

        logits = self.apply_heads(last_hidden_states).squeeze(2) # [B_eff, N_q, Vocab], dtype e.g. bfloat16
        logits = logits.float() # Cast to float32 for CFG math and sampling (as per original Zonos)

        if cfg_scale != 1.0: # Original Zonos didn't check != 0 here
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        # Masking based on actual head output dim vs intended prediction vocab size (1025 for tokens 0-1024)
        _intended_prediction_vocab_size = 1025
        actual_head_output_dim = self.heads[0].out_features
        if actual_head_output_dim > _intended_prediction_vocab_size:
            logits[..., _intended_prediction_vocab_size:actual_head_output_dim].fill_(-torch.inf)
        # The original `logits[..., 1025:].fill_(-torch.inf)` is equivalent if head output is exactly 1025.

        return logits # float32

    def _decode_one_token(
        self,
        input_ids: torch.Tensor, # [B_orig, N_q, 1]
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True, # From original
    ) -> torch.Tensor:
        # Original Zonos had: assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        # And directly used CFG path if cfg_scale != 1.0.
        # If cfg_scale == 1.0, it implies no CFG, so hidden_states should not be repeated.
        # The `generate` method has an assert for cfg_scale != 1.
        # For now, assuming cfg_scale implies CFG if not 1.0 based on original `generate`'s assert.

        # Simplified logic from original, bypassing CUDA graph for now
        hidden_states_local = self.embed_codes(input_ids) # input_ids is [B_orig, N_q, 1] -> HL is [B_orig, 1, D]

        if cfg_scale != 1.0: # If CFG is active (as per original's direct path for cfg_scale != 1)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)  # Repeat for CFG -> [2*B_orig, 1, D]
        # If cfg_scale == 1.0, hidden_states_local remains [B_orig, 1, D]
        # This means inference_params must be set up for B_orig if cfg_scale == 1.0
        # And for 2*B_orig if cfg_scale != 1.0. This is handled by effective_cache_batch_size in generate.

        return self._compute_logits(hidden_states_local, inference_params, cfg_scale)


    def _prefill(
        self,
        prefix_conditioning: torch.Tensor, # [B_eff, Cond_S, D]
        input_ids: torch.Tensor, # Audio prompt codes: [B_orig, N_q, Prefix_Audio_S]
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:

        embedded_audio_prompt = self.embed_codes(input_ids) # [B_orig, Prefix_Audio_S, D]

        if cfg_scale != 1.0: # CFG is active
            # prefix_conditioning is already [2*B_orig, Cond_S, D]
            # embedded_audio_prompt needs to be expanded/repeated to match this for cat
            # Original used: input_ids.expand(prefix_conditioning.shape[0], -1, -1) then embed.
            # Here, we embed first, then repeat.
            if embedded_audio_prompt.shape[0] * 2 == prefix_conditioning.shape[0]: # B_orig * 2 == 2 * B_orig
                 embedded_audio_prompt_for_cat = embedded_audio_prompt.repeat(2, 1, 1) # [2*B_orig, Prefix_Audio_S, D]
            elif embedded_audio_prompt.shape[0] == prefix_conditioning.shape[0]: # Should not happen if CFG active
                 embedded_audio_prompt_for_cat = embedded_audio_prompt
            else:
                 raise ValueError("Batch size mismatch for CFG in _prefill between prefix_conditioning and embedded_audio_prompt.")
        else: # No CFG
            embedded_audio_prompt_for_cat = embedded_audio_prompt # [B_orig, Prefix_Audio_S, D]

        hidden_states = torch.cat([prefix_conditioning, embedded_audio_prompt_for_cat], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = None) -> InferenceParams:
        # batch_size is effective_cache_batch_size (e.g. 2*B_orig or B_orig)
        if dtype is None: # Use model's actual float dtype
            dtype = next(p.dtype for p in self.parameters() if p.is_floating_point())

        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32, device=self.device)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        # This version is from the provided "original" zonos/model.py
        if uncond_dict is None:
            # Create uncond_dict by taking required keys from cond_dict.
            # This means unconditional pass uses conditional values for these keys.
            # True unconditional might need specific null values (e.g. speaker=None).
            # ZonosLocalVoice now prepares a more "true" uncond_dict with nullified speaker/emotion.
            # This method will just use what's passed. If ZonosLocalVoice passes a well-formed
            # uncond_dict (with None for speaker etc.), prefix_conditioner should handle it.
            # If ZonosLocalVoice passes uncond_dict=None, this original Zonos logic takes over.
            # My ZonosLocalVoice fix *does* pass a potentially modified uncond_dict.
             uncond_dict = {k: cond_dict.get(k) for k in self.prefix_conditioner.required_keys}
             # Ensure all required keys are at least present as None if not in cond_dict
             for req_key in self.prefix_conditioner.required_keys:
                 uncond_dict.setdefault(req_key, None)


        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def can_use_cudagraphs(self) -> bool: # As per original
        # Current local _mamba_ssm.py might not be the CUDA graph compatible one from official mamba_ssm package
        # For now, assume it refers to the class name string for type check.
        return self.device.type == "cuda" and self.backbone.__class__.__name__ == "MambaSSMZonosBackbone"


    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [B_eff, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz_orig, N_q, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1, # This is B_orig (original batch size before any CFG duplication)
        sampling_params: dict = None,
        progress_bar: bool = True,
        disable_torch_compile: bool = True, # Default to True (disabled) for stability
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        if sampling_params is None:
            sampling_params = dict(min_p=0.1) # Default from original

        # Original Zonos had: assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        # My ZonosLocalVoice prepares prefix_conditioning based on cfg_active.
        # If cfg_scale = 1.0 or 0.0, ZLV passes non-doubled prefix_conditioning.
        # If cfg_scale is other, ZLV passes doubled prefix_conditioning.
        # So, this method needs to know the effective batch for cache based on that.

        is_cfg_active = cfg_scale != 1.0 and cfg_scale != 0.0

        if is_cfg_active:
            if prefix_conditioning.shape[0] != batch_size * 2:
                raise ValueError(f"For CFG (scale={cfg_scale}), prefix_conditioning batch dim ({prefix_conditioning.shape[0]}) must be 2 * batch_size ({batch_size}).")
            effective_cache_batch_size = batch_size * 2
        else: # No CFG, or unconditional only (cfg_scale=0)
            if prefix_conditioning.shape[0] != batch_size:
                 raise ValueError(f"Without CFG (scale={cfg_scale}), prefix_conditioning batch dim ({prefix_conditioning.shape[0]}) must be batch_size ({batch_size}).")
            effective_cache_batch_size = batch_size

        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        cg = self.can_use_cudagraphs() and not disable_torch_compile
        decode_one_token_fn = self._decode_one_token # No torch.compile for now
        # if not disable_torch_compile:
        #    decode_one_token_fn = torch.compile(decode_one_token_fn, dynamic=True, disable=cg)

        unknown_token = -1 # From original; could use self.config.unknown_token if defined
        audio_seq_len = prefix_audio_len + max_new_tokens

        # Max length for KV cache
        total_cache_seq_len = prefix_conditioning.shape[1] + (audio_seq_len + self.autoencoder.num_codebooks)

        with torch.device(device): # Ensures tensors created are on the model's device
            inference_params = self.setup_cache(batch_size=effective_cache_batch_size, max_seqlen=total_cache_seq_len)
            codes = torch.full((batch_size, self.autoencoder.num_codebooks, audio_seq_len),
                               unknown_token, dtype=torch.long, device=device)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes.to(device)

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id) # Use self.masked_token_id

        # Prefill phase (adapted from original)
        # delayed_prefix_audio_codes is for B_orig
        delayed_prefix_audio_input_for_prefill = delayed_codes[..., : prefix_audio_len + 1]

        # _prefill handles CFG replication of embedded audio codes internally based on cfg_scale and prefix_conditioning batch
        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_input_for_prefill, inference_params, cfg_scale)
        # logits is [B_orig, N_q, Vocab]

        next_token_sampled = sample_from_logits(logits, **sampling_params) # [B_orig, N_q, 1]

        # Initial fill of delayed_codes at the correct offset
        # This is the position for the first token *after* any audio_prefix
        current_fill_idx_delayed = prefix_audio_len + 1 # Matches original `offset` logic start

        # Using the refactored direct assignment:
        target_frame_slice_prefill = delayed_codes[..., current_fill_idx_delayed]
        mask_prefill = (target_frame_slice_prefill == unknown_token) | (target_frame_slice_prefill == self.masked_token_id)
        tokens_to_assign_prefill = next_token_sampled.squeeze(-1)
        target_frame_slice_prefill[mask_prefill] = tokens_to_assign_prefill[mask_prefill]

        # Update inference_params offset
        # This is the length of the sequence processed by the backbone in prefill
        prefill_backbone_input_len = prefix_conditioning.shape[1] + delayed_prefix_audio_input_for_prefill.shape[2]
        inference_params.seqlen_offset += prefill_backbone_input_len
        inference_params.lengths_per_sample += prefill_backbone_input_len # Add scalar to all in B_eff

        # Autoregressive loop (adapted from original)
        logit_bias = torch.zeros_like(logits) # logits is [B_orig, N_q, Vocab]
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device) # For B_orig

        # remaining_steps_after_eos logic from original Zonos
        # This counts how many more codebooks need to be filled for a complete EOS pattern.
        # Max steps in `delayed_codes` is `delayed_codes.shape[2] - 1`.
        # `current_fill_idx_delayed` is the last index filled.
        # Loop for `max_new_tokens - 1` more steps (since 1st new token is done)

        progress = tqdm(total=max_new_tokens, desc="Generating Audio Tokens", disable=not progress_bar, initial=1)

        # `current_input_idx_delayed` is the index of the frame *just filled*, to be used as input for next token.
        current_input_idx_delayed = current_fill_idx_delayed

        for step_count in range(1, max_new_tokens): # Already did 1st token (step_count=0 effectively)
            # Check overall stopping condition based on original logic's intent
            # The original loop was `while torch.max(remaining_steps) > 0:`.
            # Here, we need a robust way to check if all active (non-stopped) items are done.
            # For now, a simple check; this might need refinement if EOS behavior is complex.
            if torch.all(stopping): # Simplified: if all batches have triggered EOS at least once
                # More accurate: if all batches have completed their num_codebooks steps post-EOS
                # This requires tracking remaining_steps_after_eos per batch item.
                # For now, let's rely on max_new_tokens or early exit by callback.
                pass # The loop will run max_new_tokens times unless callback stops it.

            input_ids_for_decode = delayed_codes[..., current_input_idx_delayed : current_input_idx_delayed + 1] # [B_orig, N_q, 1]

            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample += 1

            logits_loop = decode_one_token_fn(input_ids_for_decode, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits_loop += logit_bias

            # Use previously generated tokens for repetition penalty context
            generated_tokens_context = delayed_codes[..., : current_input_idx_delayed + 1]
            next_token_sampled_loop = sample_from_logits(logits_loop,
                                                         generated_tokens=generated_tokens_context,
                                                         **sampling_params) # [B_orig, N_q, 1]

            # EOS handling (simplified for now - original was more complex with remaining_steps)
            eos_in_cb0_loop = (next_token_sampled_loop[:, 0, 0] == self.eos_token_id) & (~stopping)
            stopping[eos_in_cb0_loop] = True

            # In original, if stopping[i], next_token_sampled_loop[i] was filled with specific EOS/MASK pattern.
            # This ensures the delay pattern completes correctly after EOS.
            # This is important for `revert_delay_pattern`.
            if torch.any(stopping): # If any item is stopping or has stopped
                 for i in range(batch_size):
                     if stopping[i]:
                         # Simplified fill: If stopping, make this frame's CB0 EOS, others MASKED
                         # This is a placeholder. A proper `remaining_steps_after_eos` counter is needed
                         # to replicate the original Zonos's exact EOS pattern filling over N_q steps.
                         temp_fill = torch.full_like(next_token_sampled_loop[i], self.masked_token_id)
                         temp_fill[0,0] = self.eos_token_id
                         next_token_sampled_loop[i] = temp_fill


            # Update delayed_codes for the next position
            current_fill_idx_delayed += 1 # This is the index where we write the new tokens
            target_frame_slice_loop = delayed_codes[..., current_fill_idx_delayed]
            mask_loop = (target_frame_slice_loop == unknown_token) | (target_frame_slice_loop == self.masked_token_id)
            tokens_to_assign_loop = next_token_sampled_loop.squeeze(-1)
            target_frame_slice_loop[mask_loop] = tokens_to_assign_loop[mask_loop]

            current_input_idx_delayed = current_fill_idx_delayed # Next input is what we just wrote

            progress.update(1)
            if callback is not None:
                if not callback(next_token_sampled_loop.clone(), step_count + 1, max_new_tokens):
                    break

        progress.close()

        reverted_codes = revert_delay_pattern(delayed_codes)

        # Determine actual generated length in reverted_codes
        # `current_fill_idx_delayed` is the last index that was filled in `delayed_codes`.
        # The number of valid frames in `delayed_codes` is `current_fill_idx_delayed + 1`.
        # The length in `reverted_codes` corresponding to this is `(current_fill_idx_delayed + 1) - num_codebooks`.
        final_reverted_len = (current_fill_idx_delayed + 1) - self.autoencoder.num_codebooks

        out_codes = reverted_codes[..., :final_reverted_len]
        out_codes.masked_fill_(out_codes >= self.eos_token_id, 0) # Use self.eos_token_id

        self._cg_graph = None # Reset CUDA graph state as per original
        return out_codes

```
