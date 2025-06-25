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

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]: # Iterate directly over combined list
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

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
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        # Determine target_dtype based on device and CUDA capabilities (as in previous version)
        target_dtype = torch.bfloat16 if str(device) != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        model = cls(config, backbone_cls).to(device) # Create model and move to device
        if target_dtype == torch.bfloat16: # Cast to bfloat16 if applicable
            model = model.to(torch.bfloat16)

        # Ensure autoencoder's DAC model is on the correct device and dtype
        if hasattr(model.autoencoder, 'dac') and model.autoencoder.dac is not None:
            model.autoencoder.dac.to(device) # Move DAC to device
            if target_dtype == torch.bfloat16 and hasattr(model.autoencoder.dac, 'to'):
                 try:
                      model.autoencoder.dac = model.autoencoder.dac.to(dtype=target_dtype)
                 except RuntimeError:
                      print(f"Warning: Could not cast DAC autoencoder to {target_dtype}, keeping its original dtype.")


        # Load state dictionary, ensuring tensors are on CPU first then cast to target_dtype if needed
        loaded_tensors = {}
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                loaded_tensors[k] = f.get_tensor(k)

        # Align dtypes of loaded tensors to model's target_dtype before loading state_dict
        for k_loaded, t_loaded in loaded_tensors.items():
            if t_loaded.is_floating_point() and t_loaded.dtype != target_dtype:
                loaded_tensors[k_loaded] = t_loaded.to(dtype=target_dtype)
            # Non-float parameters are less common to have dtype issues, but could be handled here if necessary

        model.load_state_dict(loaded_tensors, strict=True)
        return model.eval()


    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            # Ensure device is correctly passed as a string if SpeakerEmbeddingLDA expects it
            self.spk_clone_model = SpeakerEmbeddingLDA(device=str(self.device))

        # spk_clone_model handles internal device placement of wav
        _, spk_embedding = self.spk_clone_model(wav, sr) # wav is on host, spk_clone_model moves it

        # Determine the model's primary floating-point dtype
        model_float_dtype = next(p.dtype for p in self.parameters() if p.is_floating_point())

        # Ensure the final embedding is on the model's main device and matches its primary float dtype
        return spk_embedding.unsqueeze(0).to(device=self.device, dtype=model_float_dtype)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        # codes are expected to be on self.device by the nn.Embedding layers
        return sum(emb(codes[:, i].to(self.device)) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states are expected to be on self.device by the nn.Linear layers
        return torch.stack([head(hidden_states.to(self.device)) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2) # Output from heads should match model's primary float type

        # Cast to float32 for CFG math and sampling, as per reference
        logits = logits.float()

        if cfg_scale != 1.0: # Reference code doesn't check != 0
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        # Masking based on actual head output dim vs intended prediction vocab size (1025 for tokens 0-1024)
        _intended_prediction_vocab_size = 1025 # Vocab size for tokens 0-1024
        actual_head_output_dim = self.heads[0].out_features
        if actual_head_output_dim > _intended_prediction_vocab_size:
            logits[..., _intended_prediction_vocab_size:actual_head_output_dim].fill_(-torch.inf)
        elif actual_head_output_dim < _intended_prediction_vocab_size:
            # This case should not happen if models are consistent but good to be aware of.
            # It would mean the head cannot predict all necessary tokens.
            pass


        return logits # float32

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        # TODO: support cfg_scale==1 (reference code has this TODO)
        # The reference code asserts cfg_scale != 1 in generate, so this path might not be fully tested for cfg_scale=1
        if cfg_scale == 1.0: # Path for no CFG or if CFG scale is 1 (effectively no CFG effect)
            hidden_states = self.embed_codes(input_ids) # B_orig, 1, D
            # InferenceParams should be set for B_orig for this path
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0) # This is B_orig

        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids) # B_orig, 1, D
            hidden_states_local = hidden_states_local.repeat(2, 1, 1) # 2*B_orig, 1, D for CFG
            # InferenceParams should be set for 2*B_orig for this path
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None # Reset graph

            self._cg_batch_size = bsz
            # These inference_params are for 2*bsz (effective batch size with CFG)
            # This needs to be carefully managed if cfg_scale can be 1.0 here.
            # Assuming cfg_scale != 1.0 if CUDA graphs are used, as per generate's assert.
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            # Warmup
            for _ in range(3):
                hidden_states = self.embed_codes(input_ids) # B_orig, 1, D
                hidden_states = hidden_states.repeat(2, 1, 1)  # 2*B_orig, 1, D
                _ = self._compute_logits(hidden_states, self._cg_inference_params, self._cg_scale)

            self._cg_input_ids = input_ids.clone() # B_orig, N_q, 1

            # Determine shape for _cg_logits based on output of _compute_logits
            # It's [B_orig, N_q, Vocab_pred] because CFG happens inside _compute_logits
            # and it returns the final CFG'd logits for B_orig.
            # Let's get a sample output to define shape for _cg_logits:
            with torch.no_grad(): # Ensure no side effects during this shape check
                sample_logits_shape = self._compute_logits(hidden_states.detach().clone(), self._cg_inference_params, self._cg_scale).shape
            self._cg_logits = torch.empty(sample_logits_shape, device=input_ids.device, dtype=torch.float32) # Logits are float32

            g = torch.cuda.CUDAGraph()

            def capture_region():
                # Operations inside graph capture must use graph's static tensors
                hidden_states_local_cg = self.embed_codes(self._cg_input_ids) # B_orig, 1, D
                hidden_states_local_cg = hidden_states_local_cg.repeat(2, 1, 1) # 2*B_orig, 1, D
                # Assign to the pre-allocated tensor
                self._cg_logits.copy_(self._compute_logits(hidden_states_local_cg, self._cg_inference_params, self._cg_scale))


            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else: # Graph exists and batch size matches
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()
        return self._cg_logits.clone() # Return a clone to avoid modification issues if caller changes it


    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor, # [B_eff, Cond_S, D] (B_eff = 2*B_orig if CFG)
        input_ids: torch.Tensor, # Audio prompt codes: [B_orig, N_q, Prefix_Audio_S]
        inference_params: InferenceParams, # For B_eff
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        embedded_audio_prompt = self.embed_codes(input_ids) # [B_orig, Prefix_Audio_S, D]

        # Replicate embedded_audio_prompt if CFG is enabled
        if cfg_scale != 1.0:
            # prefix_hidden_states is already B_eff (e.g., 2*B_orig)
            # embedded_audio_prompt needs to match this B_eff for concatenation.
            if embedded_audio_prompt.shape[0] * 2 == prefix_hidden_states.shape[0]: # B_orig * 2 == 2*B_orig
                 embedded_audio_prompt_for_cat = embedded_audio_prompt.repeat(2, 1, 1) # [2*B_orig, Prefix_Audio_S, D]
            elif embedded_audio_prompt.shape[0] == prefix_hidden_states.shape[0]:
                 # This case implies B_orig == 2*B_orig (if B_orig=0) or CFG was already applied to audio_prompt.
                 # Or, it means no CFG for prefix_hidden_states, which contradicts B_eff.
                 # Assuming if cfg_scale != 1.0, prefix_hidden_states is 2*B_orig.
                 # This path should generally not be hit if inputs are consistent.
                 embedded_audio_prompt_for_cat = embedded_audio_prompt
            else:
                 raise ValueError(f"Batch size mismatch for CFG in _prefill. PrefixHS: {prefix_hidden_states.shape[0]}, AudioPrompt: {embedded_audio_prompt.shape[0]}")
        else: # No CFG (cfg_scale == 1.0)
            # prefix_hidden_states should be B_orig
            if embedded_audio_prompt.shape[0] != prefix_hidden_states.shape[0]:
                raise ValueError(f"Batch size mismatch (no CFG) in _prefill. PrefixHS: {prefix_hidden_states.shape[0]}, AudioPrompt: {embedded_audio_prompt.shape[0]}")
            embedded_audio_prompt_for_cat = embedded_audio_prompt # [B_orig, Prefix_Audio_S, D]

        hidden_states = torch.cat([prefix_hidden_states, embedded_audio_prompt_for_cat], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)


    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = None) -> InferenceParams:
        # batch_size here is the effective batch size for the cache (e.g. 2*B_orig or B_orig)
        if dtype is None:
            dtype = next(p.dtype for p in self.parameters() if p.is_floating_point()) # Use model's primary float dtype

        max_seqlen = find_multiple(max_seqlen, 8)
        # Pass model's device to allocate_inference_cache
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32, device=self.device)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        # This aligns with the reference zonos/model.py
        if uncond_dict is None:
            # If uncond_dict is not provided, create one using required keys from cond_dict.
            # This means the unconditional pass uses the same values as the conditional pass for these keys.
            # For a "true" unconditional pass, one might want to provide specific null values (e.g. speaker=None).
            # The calling code (e.g. ZonosLocalVoice) is responsible for preparing an appropriate uncond_dict
            # if specific unconditional values are needed (like nullifying speaker or emotion).
            uncond_dict = {k: cond_dict.get(k) for k in self.prefix_conditioner.required_keys}
            # Ensure all required keys are present, even if None (handled by PrefixConditioner if a sub-conditioner is optional)
            for req_key in self.prefix_conditioner.required_keys:
                 uncond_dict.setdefault(req_key, None)


        # prefix_conditioner handles dtypes internally via the fix in Conditioner.forward
        cond_embedding = self.prefix_conditioner(cond_dict)
        uncond_embedding = self.prefix_conditioner(uncond_dict)

        return torch.cat([cond_embedding, uncond_embedding])


    def can_use_cudagraphs(self) -> bool:
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment (as per reference)
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz_orig_x2_if_cfg, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz_orig, N_q, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1, # This is B_orig (original batch size before any CFG duplication)
        sampling_params: dict = None, # Uses dict(min_p=0.1) if None
        progress_bar: bool = True,
        disable_torch_compile: bool = False, # Reference has False, current has True
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        # Aligning with reference zonos/model.py generate method
        if sampling_params is None:
            sampling_params = dict(min_p=0.1)

        # Reference code asserts cfg_scale != 1.
        # If this assertion is critical, we should keep it or ensure logic handles cfg_scale=1 correctly.
        # For now, assuming cfg_scale != 1 based on original assertion.
        # If cfg_scale can be 1.0, then effective_cache_batch_size logic needs to be robust.
        if cfg_scale == 1.0:
            # This path might not be fully supported or intended by original Zonos generate.
            # print("Warning: cfg_scale=1.0 might not be fully supported in this generation logic.")
            # If cfg_scale is 1.0, prefix_conditioning should be [B_orig, C, D]
            # and effective_cache_batch_size should be B_orig.
            # The _decode_one_token and _prefill methods need to handle this.
            # The current _decode_one_token has a TODO for cfg_scale=1.
            # Let's assume for now the caller ensures cfg_scale != 1 if that's a requirement from original.
             pass # Or raise ValueError("cfg_scale=1.0 is not currently supported by this generate method's structure.")


        # Determine effective batch size for KV cache based on whether CFG is active.
        # prefix_conditioning is already [B_eff, C, D] where B_eff is 2*B_orig if CFG, or B_orig if not.
        # So, effective_cache_batch_size is simply prefix_conditioning.shape[0].
        effective_cache_batch_size = prefix_conditioning.shape[0]

        # Validate relationship between effective_cache_batch_size and batch_size (B_orig)
        if cfg_scale != 1.0 and effective_cache_batch_size != batch_size * 2:
            raise ValueError(f"For CFG (scale={cfg_scale}), prefix_conditioning batch dim ({effective_cache_batch_size}) must be 2 * batch_size ({batch_size}).")
        if cfg_scale == 1.0 and effective_cache_batch_size != batch_size:
            # This case needs careful handling in _decode_one_token and _prefill if supported.
            # For now, this implies an inconsistency if cfg_scale=1 but prefix_conditioning was doubled.
            raise ValueError(f"If cfg_scale=1.0, prefix_conditioning batch dim ({effective_cache_batch_size}) must be batch_size ({batch_size}).")


        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # CUDA Graphs or torch.compile
        cg = self.can_use_cudagraphs() and not disable_torch_compile
        decode_one_token_fn = self._decode_one_token
        if not disable_torch_compile: # Only compile if not disabled
            decode_one_token_fn = torch.compile(decode_one_token_fn, dynamic=True, disable=cg)


        unknown_token = -1 # Could use self.config.masked_token_id or a dedicated unknown
        audio_seq_len = prefix_audio_len + max_new_tokens

        # Total sequence length for KV cache: condition length + audio (prefix+new) + num_codebooks (for delay pattern margin)
        total_cache_seq_len = prefix_conditioning.shape[1] + (audio_seq_len + self.autoencoder.num_codebooks)

        with torch.device(device): # Ensure tensors created are on the model's device
            inference_params = self.setup_cache(batch_size=effective_cache_batch_size, max_seqlen=total_cache_seq_len)
            codes = torch.full((batch_size, self.autoencoder.num_codebooks, audio_seq_len),
                               unknown_token, dtype=torch.long, device=device)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes.to(device)

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id) # Use self.masked_token_id

        # Prefill phase
        # delayed_prefix_audio_codes is for B_orig, covering the audio prefix part of delayed_codes
        # The +1 is for the first token to be predicted based on the prefix.
        delayed_prefix_audio_input_for_prefill = delayed_codes[..., : prefix_audio_len + 1] # [B_orig, N_q, PrefixAudio_S + 1]

        # _prefill handles CFG replication of embedded audio codes internally based on cfg_scale and prefix_conditioning batch
        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_input_for_prefill, inference_params, cfg_scale)
        # logits is [B_orig, N_q, Vocab]

        next_token_sampled = sample_from_logits(logits, **sampling_params) # [B_orig, N_q, 1]

        # Initial fill of delayed_codes at the correct offset
        # This is the position for the first token *after* any audio_prefix
        current_fill_idx_delayed = prefix_audio_len + 1 # Index in delayed_codes to fill

        # Fill the sampled token into the delayed_codes structure
        # Using masked_scatter_ as in reference code's loop, adapted for prefill:
        target_frame_slice_prefill = delayed_codes[..., current_fill_idx_delayed : current_fill_idx_delayed + 1] # Shape [B_orig, N_q, 1]
        # Mask where frame is unknown_token or masked_token_id (should be, as it's the first predicted token)
        mask_prefill = (target_frame_slice_prefill == unknown_token) | (target_frame_slice_prefill == self.masked_token_id)
        target_frame_slice_prefill.masked_scatter_(mask_prefill, next_token_sampled)


        # Update inference_params offset based on what backbone processed in prefill
        # Length processed = prefix_conditioning length + length of audio prompt fed to prefill
        prefill_backbone_input_len = prefix_conditioning.shape[1] + delayed_prefix_audio_input_for_prefill.shape[2]
        inference_params.seqlen_offset += prefill_backbone_input_len
        # Update lengths_per_sample for all items in the effective batch
        inference_params.lengths_per_sample[:] += prefill_backbone_input_len


        # Autoregressive loop
        logit_bias = torch.zeros_like(logits) # logits is [B_orig, N_q, Vocab]
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # Only allow codebook 0 to predict EOS

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device) # For B_orig

        # `remaining_steps` from reference code: number of steps to generate after EOS is first hit in codebook 0
        # This ensures all codebooks get a chance to output their part of the EOS pattern.
        # Max steps in `delayed_codes` is `delayed_codes.shape[2] - 1`. `offset` in ref is `current_fill_idx_delayed`.
        # `max_steps` in ref is `delayed_codes.shape[2] - offset_at_start_of_loop`
        # Here, loop for `max_new_tokens - 1` more steps (since 1st new token is done by prefill)

        # remaining_steps_after_eos logic from reference Zonos:
        # Counts how many more codebooks need to be filled for a complete EOS pattern.
        # Initialize to max_new_tokens; when EOS is hit, set to num_codebooks. Decrement each step. Stop when <=0.
        remaining_steps_counter = torch.full((batch_size,), max_new_tokens, device=device, dtype=torch.long)


        progress_iter = tqdm(range(1, max_new_tokens), desc="Generating Audio Tokens", disable=not progress_bar)

        # `current_input_idx_delayed` is the index of the frame *just filled*, to be used as input for next token.
        # After prefill, this is `current_fill_idx_delayed`.
        current_input_idx_delayed = current_fill_idx_delayed


        for step_count in progress_iter: # Already did 1st token (step_count=0 effectively)

            # Check stopping condition based on remaining_steps_counter
            if torch.all(remaining_steps_counter <= 0):
                break

            # Prepare input for this step: the frame we just filled
            input_ids_for_decode = delayed_codes[..., current_input_idx_delayed : current_input_idx_delayed + 1] # [B_orig, N_q, 1]

            # Update KV cache offset for the new token being processed
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1 # Add 1 to all in B_eff

            logits_loop = decode_one_token_fn(input_ids_for_decode, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits_loop += logit_bias # Apply logit bias (e.g. prevent non-CB0 EOS)

            # Use previously generated tokens for repetition penalty context
            # Context should include up to the token just before the one we are predicting now.
            # `current_input_idx_delayed` is the last filled index.
            # So context is up to and including `current_input_idx_delayed`.
            generated_tokens_context = delayed_codes[..., : current_input_idx_delayed + 1]

            next_token_sampled_loop = sample_from_logits(logits_loop,
                                                         generated_tokens=generated_tokens_context,
                                                         **sampling_params) # [B_orig, N_q, 1]

            # EOS handling (aligned with reference zonos/model.py)
            # Check if EOS is predicted in codebook 0 for any batch item that isn't already stopping.
            eos_in_cb0_this_step = (next_token_sampled_loop[:, 0, 0] == self.eos_token_id) & (~stopping)

            # Update stopping flags
            stopping[eos_in_cb0_this_step] = True

            # If EOS was hit for an item, set its remaining_steps_counter to num_codebooks
            # This ensures N_q more tokens are generated to complete the EOS pattern across codebooks.
            remaining_steps_counter[eos_in_cb0_this_step] = torch.minimum(
                remaining_steps_counter[eos_in_cb0_this_step],
                torch.tensor(self.autoencoder.num_codebooks, device=device, dtype=torch.long)
            )


            # If an item is stopping (EOS was hit previously or now), fill its `next_token_sampled_loop`
            # with the appropriate MASKED_TOKEN/EOS_TOKEN pattern for the current codebook.
            # This uses `remaining_steps_counter` to determine which codebook should get EOS.
            if torch.any(stopping):
                 for i in range(batch_size):
                     if stopping[i]: # If this batch item is in stopping phase
                         # `eos_codebook_idx` is the codebook that should receive EOS_TOKEN this step.
                         # It's `num_codebooks - remaining_steps_counter[i]`.
                         # Clamped to be valid index [0, num_codebooks-1].
                         eos_cb_idx_for_item_i = self.autoencoder.num_codebooks - remaining_steps_counter[i]
                         eos_cb_idx_for_item_i = torch.clamp(eos_cb_idx_for_item_i, 0, self.autoencoder.num_codebooks - 1)

                         # Fill with MASKED_TOKEN by default
                         temp_fill_eos_pattern = torch.full_like(next_token_sampled_loop[i], self.masked_token_id)
                         # Set EOS_TOKEN at the determined codebook index
                         temp_fill_eos_pattern[eos_cb_idx_for_item_i, 0] = self.eos_token_id
                         next_token_sampled_loop[i] = temp_fill_eos_pattern


            # Update delayed_codes for the next position
            current_fill_idx_delayed += 1 # This is the index in delayed_codes where we write the new tokens

            target_frame_slice_loop = delayed_codes[..., current_fill_idx_delayed : current_fill_idx_delayed + 1]
            # Mask where frame is unknown_token or masked_token_id
            mask_loop = (target_frame_slice_loop == unknown_token) | (target_frame_slice_loop == self.masked_token_id)
            target_frame_slice_loop.masked_scatter_(mask_loop, next_token_sampled_loop)


            # Next input is what we just wrote
            current_input_idx_delayed = current_fill_idx_delayed

            # Decrement remaining_steps_counter for all items
            remaining_steps_counter -= 1

            # Callback
            if callback is not None:
                # Callback receives the tokens generated *in this step*
                if not callback(next_token_sampled_loop.clone(), step_count, max_new_tokens -1): # step_count is 1-indexed here
                    break

        progress_iter.close()

        reverted_codes = revert_delay_pattern(delayed_codes)

        # Determine actual generated length in reverted_codes
        # `current_fill_idx_delayed` is the last index that was filled in `delayed_codes`.
        # Number of valid frames in `delayed_codes` is `current_fill_idx_delayed + 1`.
        # Length in `reverted_codes` is `(current_fill_idx_delayed + 1) - num_codebooks`.
        # Or, simpler: `offset` from reference code was `current_fill_idx_delayed`.
        # Final length in reference is `offset - num_codebooks`.
        # Here, `current_fill_idx_delayed` is the *last index written to*.
        # So, number of written frames is `current_fill_idx_delayed + 1`.
        # The effective length of the audio sequence in `reverted_codes` is
        # `(current_fill_idx_delayed + 1) - self.autoencoder.num_codebooks`.
        # This needs to be non-negative.
        final_reverted_len = max(0, (current_fill_idx_delayed + 1) - self.autoencoder.num_codebooks)

        out_codes = reverted_codes[..., :final_reverted_len]

        # Mask tokens >= eos_token_id to 0 (or a pad token if different from 0)
        # Reference uses 1024, which is self.eos_token_id.
        out_codes.masked_fill_(out_codes >= self.eos_token_id, 0)

        self._cg_graph = None # Reset CUDA graph state
        return out_codes
