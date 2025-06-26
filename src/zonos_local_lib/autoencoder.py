import math
import torch
import torchaudio
# from huggingface_hub import hf_hub_download # No longer needed for weights path

# Import the vendored DAC model loader
from .dac_codec.utils import load_model as load_dac_model
# The DAC class itself will be available via the loaded model instance

class DACAutoencoder:
    def __init__(self, device_str: str = "cuda"):
        super().__init__()

        self.device = torch.device(device_str)

        try:
            print("INFO: DACAutoencoder - Attempting to load DAC model via vendored utility...")
            # Load the 44kHz, 9-codebook model (often referred to as 8kbps target, tag "0.0.1" or "latest")
            # The load_dac_model utility handles finding/downloading the correct weights.
            self.dac_model = load_dac_model(model_type="44khz", model_bitrate="8kbps", tag="latest")
            print(f"INFO: DACAutoencoder - DAC model loaded successfully. Type: {type(self.dac_model)}")

            # Ensure model is on the correct device and in eval mode
            self.dac_model.to(self.device)
            self.dac_model.eval()

            # Set attributes from the loaded model
            self.codebook_size = self.dac_model.codebook_size
            self.num_codebooks = self.dac_model.n_codebooks
            self.sampling_rate = self.dac_model.sample_rate # Should be 44100 for this model

            print(f"INFO: DACAutoencoder - Loaded DAC model properties: SR={self.sampling_rate}, Codebooks={self.num_codebooks}, CodebookSize={self.codebook_size}")

        except Exception as e:
            print(f"CRITICAL: DACAutoencoder - Failed to load or configure DAC model. Error: {e}")
            # Propagate error or ensure subsequent calls fail gracefully
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DACAutoencoder initialization failed: {e}")


    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # wav is [B, L] or [L]
        # Ensure wav is on the same device as dac_model for preprocessing
        wav = wav.to(self.dac_model.device)

        # Convert to [B, 1, L] for DAC model's preprocess & encode
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
<<<<<<< fix/dtype-mismatch-conditioning

=======
>>>>>>> master
        if sr != self.dac_model.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.dac_model.sample_rate).to(wav.device)
            wav = resampler(wav)

        # Call the dac_model's own preprocess method
        return self.dac_model.preprocess(wav, sample_rate=self.dac_model.sample_rate)

    def encode(self, wav: torch.Tensor) -> torch.Tensor: # wav is preprocessed: [B, 1, T_padded]
        # DAC's encode returns: z_q, codes, latents, commitment_loss, codebook_loss
        # We need the `codes` which are [B, N_codebooks, T_codes]
        # Ensure input is on the correct device
        _, codes, _, _, _ = self.dac_model.encode(wav.to(self.dac_model.device), n_quantizers=self.num_codebooks)
        return codes

<<<<<<< fix/dtype-mismatch-conditioning
    def decode(self, codes: torch.Tensor, chunk_size_codes: int = None) -> torch.Tensor: # codes are [B, N_codebooks, T_codes]
        codes = codes.to(self.dac_model.device)
        _B, _Nq, T_codes = codes.shape # Get Batch, NumQuantizers, TimeCodes

        if chunk_size_codes is None:
            # Determine a reasonable chunk size for codes if not provided
            # Based on DAC's model hop_length (e.g., 44100 SR / 512 hop = ~86 code frames per second)
            # Let's aim for approx 2-4 seconds of codes per chunk to balance memory and overhead.
            # Example: 86 frames/sec * 2 sec = 172 code frames.
            # Using a slightly larger default for more efficiency if memory allows.
            # This default might need tuning.
            default_chunk_duration_sec = 2.0
            if hasattr(self.dac_model, 'hop_length') and self.dac_model.hop_length > 0:
                 frames_per_sec = self.dac_model.sample_rate / self.dac_model.hop_length
                 chunk_size_codes = int(frames_per_sec * default_chunk_duration_sec)
            else: # Fallback if hop_length not directly available on dac_model instance
                 chunk_size_codes = 172 # Default to ~2s at 44.1kHz/512hop

            # Ensure chunk_size is at least 1 and not greater than total codes
            chunk_size_codes = max(1, chunk_size_codes)
            if T_codes <= chunk_size_codes:
                 chunk_size_codes = T_codes

        if T_codes == 0:
            print("WARNING: DACAutoencoder.decode received codes with T_codes=0. Returning empty tensor.")
            return torch.tensor([], device=self.dac_model.device, dtype=torch.float)

        reconstructed_audio_chunks = []

        print(f"INFO: DACAutoencoder.decode: Total codes T_codes={T_codes}, processing in chunk_size_codes={chunk_size_codes}")

        for i in range(0, T_codes, chunk_size_codes):
            code_chunk = codes[..., i : i + chunk_size_codes]
            # print(f"DEBUG: Decoding chunk {i // chunk_size_codes + 1}, code_chunk shape: {code_chunk.shape}")

            z_q_chunk, _, _ = self.dac_model.quantizer.from_codes(code_chunk)
            audio_chunk_padded = self.dac_model.decode(z_q_chunk)
            reconstructed_audio_chunks.append(audio_chunk_padded)

        if not reconstructed_audio_chunks:
            # This case should ideally be caught by T_codes == 0 check above
            print("ERROR: No audio chunks were decoded, though T_codes > 0. This is unexpected.")
            return torch.tensor([], device=self.dac_model.device, dtype=torch.float)

        reconstructed_audio_padded = torch.cat(reconstructed_audio_chunks, dim=-1)

        # Note: Simple concatenation is used. If chunking introduces audible artifacts
        # at boundaries, overlap-add decoding might be necessary. The original
        # CodecMixin.decompress also used simple concatenation.
        # The DAC model's `decode` should ideally handle overlaps if its receptive field
        # is managed correctly during chunked `encode` (which `compress` tries to do).
        # Since Zonos `generate` produces one long stream of codes, we are essentially
        # decoding this stream in chunks.
=======
    def decode(self, codes: torch.Tensor) -> torch.Tensor: # codes are [B, N_codebooks, T_codes]
        codes = codes.to(self.dac_model.device)

        # The vendored DAC.decode takes z_q (quantized latents).
        # We get z_q from codes using self.dac_model.quantizer.from_codes
        z_q, _, _ = self.dac_model.quantizer.from_codes(codes)

        reconstructed_audio_padded = self.dac_model.decode(z_q)
>>>>>>> master

        return reconstructed_audio_padded.float() # Ensure float output, shape [B, 1, L]
