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

    def decode(self, codes: torch.Tensor) -> torch.Tensor: # codes are [B, N_codebooks, T_codes]
        codes = codes.to(self.dac_model.device)

        # The vendored DAC.decode takes z_q (quantized latents).
        # We get z_q from codes using self.dac_model.quantizer.from_codes
        z_q, _, _ = self.dac_model.quantizer.from_codes(codes)

        reconstructed_audio_padded = self.dac_model.decode(z_q)

        return reconstructed_audio_padded.float() # Ensure float output, shape [B, 1, L]
