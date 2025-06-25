import math

import torch
import torchaudio
# from transformers.models.dac import DacModel # Remove this direct import
from transformers import AutoModel # Use AutoModel


class DACAutoencoder:
    def __init__(self):
        super().__init__()
        # Load using AutoModel, let transformers figure out the correct class
        self.dac = AutoModel.from_pretrained("descript/dac_44khz")
        # It's good practice to check if the loaded model is what we expect,
        # though for well-known models, AutoModel is usually reliable.
        # You might need to ensure methods like .encode, .decode, .quantizer.n_codebooks,
        # .config.codebook_size, .config.sampling_rate are available on the object returned by AutoModel.
        # For "descript/dac_44khz", AutoModel should indeed return a DacModel instance or compatible.

        self.dac.eval().requires_grad_(False)

        # These attributes might need to be accessed differently if AutoModel wraps DacModel unexpectedly
        # or if the config structure changes slightly with AutoModel.
        # For now, assume they are available as before.
        if hasattr(self.dac, 'quantizer') and hasattr(self.dac.quantizer, 'n_codebooks'):
             self.num_codebooks = self.dac.quantizer.n_codebooks
        elif hasattr(self.dac, 'config') and hasattr(self.dac.config, 'num_codebooks'): # Fallback if structure differs
             self.num_codebooks = self.dac.config.num_codebooks
        elif hasattr(self.dac, 'config') and hasattr(self.dac.config, 'n_codebooks'): # Zonos reference uses this
            self.num_codebooks = self.dac.config.n_codebooks
        else:
            # Fallback or error if n_codebooks cannot be found.
            # This might indicate an incompatible model or version.
            # For "descript/dac_44khz", it should typically be found.
            print("WARNING: Could not determine num_codebooks from DAC model config. Attempting direct access or default.")
            # Try direct access as per original transformers DacModel structure if possible
            if hasattr(self.dac, 'quantizer') and hasattr(self.dac.quantizer, 'n_codebooks'):
                 self.num_codebooks = self.dac.quantizer.n_codebooks
            else:
                 # Defaulting to a common value if not found, though this is risky.
                 self.num_codebooks = getattr(self.dac.config, "n_codebooks", 9)
                 print(f"WARNING: num_codebooks defaulted to {self.num_codebooks}. This might be incorrect.")


        self.codebook_size = self.dac.config.codebook_size
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # Resample to the DAC model's native sampling rate
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)

        # Pad to be a multiple of model's hop length (often 512 for DAC/EnCodec style models)
        # This might depend on the specific model's requirements, often related to its downsampling factor.
        # A common hop length or total downsampling factor for such models is 2^N, e.g., 256, 320, 512.
        # The original code used 512. Let's assume this is a stride/hop factor.
        # If the model has a specific stride, use that. Otherwise, 512 is a common value.
        # For descript/dac_44khz, the total stride is 320.
        model_stride = getattr(self.dac.config, "hop_length", None) # Check if hop_length is in config
        if model_stride is None and hasattr(self.dac.config, "upsample_rates"): # For DAC, stride is product of upsample_rates
            model_stride = 1
            for r in self.dac.config.upsample_rates:
                model_stride *= r

        if model_stride is None: # Fallback if not found
            print("Warning: Could not determine model_stride for DACAutoencoder padding. Using default 512.")
            model_stride = 512

        if wav.shape[-1] % model_stride != 0:
            right_pad = model_stride - (wav.shape[-1] % model_stride)
            wav = torch.nn.functional.pad(wav, (0, right_pad))

        return wav

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # The .encode() method should exist on the model returned by AutoModel for DAC
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # The .decode() method should exist
        # Ensure device handling for autocast is correct; self.dac.device might be needed
        device_type = self.dac.device.type if hasattr(self.dac, 'device') else "cpu"
        # Ensure codes are on the same device as the model for decode
        codes = codes.to(self.dac.device)
        with torch.autocast(device_type, dtype=torch.float16, enabled=(device_type != "cpu")):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()
