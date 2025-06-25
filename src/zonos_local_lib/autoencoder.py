import math
import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Import the vendored DAC model
from .dac_codec import DAC # Assuming dac_codec/__init__.py exports DAC

class DACAutoencoder:
    def __init__(self, device_str: str = "cuda"): # Added device_str
        super().__init__()

        self.device = torch.device(device_str)

        # Parameters for the "descript/dac_44khz" model (9 codebooks, 44.1kHz)
        # These typically correspond to a specific configuration, e.g., 44khz.yml
        # From dac/conf/final/44khz.yml:
        dac_params = {
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 8, 8],
            "latent_dim": None, # Calculated internally by DAC based on encoder_dim and rates
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 4, 2],
            "n_codebooks": 9,
            "codebook_size": 1024,
            "codebook_dim": 8,
            "quantizer_dropout": 0.0, # Typically 0.0 for inference
            "sample_rate": 44100,
        }

        # Instantiate the vendored DAC model
        self.dac_model = DAC(**dac_params)

        try:
            # Download checkpoint using hf_hub_download, targeting "weights.pth" as per descript-audio-codec releases
            weights_path = hf_hub_download(repo_id="descript/dac_44khz", filename="weights.pth")
            print(f"INFO: Vendored DAC - Attempting to load weights from: {weights_path}")

            # The standalone DAC model has a class method `load` which handles everything.
            # DAC.load(path) returns a new, configured, and loaded model instance.
            # This is cleaner than manual state_dict loading if the class supports it.
            # The DAC class from your dump inherits from audiotools.ml.BaseModel, which has .load
            self.dac_model = DAC.load(weights_path) # This re-initializes and loads.
            print(f"INFO: Vendored DAC model loaded successfully using DAC.load from {weights_path}")

        except Exception as e:
            print(f"WARNING: Vendored DAC - Failed to load weights for 'descript/dac_44khz' using DAC.load. Error: {e}. Model might have random weights or fail.")
            # Fallback or error if DAC.load fails. The model might be partially initialized from __init__
            # but weights would be random.

        self.dac_model.to(self.device)
        self.dac_model.eval()

        self.codebook_size = self.dac_model.codebook_size
        self.num_codebooks = self.dac_model.n_codebooks
        self.sampling_rate = self.dac_model.sample_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # wav is [B, L] or [L], must be on self.device by the time it gets here
        # Convert to [B, 1, L] for DAC model's preprocess & encode
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)

        # DAC model's preprocess expects audio at its native sample rate.
        # ZonosLocalVoice._get_speaker_embedding already resamples to self.target_device (cuda)
        # and then calls make_speaker_embedding which eventually calls this.
        # The Zonos main model's autoencoder.decode also calls this.
        # Ensure wav is on the same device as dac_model for preprocess.
        wav = wav.to(self.dac_model.device)

        if sr != self.dac_model.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.dac_model.sample_rate).to(wav.device)
            wav = resampler(wav)

        return self.dac_model.preprocess(wav, sample_rate=self.dac_model.sample_rate)

    def encode(self, wav: torch.Tensor) -> torch.Tensor: # wav is preprocessed: [B, 1, T_padded]
        # DAC's encode returns: z_q, codes, latents, commitment_loss, codebook_loss
        _, codes, _, _, _ = self.dac_model.encode(wav.to(self.dac_model.device), n_quantizers=self.num_codebooks)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor: # codes are [B, N_codebooks, T_codes]
        codes = codes.to(self.dac_model.device) # Ensure codes are on model device

        # The vendored DAC.decode takes z_q (quantized latents).
        # We need to get z_q from codes using self.dac_model.quantizer.from_codes
        # The from_codes method in the vendored RVQ returns: z_q_out, z_p_concat, codes
        z_q, _, _ = self.dac_model.quantizer.from_codes(codes)

        reconstructed_audio_padded = self.dac_model.decode(z_q)

        return reconstructed_audio_padded.float() # Ensure float output, shape [B, 1, L]

