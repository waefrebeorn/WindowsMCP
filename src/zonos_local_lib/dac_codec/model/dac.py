import math
from typing import List
from typing import Union

import numpy as np
import torch
# from audiotools import AudioSignal # This will be imported if available by .base
# from audiotools.ml import BaseModel # This will be imported if available by .base
from torch import nn

# from .base import CodecMixin # This should be fine as base.py is in the same directory
# The __init__.py in model/ handles this.
from .base import CodecMixin, DACFile # DACFile is also needed for compress/decompress type hints

# Corrected relative imports for nn module
from ..nn.layers import Snake1d
from ..nn.layers import WNConv1d
from ..nn.layers import WNConvTranspose1d
from ..nn.quantize import ResidualVectorQuantize

# audiotools.ml.BaseModel is used as a base class for DAC
# If audiotools is not installed, this will fail.
# We need a placeholder or conditional import for BaseModel if audiotools is optional.
# For now, assuming audiotools will be installed by the user as it's a core dependency for this code.
try:
    from audiotools.ml import BaseModel
    from audiotools import AudioSignal
except ImportError:
    print("WARNING: Vendored DAC - audiotools.ml.BaseModel or audiotools.AudioSignal not found. DAC class might not fully initialize or work.")
    # Define a dummy BaseModel if audiotools is not present, to allow script to load
    # This is a workaround; full functionality requires audiotools.
    class BaseModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        # Add any methods that are expected by DAC that would normally come from audiotools.ml.BaseModel
        # For example, .load / .save / .load_from_folder if DAC itself doesn't override all of them.
        # The provided DAC code overrides load_from_folder via audiotools.ml.ModelLoader,
        # which is part of BaseModel. So this dummy might not be enough.
        # The provided DAC class itself inherits from audiotools.ml.BaseModel.
        # If audiotools is not present, this class definition will fail.
        # The best approach is to ensure audiotools is installed.

# init_weights is defined and used locally, so no import change needed for it.
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: # Check if bias exists
            nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        # Ensure y is not shorter than x for residual connection if padding was 'valid' style implicitly
        # This padding logic seems to assume output y can be shorter and x needs trimming.
        if x.shape[-1] > y.shape[-1]:
            pad = (x.shape[-1] - y.shape[-1]) // 2
            x = x[..., pad : pad + y.shape[-1]]
        elif y.shape[-1] > x.shape[-1]: # Should not happen with 'same' padding goal
            pad = (y.shape[-1] - x.shape[-1]) // 2
            y = y[..., pad: pad + x.shape[-1]]

        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1): # dim here is input dim to the block
        super().__init__()
        # The ResidualUnits operate on dim // 2 channels.
        # The final WNConv1d upsamples channels from dim // 2 to dim (which is output_dim of block).
        # This seems to imply `dim` for EncoderBlock is the *output* channel dim of the block.
        # Let's assume `dim` is the number of channels *after* the strided convolution.
        # So, input to ResidualUnits is `dim_input_to_res_units = dim / stride_effect_on_channels`
        # The original code has `ResidualUnit(dim // 2, ...)`
        # and `WNConv1d(dim // 2, dim, ...)`
        # This means the input to this block has `dim // 2` channels, and output is `dim` channels.
        # This is unusual. Typically, `dim` refers to a consistent channel size through a block
        # or input channels. Let's trace:
        # Encoder init: WNConv1d(1, d_model) -> d_model channels
        # Loop: d_model *= 2; EncoderBlock(d_model, stride)
        # So, EncoderBlock receives `d_model_prev = d_model / 2` (current d_model is doubled target)
        # And it should output `d_model`.
        # Thus, `dim` for EncoderBlock should be `d_model` (target output).
        # Input to its first ResidualUnit is `d_model_prev`.
        # The WNConv1d at end of EncoderBlock: input `d_model_prev`, output `d_model`.

        # If `dim` parameter to EncoderBlock is the *output* dimension of this block (e.g. `current_d_model`)
        # then the input to this block was `current_d_model / 2`.
        # So the ResidualUnits should operate on `current_d_model / 2`.
        # And the final Conv1d should map `current_d_model / 2` to `current_d_model`.

        # Let's stick to the original structure: dim is the output channels of the final conv in this block.
        # The input to the ResidualUnits is dim // 2.
        input_to_res_units = dim // 2 # This is the channel depth for most of the block

        self.block = nn.Sequential(
            ResidualUnit(input_to_res_units, dilation=1),
            ResidualUnit(input_to_res_units, dilation=3),
            ResidualUnit(input_to_res_units, dilation=9),
            Snake1d(input_to_res_units),
            WNConv1d(
                input_to_res_units, # input channels
                dim,                # output channels
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2), # Ensure output length is input_length / stride
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_initial_model: int = 64, # Renamed from d_model to clarify it's initial
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64, # Dimensionality of the final latent space
    ):
        super().__init__()
        # Create first convolution: maps 1 input audio channel to d_initial_model channels
        self.block = [WNConv1d(1, d_initial_model, kernel_size=7, padding=3)]

        current_d_model = d_initial_model
        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride_val in enumerate(strides):
            input_channels_for_block = current_d_model
            output_channels_for_block = current_d_model * 2
            self.block += [EncoderBlock(output_channels_for_block, stride=stride_val)] # Pass target output dim
            current_d_model = output_channels_for_block

        # Last convolution: maps from final encoder dimensionality to d_latent
        self.block += [
            Snake1d(current_d_model),
            WNConv1d(current_d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = current_d_model # This is the channel dim before the final projection to d_latent
                                     # The original had self.enc_dim = d_model, where d_model was the iteratively doubled one.

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        # ConvTranspose1d upsamples. Input_dim is input channels, output_dim is output channels.
        # ResidualUnits operate on output_dim channels.
        self.block = nn.Sequential(
            Snake1d(input_dim), # Operates on input channels
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2), # Padding for ConvTranspose1d to control output length
                                                # (L_out - 1)*stride - 2*pad + k_eff = L_in
                                                # For L_out = L_in * stride, padding should be (k - stride)/2
                                                # Here, k_eff = k - 1 (dilation is 1)
                                                # padding = (kernel_size - stride)//2 might be more standard, or derived for exact length matching.
                                                # math.ceil(stride/2) is from original.
            ),
            # ResidualUnits now operate on `output_dim` channels
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_latent_dim: int, # Dimensionality of the latent space from encoder/quantizer
        d_initial_decoder: int, # Initial channel depth for decoder (e.g., 1536)
        rates: List[int], # Strides for upsampling, in reverse order of encoder
        d_out: int = 1, # Output channels (1 for mono audio)
    ):
        super().__init__()

        # Add first conv layer: maps input_latent_dim to d_initial_decoder channels
        layers = [WNConv1d(input_latent_dim, d_initial_decoder, kernel_size=7, padding=3)]

        current_d_model = d_initial_decoder
        # Add upsampling + MRF blocks
        for i, stride_val in enumerate(rates):
            input_channels_for_block = current_d_model
            output_channels_for_block = current_d_model // 2 # Halve channels as we upsample
            layers += [DecoderBlock(input_channels_for_block, output_channels_for_block, stride_val)]
            current_d_model = output_channels_for_block

        # Add final conv layer: maps final decoder stage channels to d_out audio channels
        layers += [
            Snake1d(current_d_model),
            WNConv1d(current_d_model, d_out, kernel_size=7, padding=3),
            nn.Tanh(), # To ensure output is in [-1, 1]
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Ensure BaseModel is defined, even if dummy, for class DAC to be defined.
# If audiotools is not installed, this will be the dummy.
if 'BaseModel' not in globals():
    class BaseModel(nn.Module): pass


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None, # If None, calculated from encoder_dim and rates
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2], # Should be reverse of encoder_rates for symmetry
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8, # Can be int or list for RVQ
        quantizer_dropout: float = 0.0, # Changed from bool to float for RVQ
        sample_rate: int = 44100,
    ):
        super().__init__() # Call BaseModel.__init__

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        calculated_latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim if latent_dim is not None else calculated_latent_dim

        self.hop_length = np.prod(encoder_rates) # Used for preprocess padding
        self.encoder = Encoder(encoder_dim, encoder_rates, self.latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim # Passed to RVQ
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim, # RVQ input is the full latent_dim
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim, # RVQ handles int or list
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            self.latent_dim, # Decoder input is also full latent_dim (output of RVQ)
            decoder_dim,
            decoder_rates,
        )
        # self.sample_rate = sample_rate # Already set
        self.apply(init_weights)

        # CodecMixin methods like get_delay require the model to be built.
        # Call it after all submodules are defined.
        if isinstance(self, CodecMixin): # Check if CodecMixin is properly inherited
             self.delay = self.get_delay()


    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_rate != self.sample_rate:
            # Resampling should be handled by the caller if input SR differs from model SR
            # Or, resample here:
            # resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate).to(audio_data.device)
            # audio_data = resampler(audio_data)
            raise ValueError(f"Input sample rate {sample_rate} does not match model sample rate {self.sample_rate}. Please resample.")


        length = audio_data.shape[-1]
        if length % self.hop_length != 0:
            right_pad = self.hop_length - (length % self.hop_length)
            audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor, # Expects [B, 1, T_padded]
        n_quantizers: int = None,
    ):
        z_e = self.encoder(audio_data) # Encoder output: [B, latent_dim, T_codes]
        # z_e (encoder output) is input to RVQ
        z_q, codes, latents_projected, commitment_loss, codebook_loss = self.quantizer(
            z_e, n_quantizers
        )
        # z_q is the quantized version of z_e, same shape [B, latent_dim, T_codes]
        # codes is [B, n_codebooks, T_codes]
        # latents_projected is [B, sum(codebook_dims), T_codes]
        return z_q, codes, latents_projected, commitment_loss, codebook_loss

    def decode(self, z_q: torch.Tensor): # z_q is the output of RVQ, shape [B, latent_dim, T_codes]
        return self.decoder(z_q) # Decoder output: [B, 1, T_audio_reconstructed]

    def forward(
        self,
        audio_data: torch.Tensor, # Raw audio [B, 1, T_original]
        sample_rate: int = None,  # SR of input audio_data
        n_quantizers: int = None,
    ):
        length = audio_data.shape[-1] # Original length

        # Preprocess (e.g., resample to self.sample_rate if different, pad to hop_length multiple)
        # The current DAC.preprocess expects audio_data to be already at self.sample_rate.
        # It only does padding. Resampling should happen before calling DAC.forward or inside preprocess.
        # For simplicity with external use (like Zonos autoencoder), let's assume SR matches.
        if sample_rate is not None and sample_rate != self.sample_rate:
             raise ValueError(f"Input sample rate {sample_rate} must match model sample rate {self.sample_rate}")

        audio_data_padded = self.preprocess(audio_data, self.sample_rate) # Use model's SR

        z_q, codes, latents_projected, commitment_loss, codebook_loss = self.encode(
            audio_data_padded, n_quantizers
        )

        x_reconstructed_padded = self.decode(z_q)

        return {
            "audio": x_reconstructed_padded[..., :length], # Trim to original length
            "z_quantized": z_q, # Output of RVQ (sum of VQ outputs projected back to latent_dim)
            "codes": codes,
            "latents_projected": latents_projected, # Concatenated projected latents from each VQ's in_proj
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    # Example: Create a DAC model instance
    # These parameters should match one of the configs, e.g., 44khz from base.yml
    model_config = {
        "encoder_dim": 64,
        "encoder_rates": [2, 4, 8, 8],       # hop_length = 2*4*8*8 = 512
        "latent_dim": 64 * (2**4),          # This is 64 * 16 = 1024. Original DAC paper might use specific latent_dim.
                                            # The provided dac.py calculates latent_dim if None, else uses provided.
                                            # If latent_dim in __init__ is set, it overrides calculation.
                                            # For "44khz" model, latent_dim is likely fixed (e.g. 1024 from encoder_dim * 2^len(rates))
                                            # Let's assume latent_dim is the output of encoder before RVQ.
                                            # The RVQ input_dim must match this.
        "decoder_dim": 1536,
        "decoder_rates": [8, 8, 4, 2],       # Should be reverse of encoder_rates
        "n_codebooks": 9,
        "codebook_size": 1024,
        "codebook_dim": 8,                   # RVQ will make this a list [8,8,...]
        "quantizer_dropout": 0.0,            # Changed from 1.0 to 0.0 for inference
        "sample_rate": 44100,
    }
    # Calculated latent_dim for these params: 64 * 2^4 = 64 * 16 = 1024.
    # So, RVQ input_dim should be 1024.
    # The DAC class __init__ sets self.latent_dim.
    # Let's pass it explicitly to RVQ if it's different from default 512.
    # The DAC class does: self.latent_dim = latent_dim if latent_dim is not None else encoder_dim * (2 ** len(encoder_rates))
    # Then self.quantizer = ResidualVectorQuantize(input_dim=self.latent_dim, ...)
    # This seems correct.

    model = DAC(**model_config).to("cpu") # Or "cuda" if available

    # Print model summary (from original test code)
    # for n, m in model.named_modules():
    #     o = m.extra_repr()
    #     p = sum([np.prod(p.size()) for p in m.parameters()])
    #     fn = lambda o_val, p_val: o_val + f" {p_val/1e6:<.3f}M params."
    #     setattr(m, "extra_repr", partial(fn, o_val=o, p_val=p))
    # print(model)
    print(f"Total # of params: {sum([np.prod(p.size()) for p in model.parameters()]):,}")


    # Test with dummy audio
    B, C, T = 1, 1, 44100 * 2 # Batch, Channels, Time
    dummy_audio = torch.randn(B, C, T).to(model.device)

    # Test forward pass
    output_dict = model(dummy_audio, sample_rate=model.sample_rate)
    reconstructed_audio = output_dict["audio"]
    codes = output_dict["codes"]

    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
    print(f"Codes shape: {codes.shape}") # Expected [B, n_codebooks, T_codes]
                                       # T_codes = T_padded / hop_length

    # Test compression and decompression using CodecMixin methods
    # This requires audiotools to be installed for AudioSignal
    try:
        from audiotools import AudioSignal
        dummy_signal = AudioSignal(dummy_audio, model.sample_rate)
        print(f"\nTesting compression/decompression with CodecMixin (requires audiotools):")
        # Compress
        dac_file_obj = model.compress(dummy_signal, n_quantizers=model.n_codebooks)
        print(f"  Compressed codes shape: {dac_file_obj.codes.shape}")
        print(f"  DACFile metadata: chunk_length={dac_file_obj.chunk_length}, original_length={dac_file_obj.original_length}")

        # Decompress
        decompressed_signal = model.decompress(dac_file_obj)
        print(f"  Decompressed signal shape: {decompressed_signal.audio_data.shape}")
        assert decompressed_signal.audio_data.shape == dummy_signal.audio_data.shape

        # Check if content is somewhat similar (won't be perfect due to VQ)
        # This is a very loose check.
        mse_loss = torch.nn.functional.mse_loss(decompressed_signal.audio_data, dummy_signal.audio_data)
        print(f"  MSE between original and decompress(compress(original)): {mse_loss.item()}")

    except ImportError:
        print("\nSkipping CodecMixin compress/decompress test because audiotools is not installed.")
    except Exception as e:
        print(f"\nError during CodecMixin test: {e}")

    print("\nStandalone DAC model test finished.")
