import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import tqdm
from audiotools import AudioSignal # This will require audiotools to be installed
from torch import nn

SUPPORTED_VERSIONS = ["1.0.0"]


@dataclass
class DACFile:
    codes: torch.Tensor

    # Metadata
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

    def save(self, path):
        artifacts = {
            "codes": self.codes.numpy().astype(np.uint16),
            "metadata": {
                "input_db": self.input_db.numpy().astype(np.float32),
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "padding": self.padding,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        with open(path, "wb") as f:
            np.save(f, artifacts)
        return path

    @classmethod
    def load(cls, path):
        artifacts = np.load(path, allow_pickle=True)[()]
        codes = torch.from_numpy(artifacts["codes"].astype(int))
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])


class CodecMixin:
    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        assert isinstance(value, bool)

        layers = [
            l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L

    @torch.no_grad()
    def compress(
        self,
        audio_path_or_signal: Union[str, Path, AudioSignal],
        win_duration: float = 1.0, # Default in original dac/model/base.py was 1.0, but compress in dac.utils.encode uses 5.0
        verbose: bool = False,
        normalize_db: float = -16,
        n_quantizers: int = None,
    ) -> DACFile:
        audio_signal = audio_path_or_signal
        if isinstance(audio_signal, (str, Path)):
            try:
                audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))
            except NameError:
                 raise ImportError("audiotools.AudioSignal is required. Please ensure audiotools is installed.")
            except Exception as e:
                 raise RuntimeError(f"Failed to load audio file {audio_path_or_signal}: {e}")

        self.eval()
        original_padding_state = self.padding # Store original padding state
        original_device = audio_signal.device
        audio_signal = audio_signal.clone()
        original_sr = audio_signal.sample_rate
        original_length = audio_signal.signal_length

        # Resample and normalize
        if self.sample_rate != original_sr: # self.sample_rate is the model's rate
            audio_signal.resample(self.sample_rate)

        input_db = audio_signal.loudness() # Calculate loudness after resampling to model's rate

        if normalize_db is not None:
            audio_signal.normalize(normalize_db)
        audio_signal.ensure_max_of_audio() # Ensure audio is not clipping

        # Reshape for model if needed (e.g. [B, C, T] -> [B*C, 1, T])
        # The DAC model's preprocess expects [B, 1, T] or [B, T]
        # AudioSignal typically stores data as [B, C, T]
        if audio_signal.audio_data.ndim == 3:
            nb, nac, nt = audio_signal.audio_data.shape
            audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
        elif audio_signal.audio_data.ndim == 2: # Assume [C, T] or [B, T]
            # If [C,T] and C > 1, this could be an issue. For now, assume it's [B,T] or [1,T]
            audio_signal.audio_data = audio_signal.audio_data.unsqueeze(1) # to [B,1,T]
        else: # Should be [B, 1, T]
            nt = audio_signal.audio_data.shape[-1]
            nac = audio_signal.audio_data.shape[1] if audio_signal.audio_data.ndim > 1 else 1


        # Chunking logic from original dac/model/base.py
        # self.delay and self.hop_length must be attributes of the DAC instance
        delay = self.get_delay() # Call method from CodecMixin

        # hop_length here refers to the hop of the encoder in terms of input samples
        # not to be confused with STFT hop_length for other losses.
        # This is `np.prod(self.encoder_rates)` in the DAC class.
        model_hop_length = self.hop_length if hasattr(self, 'hop_length') else np.prod(self.encoder_rates)


        # Determine if chunking is needed
        # Effective win_duration from input or signal duration if shorter
        effective_win_duration = min(audio_signal.signal_duration, win_duration) if win_duration is not None else audio_signal.signal_duration

        if audio_signal.signal_duration <= effective_win_duration : # No chunking or signal shorter than window
            self.padding = True # Pad this single chunk
            n_samples_input_chunk = audio_signal.audio_data.shape[-1] # Process the whole signal
            # This processing_hop is for iterating over the input signal in chunks.
            # For unchunked, it's just the total number of samples, so loop runs once.
            processing_hop_input_domain = n_samples_input_chunk
        else: # Chunked inference
            self.padding = False # No padding for chunks, rely on overlap (via delay)
            audio_signal.zero_pad(delay, delay) # Pad for chunked processing
            n_samples_input_chunk = int(effective_win_duration * self.sample_rate)
            n_samples_input_chunk = int(math.ceil(n_samples_input_chunk / model_hop_length) * model_hop_length)
            # The hop for processing chunks in the input domain.
            # This should be related to the output code length of a chunk to avoid re-encoding same parts.
            # output_code_hop_for_chunk = self.get_output_length(n_samples_input_chunk) # This is in terms of code frames
            # This needs to be mapped back to input domain samples.
            # The original code's loop `for i in range_fn(0, nt, hop):` suggests `hop` is input domain.
            # Let's use a hop that corresponds to the number of samples that produce one chunk of codes,
            # without overlap initially for simplicity, then adjust if overlap-add is needed for recombination.
            # The original dac.utils.encode.py calls dac.compress which uses this method.
            # Let's assume for now that `n_samples_input_chunk` is processed, and the iteration hop matches this.
            # This is non-overlapping chunking. The `delay` padding handles edge effects.
            processing_hop_input_domain = n_samples_input_chunk - (2 * delay) # Effective advance if chunks are to be overlapped by `delay`
            if processing_hop_input_domain <=0: processing_hop_input_domain = n_samples_input_chunk


        codes_list = []
        range_fn = tqdm.trange if verbose else range

        current_pos = 0
        total_input_samples_to_process = audio_signal.audio_data.shape[-1] # After initial padding for chunking if any

        while current_pos < total_input_samples_to_process:
            chunk_end = min(current_pos + n_samples_input_chunk, total_input_samples_to_process)
            x_chunk = audio_signal.audio_data[..., current_pos:chunk_end]

            # Ensure chunk has enough length for the model's preprocess step or internal needs
            # DAC.preprocess pads to a multiple of self.hop_length (encoder total stride)
            # This should be fine.

            # Preprocess chunk (handles device placement and padding to model's hop_length)
            # self.preprocess is DAC.preprocess, not CodecMixin.preprocess
            # It expects sample_rate to match self.sample_rate
            audio_data_chunk_processed = self.preprocess(x_chunk.to(self.device), self.sample_rate)

            # Encode chunk
            # self.encode is DAC.encode
            _, c, _, _, _ = self.encode(audio_data_chunk_processed, n_quantizers) # c is [B*C, Nq, T_codes_chunk]
            codes_list.append(c.to(original_device))

            if current_pos == 0: # First chunk determines the code chunk length for DACFile metadata
                # This chunk_length is in terms of code frames per processed window
                # For unchunked, it's total code frames. For chunked, it's codes from one window.
                # The DACFile.chunk_length is used in decompress to iterate.
                # It should be the number of code frames produced by one window `n_samples_input_chunk`
                # *after* encoder downsampling.
                # If unchunked (self.padding=True), it is total codes.
                # If chunked (self.padding=False), it's codes from one win_duration.
                dac_file_chunk_length = c.shape[-1]


            if not self.padding and (current_pos + processing_hop_input_domain >= total_input_samples_to_process): # Last chunk in chunked mode
                break
            current_pos += processing_hop_input_domain
            if processing_hop_input_domain == 0 : break # Safety for non-advancing hop

        codes_tensor = torch.cat(codes_list, dim=-1)

        dac_file = DACFile(
            codes=codes_tensor,
            chunk_length=dac_file_chunk_length,
            original_length=original_length,
            input_db=input_db,
            channels=nac, # Original number of channels
            sample_rate=original_sr,
            padding=self.padding, # Reflects padding mode used for the whole signal / final chunk
            dac_version=SUPPORTED_VERSIONS[-1],
        )

        if n_quantizers is not None:
            dac_file.codes = dac_file.codes[:, :n_quantizers, :]

        self.padding = original_padding_state # Restore padding state
        return dac_file

    @torch.no_grad()
    def decompress(
        self,
        obj: Union[str, Path, DACFile],
        verbose: bool = False,
    ) -> AudioSignal:
        """Reconstruct audio from a given .dac file

        Parameters
        ----------
        obj : Union[str, Path, DACFile]
            .dac file location or corresponding DACFile object.
        verbose : bool, optional
            Prints progress if True, by default False

        Returns
        -------
        AudioSignal
            Object with the reconstructed audio
        """
        self.eval()
        if isinstance(obj, (str, Path)):
            obj = DACFile.load(obj)

        original_padding = self.padding
        self.padding = obj.padding # Use padding mode from the DACFile

        range_fn = range if not verbose else tqdm.trange
        codes_tensor = obj.codes # Renamed from codes
        original_device = codes_tensor.device
        chunk_length = obj.chunk_length # This is codes per chunk
        recons_list = [] # Renamed

        # Loop over code chunks for decoding
        for i in range_fn(0, codes_tensor.shape[-1], chunk_length):
            c_chunk = codes_tensor[..., i : i + chunk_length].to(self.device)
            # self.quantizer and self.decode are part of the main DAC class
            z_chunk = self.quantizer.from_codes(c_chunk)[0] # Get z_q from RVQ.from_codes
            r_chunk = self.decode(z_chunk) # Decode this chunk of z_q
            recons_list.append(r_chunk.to(original_device))

        recons_tensor = torch.cat(recons_list, dim=-1)
        # The delay removal / overlap-add is implicitly handled if self.decode accounts for it
        # or if chunk_length for codes and processing in decode are correctly aligned.
        # The original DAC model's `decode` takes Z and produces output.
        # If chunking was done correctly in `compress` (respecting model delay/receptive field),
        # then simple concatenation might be okay, or overlap-add needed.
        # For now, direct concatenation as per original structure.

        recons_signal = AudioSignal(recons_tensor, self.sample_rate) # Use self.sample_rate (model's rate)

        resample_fn = recons_signal.resample
        loudness_fn = recons_signal.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        if recons_signal.signal_duration >= 10 * 60: # Corrected: 10 min * 60 sec/min
            resample_fn = recons_signal.ffmpeg_resample
            loudness_fn = recons_signal.ffmpeg_loudness

        # Normalize to original loudness and resample to original sample rate
        # This order (normalize then resample) might differ from original if loudness was on original SR signal
        recons_signal.normalize(obj.input_db)
        if self.sample_rate != obj.sample_rate:
            resample_fn(obj.sample_rate)

        # Trim to original length
        recons_final = recons_signal[..., : obj.original_length]
        # Ensure loudness is calculated on the final signal if needed for metadata, though not directly used here.
        # loudness_fn() # This call was present but its return not used.

        # Reshape to original number of channels
        # Ensure recons_final has audio_data attribute if it's an AudioSignal
        final_audio_data = recons_final.audio_data if isinstance(recons_final, AudioSignal) else recons_final
        final_audio_data = final_audio_data.reshape(
            -1, obj.channels, obj.original_length
        )
        recons_output_signal = AudioSignal(final_audio_data, obj.sample_rate)


        self.padding = original_padding
        return recons_output_signal
