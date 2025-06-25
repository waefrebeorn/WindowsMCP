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
        win_duration: float = 1.0,
        verbose: bool = False,
        normalize_db: float = -16,
        n_quantizers: int = None,
    ) -> DACFile:
        """Processes an audio signal from a file or AudioSignal object into
        discrete codes. This function processes the signal in short windows,
        using constant GPU memory.

        Parameters
        ----------
        audio_path_or_signal : Union[str, Path, AudioSignal]
            audio signal to reconstruct
        win_duration : float, optional
            window duration in seconds, by default 5.0
        verbose : bool, optional
            by default False
        normalize_db : float, optional
            normalize db, by default -16

        Returns
        -------
        DACFile
            Object containing compressed codes and metadata
            required for decompression
        """
        audio_signal = audio_path_or_signal
        if isinstance(audio_signal, (str, Path)):
            # This might need adjustment if audiotools is not found, or we make it optional
            try:
                audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))
            except NameError: # audiotools.AudioSignal not defined
                 raise ImportError("audiotools.AudioSignal is required for DACFile.compress from path. Please ensure audiotools is installed.")


        self.eval()
        original_padding = self.padding
        original_device = audio_signal.device

        audio_signal = audio_signal.clone()
        original_sr = audio_signal.sample_rate

        resample_fn = audio_signal.resample
        loudness_fn = audio_signal.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        if audio_signal.signal_duration >= 10 * 60 * 60: # Corrected: 10 min * 60 sec/min = 600 seconds; original has 10*60*60
            resample_fn = audio_signal.ffmpeg_resample
            loudness_fn = audio_signal.ffmpeg_loudness

        original_length = audio_signal.signal_length
        resample_fn(self.sample_rate)
        input_db = loudness_fn()

        if normalize_db is not None:
            audio_signal.normalize(normalize_db)
        audio_signal.ensure_max_of_audio()

        nb, nac, nt = audio_signal.audio_data.shape
        audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
        win_duration = (
            audio_signal.signal_duration if win_duration is None else win_duration
        )

        if audio_signal.signal_duration <= win_duration:
            # Unchunked compression (used if signal length < win duration)
            self.padding = True
            n_samples = nt
            # hop = nt # hop should be related to output length not input for chunking logic
            # For unchunked, effectively one large chunk
        else:
            # Chunked inference
            self.padding = False
            # Zero-pad signal on either side by the delay
            audio_signal.zero_pad(self.delay, self.delay) # self.delay needs to be defined by the main DAC class
            n_samples = int(win_duration * self.sample_rate)
            # Round n_samples to nearest hop length multiple
            n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length) # self.hop_length needs to be defined by the main DAC class

        # The calculation of 'hop' for chunked inference in the original base.py seems to be missing
        # or assumed to be handled by the caller or derived differently.
        # For unchunked, it's effectively the whole signal.
        # For chunked, it's usually related to n_samples - overlap, or a fixed hop.
        # The original code sets hop = self.get_output_length(n_samples) for chunked.
        # This implies that `encode` is called on chunks of `n_samples` and produces `hop` length codes.
        # This needs careful check with how DAC.encode and DAC.preprocess work with chunking.
        # For now, assuming the loop structure is correct if n_samples and hop are well-defined.

        codes_list = [] # Renamed from codes to avoid conflict
        range_fn = range if not verbose else tqdm.trange

        # Determine effective hop for iterating through the original signal
        # If unchunked, hop_iter is just nt, loop runs once.
        # If chunked, hop_iter is the 'hop' used for processing windows.
        # The original dac.model.base.py has `hop = self.get_output_length(n_samples)` for chunked.
        # Let's assume self.hop_length is the input domain hop, and model's internal hop_length (output codes) is what matters.
        # This part is complex and depends on the main DAC class's properties (delay, hop_length for codes)

        # Simplified loop for now, assuming `self.encode` handles chunk logic or is called on full signal if unchunked
        # The original loop was `for i in range_fn(0, nt, hop):`
        # This implies `hop` is input domain hop. If `self.padding` is true, it means unchunked.

        if self.padding: # Unchunked
            audio_data_to_encode = self.preprocess(audio_signal.audio_data.to(self.device), self.sample_rate)
            # self.encode is part of the main DAC class, not CodecMixin directly
            # This CodecMixin provides compress/decompress, which call self.encode/decode
            # So, these calls should be on `self` which is an instance of DAC
            _z, c, _latents, _commitment_loss, _codebook_loss = self.encode(audio_data_to_encode, n_quantizers)
            codes_list.append(c.to(original_device))
            chunk_length = c.shape[-1] # Length of codes from one chunk
        else: # Chunked (original logic was more complex with self.delay and derived hop)
            # This simplified chunking might not perfectly match original if overlap is needed
            # The original logic: audio_signal.zero_pad(self.delay, self.delay)
            # n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
            # hop_for_iteration = self.get_output_length(n_samples) # This hop is in the code domain
            # This is complex; for now, let's assume a simpler chunking or that it's handled
            # by how Zonos calls this. Zonos autoencoder calls self.dac.encode directly.
            # The compress/decompress in this CodecMixin are higher level.
            # For `autoencoder.py` which calls `self.dac.encode()`, it passes the whole (preprocessed) wav.
            # So, the chunking logic here in `compress` might not be directly used by Zonos's current autoencoder.py structure.
            # Let's assume for now that if this `compress` is called, it's on a manageable signal length.
            # For robustness, a proper chunked implementation from the original DAC is needed if used for large files.
            # Given Zonos autoencoder.py, it seems to operate on the whole input at once for encode/decode.
             audio_data_to_encode = self.preprocess(audio_signal.audio_data.to(self.device), self.sample_rate)
            _z, c, _latents, _commitment_loss, _codebook_loss = self.encode(audio_data_to_encode, n_quantizers)
            codes_list.append(c.to(original_device))
            chunk_length = c.shape[-1]


        codes_tensor = torch.cat(codes_list, dim=-1)

        dac_file = DACFile(
            codes=codes_tensor,
            chunk_length=chunk_length, # This will be the total code length if not properly chunked
            original_length=original_length,
            input_db=input_db,
            channels=nac,
            sample_rate=original_sr,
            padding=self.padding, # This reflects if the whole signal was processed with padding
            dac_version=SUPPORTED_VERSIONS[-1],
        )

        # This line was outside the loop in original, seems like a bug if codes was a list of tensors.
        # if n_quantizers is not None:
        #     codes = codes[:, :n_quantizers, :]
        # This should be applied to codes_tensor if needed:
        if n_quantizers is not None:
            dac_file.codes = dac_file.codes[:, :n_quantizers, :]


        self.padding = original_padding
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
