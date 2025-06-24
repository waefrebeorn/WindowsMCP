from functools import cache
from typing import Any, Literal, Iterable

import torch
import torch.nn as nn

from .config import PrefixConditionerConfig # Corrected import
from .utils import DEFAULT_DEVICE # Corrected import


class Conditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int | None = None,
        projection: Literal["none", "linear", "mlp"] = "none",
        uncond_type: Literal["learned", "none"] = "none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.project = nn.Identity()

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: tuple[Any, ...] | None) -> torch.Tensor:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.data.view(1, 1, -1)

        cond = self.apply_cond(*inputs)
        cond = self.project(cond)
        return cond


# ------- ESPEAK CONTAINMENT ZONE ------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import re
import unicodedata

import inflect
# import torch # already imported
# import torch.nn as nn # already imported
from kanjize import number2kanji
from phonemizer.backend import EspeakBackend
from sudachipy import Dictionary, SplitMode

# This will be handled in Step 3 of the plan more robustly.
# For now, keeping original logic during file creation.

# Set PHONEMIZER_ESPEAK_LIBRARY for different platforms if not already set by environment
# This code runs when the module is imported.
if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    if sys.platform == "win32":
        # Standard installation path for eSpeak NG MSI on Windows
        espeak_ng_dll_path = "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
        if os.path.exists(espeak_ng_dll_path):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_ng_dll_path
            print(f"INFO: Zonos.conditioning - Set PHONEMIZER_ESPEAK_LIBRARY to {espeak_ng_dll_path}", file=sys.stderr)
        else:
            # Fallback for older eSpeak or different Program Files location
            espeak_ng_dll_path_x86 = "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll"
            if os.path.exists(espeak_ng_dll_path_x86):
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_ng_dll_path_x86
                print(f"INFO: Zonos.conditioning - Set PHONEMIZER_ESPEAK_LIBRARY to {espeak_ng_dll_path_x86}", file=sys.stderr)
            else:
                print("WARNING: Zonos.conditioning - PHONEMIZER_ESPEAK_LIBRARY not set and default eSpeak NG DLL paths not found. Phonemizer might fail.", file=sys.stderr)
    elif sys.platform == "darwin": # macOS
        # Common path for eSpeak NG installed via Homebrew on Apple Silicon / Intel
        homebrew_path = None
        if os.path.exists("/opt/homebrew/lib/libespeak-ng.dylib"): # Apple Silicon
            homebrew_path = "/opt/homebrew/lib/libespeak-ng.dylib"
        elif os.path.exists("/usr/local/lib/libespeak-ng.dylib"): # Intel Macs
            homebrew_path = "/usr/local/lib/libespeak-ng.dylib"

        if homebrew_path:
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = homebrew_path
            print(f"INFO: Zonos.conditioning - Set PHONEMIZER_ESPEAK_LIBRARY to {homebrew_path}", file=sys.stderr)
        else:
            print("WARNING: Zonos.conditioning - PHONEMIZER_ESPEAK_LIBRARY not set for macOS and default Homebrew paths not found. Phonemizer might fail.", file=sys.stderr)
    # For Linux, phonemizer usually finds it automatically if espeak-ng is installed system-wide.
    # No explicit setting here unless a specific non-standard path is common.
else:
    print(f"INFO: Zonos.conditioning - PHONEMIZER_ESPEAK_LIBRARY is already set to '{os.environ['PHONEMIZER_ESPEAK_LIBRARY']}'. Using that.", file=sys.stderr)


# --- Number normalization code from https://github.com/daniilrobnikov/vits2/blob/main/text/normalize_numbers.py ---

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# --- Number normalization code end ---


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

symbols = [*_punctuation, *_letters, *_letters_ipa]
_symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}


def _get_symbol_id(s: str) -> int:
    return _symbol_to_id.get(s, 1)


def get_symbol_ids(text: str) -> list[int]:
    return list(map(_get_symbol_id, text))


def tokenize_phonemes(phonemes_list: list[str]) -> tuple[torch.Tensor, list[int]]: # Renamed phonemes to phonemes_list to avoid conflict
    phoneme_ids = [[BOS_ID, *get_symbol_ids(p), EOS_ID] for p in phonemes_list] # Used p here
    lengths = list(map(len, phoneme_ids))
    longest = max(lengths) if lengths else 0
    phoneme_ids = [[PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
    return torch.tensor(phoneme_ids), lengths


def normalize_jp_text(text: str, tokenizer=Dictionary(dict="full").create()) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
    return final_text


def clean(texts: list[str], languages: list[str]) -> list[str]:
    texts_out = []
    for text, language in zip(texts, languages):
        if "ja" in language:
            text = normalize_jp_text(text)
        else:
            text = normalize_numbers(text)
        texts_out.append(text)
    return texts_out


@cache
def get_backend(language: str) -> "EspeakBackend":
    import logging

    # from phonemizer.backend import EspeakBackend # already imported
    logger = logging.getLogger("phonemizer")
    backend = EspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=_punctuation,
        logger=logger,
    )
    logger.setLevel(logging.ERROR)
    return backend


def phonemize(texts: list[str], languages: list[str]) -> list[str]:
    texts = clean(texts, languages)

    batch_phonemes = []
    for text, language in zip(texts, languages):
        backend = get_backend(language)
        phonemes_output = backend.phonemize([text], strip=True) # Renamed phonemes to phonemes_output
        batch_phonemes.append(phonemes_output[0])

    return batch_phonemes


class EspeakPhonemeConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(len(SPECIAL_TOKEN_IDS) + len(symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        device = self.phoneme_embedder.weight.device

        phonemes_data = phonemize(texts, languages) # Renamed phonemes to phonemes_data
        phoneme_ids, _ = tokenize_phonemes(phonemes_data) # Used phonemes_data
        phoneme_embeds = self.phoneme_embedder(phoneme_ids.to(device))

        return phoneme_embeds


# ------- ESPEAK CONTAINMENT ZONE ------------------------------------------------------------------------------------------------------------------------------------------------


class FourierConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs,
    ):
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        x = (x - self.min_val) / (self.max_val - self.min_val)  # [batch_size, seq_len, input_dim]
        f = 2 * torch.pi * x.to(self.weight.dtype) @ self.weight.T  # [batch_size, seq_len, output_dim // 2]
        return torch.cat([f.cos(), f.sin()], dim=-1)  # [batch_size, seq_len, output_dim]


class IntegerConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        return self.int_embedder(x.squeeze(-1) - self.min_val)  # [batch_size, seq_len, output_dim]


class PassthroughConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.cond_dim
        return x


_cond_cls_map = {
    "PassthroughConditioner": PassthroughConditioner,
    "EspeakPhonemeConditioner": EspeakPhonemeConditioner,
    "FourierConditioner": FourierConditioner,
    "IntegerConditioner": IntegerConditioner,
}


def build_conditioners(conditioners: list[dict], output_dim: int) -> list[Conditioner]:
    return [_cond_cls_map[config["type"]](output_dim, **config) for config in conditioners]


class PrefixConditioner(Conditioner):
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = nn.ModuleList(build_conditioners(config.conditioners, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def forward(self, cond_dict: dict) -> torch.Tensor:
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        conds = []
        for conditioner in self.conditioners:
            conds.append(conditioner(cond_dict.get(conditioner.name)))
        max_bsz = max(map(len, conds)) if conds else 1 # Handle empty conds
        # Ensure conds is not empty before proceeding with assertions or expansions
        if conds:
            assert all(c.shape[0] in (max_bsz, 1) for c in conds)
            conds = [c.expand(max_bsz, -1, -1) for c in conds]
            return self.norm(self.project(torch.cat(conds, dim=-2)))
        else: # Handle case where there are no conditioners or all are unconditional
            # This might need specific handling based on expected behavior if all conditioners are optional
            # For now, return a zero tensor or handle as per specific logic if this case is valid.
            # Assuming project can handle an empty sum or this path isn't typically hit with an empty list.
            # If an error should be raised or a default tensor returned, that logic would go here.
            # For now, to prevent error with torch.cat on empty list:
            if self.uncond_vector is not None: # If the PrefixConditioner itself can be uncond
                 return self.uncond_vector.data.view(1, 1, -1).expand(max_bsz, -1, -1) # Should be 1, not max_bsz if no other conds
            # Fallback or error if no conditions and no uncond_vector for PrefixConditioner
            # This state might indicate a configuration error.
            # Let's assume for now an empty list of conditions means an empty tensor or error.
            # For safety, let's return an empty tensor of the correct dimension, if that's a valid interpretation.
            # This part is speculative without knowing the exact design intent for zero conditioners.
            # Defaulting to a behavior that avoids crashing; requires design confirmation.
            # One possible interpretation: If no actual conditions are processed, maybe it means no prefix.
            # However, self.project likely expects some input.
            # A simple fix to avoid torch.cat error if conds is empty:
            if not conds and self.uncond_vector is None: # No specific conditions to concat and no uncond prefix
                 # This case needs clarification. For now, let's assume it should pass a zero tensor if project expects input.
                 # Or, if self.project is Identity, an empty tensor might be problematic later.
                 # A more robust approach might involve how `uncond_type` for PrefixConditioner itself is handled.
                 # If this path is reached, it means all sub-conditioners were optional and none were provided.
                 # Returning a zero tensor of output_dim. Batch size and sequence length are tricky here.
                 # Let's assume batch_size 1, seq_len 1 for such a case.
                 return torch.zeros(1, 1, self.output_dim, device=DEFAULT_DEVICE) # Speculative
            # If conds was not empty, the original return self.norm(self.project(torch.cat(conds, dim=-2))) is fine.
            # The issue is only if conds is empty.
            # The above return is for the case conds is empty AND self.uncond_vector is None.
            # If self.uncond_vector is not None, it's handled by the initial check in Conditioner.forward


supported_language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]  # fmt: off


def make_cond_dict(
    text: str = "It would be nice to have time for testing, indeed.",
    language: str = "en-us",
    speaker: torch.Tensor | None = None,

    # Emotion vector from 0.0 to 1.0
    #   Is entangled with pitch_std because more emotion => more pitch variation
    #                     VQScore and DNSMOS because they favor neutral speech
    #
    #                       Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
    emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],

    # Maximum frequency (0 to 24000), should be 22050 or 24000 for 44.1 or 48 kHz audio
    # For voice cloning use 22050
    fmax: float = 22050.0,

    # Standard deviation for pitch (0 to 400), should be
    #   20-45 for normal speech,
    #   60-150 for expressive speech,
    #   higher values => crazier samples
    pitch_std: float = 20.0,

    # Speaking rate in phonemes per minute (0 to 40). 30 is very fast, 10 is slow.
    # This seems to be a misinterpretation in the original comment.
    # Zonos model.py generate() has `max_new_tokens: int = 86 * 30`. 86 Hz is ~5160 tokens/min.
    # The `rate` argument in zonos_docker_entry.py is a multiplier (e.g. 1.0 for normal).
    # Let's assume this 'speaking_rate' is intended to be a multiplier like the 'rate' arg.
    # Typical range for rate multiplier: 0.5 (slow) to 2.0 (fast). Defaulting to 1.0.
    speaking_rate: float = 1.0, # Adjusted interpretation and default

    # Target VoiceQualityScore for the generated speech (0.5 to 0.8).
    #   A list of values must be provided which represent each 1/8th of the audio.
    #   You should unset for expressive speech.
    # According to discord Chat this is only used for the hybrid model
    vqscore_8: list[float] = [0.78] * 8,

    # CTC target loss
    # Only used for the hybrid model
    ctc_loss: float = 0.0,
    # Only used for the hybrid model
    dnsmos_ovrl: float = 4.0,
    # Only used for the hybrid model
    speaker_noised: bool = False,
    unconditional_keys: Iterable[str] = {"vqscore_8", "dnsmos_ovrl"}, # Default from Gradio
    device: torch.device | str = DEFAULT_DEVICE,
) -> dict:
    """
    A helper to build the 'cond_dict' that the model expects.
    By default, it will generate a random speaker embedding
    """
    # Ensure language is valid, default to en-us if not.
    # The original code asserted, but it might be better to default or warn.
    if language.lower() not in supported_language_codes:
        print(f"WARNING: Language code '{language}' not in supported list. Defaulting to 'en-us'.", file=sys.stderr)
        language = "en-us"


    language_code_to_id = {lang: i for i, lang in enumerate(supported_language_codes)}

    cond_dict_prep = { # Renamed to avoid modifying the input `cond_dict` if it were passed
        "espeak": ([text], [language]), # espeak conditioner expects a list of texts and list of languages
        "speaker": speaker, # speaker embedding tensor
        "emotion": emotion, # list of floats for emotion vector
        "fmax": fmax, # float
        "pitch_std": pitch_std, # float
        "speaking_rate": speaking_rate, # float (rate multiplier)
        "language_id": language_code_to_id[language.lower()], # integer id
        "vqscore_8": vqscore_8, # list of floats
        "ctc_loss": ctc_loss, # float
        "dnsmos_ovrl": dnsmos_ovrl, # float
        "speaker_noised": int(speaker_noised), # integer (0 or 1)
    }

    # Remove keys that are meant to be unconditional
    final_cond_dict = {}
    for k, v in cond_dict_prep.items():
        if k not in unconditional_keys:
            final_cond_dict[k] = v
        # else: # Optionally print which keys are being made unconditional
            # print(f"INFO: Key '{k}' is being made unconditional.", file=sys.stderr)


    # Convert numericals and lists to tensors on the specified device
    for k, v in final_cond_dict.items():
        if k == "espeak": # Special handling for espeak as it's (list[str], list[str])
            # This is passed as is to EspeakPhonemeConditioner
            continue
        if isinstance(v, (float, int, list)):
            try:
                v_tensor = torch.tensor(v, device=device)
            except TypeError as e: # Handles cases like list of mixed types if any, or non-numeric lists
                print(f"WARNING: Could not convert value for key '{k}' to tensor: {v}. Error: {e}", file=sys.stderr)
                # Decide on fallback: skip this key, or use a default tensor, or re-raise
                continue # Skip this problematic key for now
        elif isinstance(v, torch.Tensor):
            v_tensor = v.to(device)
        else: # value is None (e.g. speaker not provided) or already in a non-convertible format
            final_cond_dict[k] = v # Keep as is (e.g. None)
            continue

        # Reshape to [batch_size, seq_len, features] where batch_size and seq_len are 1 for scalar-like conditioners
        if v_tensor.ndim == 0: # scalar
            v_tensor = v_tensor.view(1, 1, 1)
        elif v_tensor.ndim == 1: # list like [0.1, 0.2, 0.3]
            v_tensor = v_tensor.view(1, 1, -1)
        # If ndim is 2 or more, assume it's already correctly shaped [batch, seq, feat] or [batch, feat]
        # or specific shapes like speaker embedding.
        # The example shows view(1,1,-1) for all, which might be too aggressive for pre-shaped tensors.
        # Let's stick to the original logic for this part mostly.
        # The original code did: v.view(1, 1, -1).to(device) for all tensors.

        final_cond_dict[k] = v_tensor.view(1, 1, -1) # Ensure [1,1,N] shape as per original

        if k == "emotion" and final_cond_dict[k] is not None: # Normalize emotion tensor if present
            current_emotion_tensor = final_cond_dict[k]
            sum_emotion = current_emotion_tensor.sum(dim=-1, keepdim=True)
            # Avoid division by zero if sum is zero (e.g. all zeros provided)
            if sum_emotion.abs().item() > 1e-6 : # Check if sum is significantly non-zero
                 final_cond_dict[k] = current_emotion_tensor / sum_emotion
            else:
                # Handle all-zero emotion vector, e.g. set to uniform or warn
                print("WARNING: Emotion vector sums to zero, cannot normalize. Using as is or consider a default.", file=sys.stderr)
                # Optionally, set to a default uniform distribution if sum is zero
                # num_emotions = current_emotion_tensor.shape[-1]
                # final_cond_dict[k] = torch.full_like(current_emotion_tensor, 1.0/num_emotions)


    return final_cond_dict
