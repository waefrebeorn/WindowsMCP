from pathlib import Path
import requests # Ensure 'requests' is installed in the environment
import argbind # Ensure 'argbind' is installed
# from audiotools import ml # Not directly used in this file, but DAC class uses it.

# Adjusted import to get DAC class from our vendored structure
# Assuming src/zonos_local_lib/dac_codec/__init__.py exports DAC correctly
from ..dac_codec import DAC
# Or, more directly if DAC is in dac_codec.model.dac:
# from ..model.dac import DAC


# Accelerator = ml.Accelerator # Not used in this file's functions

__MODEL_LATEST_TAGS__ = {
    ("44khz", "8kbps"): "0.0.1", # This is likely the one for descript/dac_44khz on HF
    ("24khz", "8kbps"): "0.0.4",
    ("16khz", "8kbps"): "0.0.5",
    ("44khz", "16kbps"): "1.0.0",
}

__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "44khz",
        "1.0.0",
        "16kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}


# argbind is used as a decorator in the original file,
# but for vendoring, we might not need to re-bind if we call functions directly.
# If argbind is essential for some internal config, it needs to be installed.
# For now, assuming it's for CLI argument parsing in original repo, not essential for library use.
# @argbind.bind(group="download", positional=True, without_prefix=True)
def download(
    model_type: str = "44khz", model_bitrate: str = "8kbps", tag: str = "latest"
) -> Path: # Return type hint
    """
    Function that downloads the weights file from URL if a local cache is not found.
    """
    model_type = model_type.lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        key = (model_type, model_bitrate)
        if key not in __MODEL_LATEST_TAGS__:
            raise ValueError(f"No 'latest' tag defined for model_type='{model_type}', bitrate='{model_bitrate}'. Available keys for latest: {list(__MODEL_LATEST_TAGS__.keys())}")
        tag = __MODEL_LATEST_TAGS__[key]


    download_key = (model_type, tag, model_bitrate)
    download_link = __MODEL_URLS__.get(download_key, None)

    if download_link is None:
        raise ValueError(
            f"Could not find model URL for type={model_type}, tag={tag}, bitrate={model_bitrate}. Available keys: {list(__MODEL_URLS__.keys())}"
        )

    # Construct cache path (mirrors original logic)
    # Ensure this path is writable and appropriate for the execution environment.
    try:
        cache_dir = Path.home() / ".cache" / "descript" / "dac"
    except Exception as e:
        # Fallback if Path.home() is problematic (e.g. restricted env)
        # This might not be ideal but better than crashing.
        # A user-configurable cache path would be best in a real app.
        print(f"Warning: Could not get home directory ({e}). Using local .dac_cache.")
        cache_dir = Path(".") / ".dac_cache" / "descript" / "dac"


    local_path = cache_dir / f"weights_{model_type}_{model_bitrate}_{tag}.pth"

    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Downloading DAC weights from {download_link} to {local_path}")
        try:
            response = requests.get(download_link, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4XX or 5XX)
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"INFO: DAC weights downloaded successfully.")
        except requests.exceptions.RequestException as e:
            # Cleanup partially downloaded file if error
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass # Ignore if deletion fails
            raise RuntimeError(f"Could not download model from {download_link}. Error: {e}")
        except IOError as e:
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Could not write model to {local_path}. Error: {e}")
    else:
        print(f"INFO: DAC weights found in cache: {local_path}")

    return local_path


def load_model(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps", # This corresponds to the 9-codebook model for 44.1kHz
    tag: str = "latest", # Default to "latest" which should map to "0.0.1" for 44khz/8kbps
    load_path: str = None, # Direct path to checkpoint, if known
):
    if not load_path: # If no direct path, download using model_type, bitrate, tag
        load_path = download(
            model_type=model_type, model_bitrate=model_bitrate, tag=tag
        )

    # The DAC class has a classmethod `load` inherited from audiotools.ml.BaseModel
    # This method is responsible for loading the model from the checkpoint path.
    # It typically handles config and weights.
    # Ensure the vendored DAC class has this .load method (it should via CodecMixin -> BaseModel from audiotools)

    # The DAC class in dac/model/dac.py inherits from BaseModel (from audiotools.ml)
    # BaseModel has `load_from_checkpoint` and `load_from_folder` classmethods.
    # The original dac.utils.load_model used `generator = DAC.load(load_path)`.
    # This implies `DAC` itself should have a `load` classmethod.
    # This `load` method is provided by `audiotools.ml.ModelLoader` which is mixed into `BaseModel`.
    # So, `DAC.load(path)` should work if `audiotools` is correctly installed and DAC inherits BaseModel.

    # Check if our vendored DAC has the 'load' class method.
    if not hasattr(DAC, 'load') or not callable(getattr(DAC, 'load')):
        raise AttributeError("Vendored DAC class does not have a callable 'load' classmethod. Ensure audiotools is installed and DAC inherits from audiotools.ml.BaseModel.")

    try:
        model_instance = DAC.load(load_path)
        print(f"INFO: DAC model loaded successfully from {load_path}")
        return model_instance
    except Exception as e:
        raise RuntimeError(f"Failed to load DAC model from path {load_path} using DAC.load(). Error: {e}")

# Example usage (for testing this file if run directly, not part of the library)
if __name__ == '__main__':
    print("Testing DAC utilities...")
    try:
        # Test download (will use cache if already downloaded)
        # Using 44khz, 8kbps, tag 0.0.1 (which is often the default 44.1kHz DAC model)
        weights_file = download(model_type="44khz", model_bitrate="8kbps", tag="0.0.1")
        print(f"Downloaded/cached weights path: {weights_file}")

        # Test loading the model
        if weights_file.exists():
            dac_model = load_model(load_path=str(weights_file))
            print(f"Successfully loaded DAC model: {type(dac_model)}")
            print(f"  Sample rate: {dac_model.sample_rate}")
            print(f"  N_codebooks: {dac_model.n_codebooks}")
        else:
            print(f"Could not test load_model as weights file {weights_file} does not exist.")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
