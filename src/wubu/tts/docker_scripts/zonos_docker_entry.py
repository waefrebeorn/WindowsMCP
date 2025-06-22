import argparse
import os
import sys
import traceback

# These imports are expected to work inside the Zonos Docker container
# which should have Zonos and its dependencies (torch, torchaudio) installed.
try:
    import torch
    import torchaudio
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
except ImportError as e:
    print(f"ERROR: Failed to import Zonos or its dependencies: {e}", file=sys.stderr)
    sys.exit(1)

def print_error(message):
    print(f"ERROR: {message}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Zonos TTS Docker Entry Script")
    parser.add_argument("--text-file", required=True, help="Path to the input text file inside the container.")
    parser.add_argument("--output-wav-file", required=True, help="Path to save the output WAV file inside the container.")
    parser.add_argument("--model-name", required=True, help="Name of the Zonos model to load (e.g., 'Zyphra/Zonos-v0.1-transformer').")
    parser.add_argument("--language", required=True, help="Language code for TTS (e.g., 'en-us').")
    parser.add_argument("--device", default="cpu", help="Device to run on ('cpu' or 'cuda').")
    parser.add_argument("--speaker-ref-file", default=None, help="Optional path to a speaker reference audio file inside the container.")
    parser.add_argument("--rate", type=float, default=1.0, help="Speech rate (e.g., 1.0 for normal, <1.0 slower, >1.0 faster).")
    # Add other Zonos parameters as needed, e.g., pitch, emotion

    args = parser.parse_args()

    # Validate paths (at least existence for inputs)
    if not os.path.exists(args.text_file):
        print_error(f"Input text file not found inside container: {args.text_file}")
        sys.exit(1)

    if args.speaker_ref_file and not os.path.exists(args.speaker_ref_file):
        print_error(f"Speaker reference file not found inside container: {args.speaker_ref_file}")
        sys.exit(1)

    # Determine device
    device = args.device.lower()
    if device == "cuda":
        if not torch.cuda.is_available():
            print_error("CUDA device requested, but CUDA is not available in this container/PyTorch build.")
            # Fallback to CPU might be an option, or just error out
            print("INFO: Falling back to CPU.", file=sys.stderr)
            device = "cpu"
        else:
            print(f"INFO: Using CUDA device (torch.cuda.is_available()={torch.cuda.is_available()}).")
    else:
        device = "cpu"
        print("INFO: Using CPU device.")

    try:
        # Read input text
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text_to_synthesize = f.read().strip()

        if not text_to_synthesize:
            print_error("Input text file is empty.")
            sys.exit(1)

        print(f"INFO: Loading Zonos model '{args.model_name}' on device '{device}'...")
        zonos_model = Zonos.from_pretrained(args.model_name, device=device)
        print("INFO: Zonos model loaded successfully.")

        speaker_embedding = None
        if args.speaker_ref_file:
            try:
                print(f"INFO: Loading reference audio from '{args.speaker_ref_file}' and creating speaker embedding...")
                wav, sr = torchaudio.load(args.speaker_ref_file)
                wav = wav.to(device) # Ensure tensor is on the correct device
                speaker_embedding = zonos_model.make_speaker_embedding(wav, sr)
                print("INFO: Speaker embedding created successfully.")
            except Exception as e:
                print_error(f"Failed to load reference audio or create speaker embedding from '{args.speaker_ref_file}': {e}")
                traceback.print_exc(file=sys.stderr)
                # Continue without speaker embedding, Zonos might use a generic one or user might want this.
                speaker_embedding = None


        print(f"INFO: Preparing conditioning dictionary for language '{args.language}', rate '{args.rate}'.")
        # Basic conditioning dictionary
        # More params can be added here from Zonos's make_cond_dict (pitch, energy, emotion, etc.)
        # if passed as arguments to this script.
        conditioning_params = {
            'text': text_to_synthesize,
            'speaker': speaker_embedding,
            'language': args.language,
            'rate': args.rate
        }

        cond_dict = make_cond_dict(**conditioning_params)
        conditioning = zonos_model.prepare_conditioning(cond_dict)
        print("INFO: Conditioning prepared.")

        print("INFO: Generating audio codes...")
        codes = zonos_model.generate(conditioning)
        print("INFO: Audio codes generated.")

        print("INFO: Decoding audio codes...")
        # .cpu() is important as torchaudio.save expects CPU tensor
        wav_tensor = zonos_model.autoencoder.decode(codes).cpu()
        print("INFO: Audio codes decoded.")

        # Ensure output directory exists (though Docker volume mount should handle parent dir)
        output_dir = os.path.dirname(args.output_wav_file)
        if output_dir: # Only create if path includes a directory
             os.makedirs(output_dir, exist_ok=True)

        print(f"INFO: Saving synthesized audio to '{args.output_wav_file}'...")
        torchaudio.save(args.output_wav_file, wav_tensor[0], zonos_model.autoencoder.sampling_rate, format="wav")
        print(f"INFO: Audio successfully saved to '{args.output_wav_file}'.")
        print("SUCCESS") # Signal success to the orchestrator

    except Exception as e:
        print_error(f"An unexpected error occurred during Zonos TTS synthesis: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
