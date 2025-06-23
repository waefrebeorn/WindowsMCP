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
    parser.add_argument("--output-embedding-file", default=None, help="Path to save the generated speaker embedding file (e.g., speaker.pt).")
    parser.add_argument("--generate-embedding-only", action="store_true", help="If set, only generate and save speaker embedding, then exit.")
    # Add other Zonos parameters as needed, e.g., pitch, emotion

    args = parser.parse_args()

    if args.generate_embedding_only:
        if not args.speaker_ref_file:
            print_error("--speaker-ref-file is required when --generate-embedding-only is set.")
            sys.exit(1)
        if not args.output_embedding_file:
            print_error("--output-embedding-file is required when --generate-embedding-only is set.")
            sys.exit(1)

    # Validate paths (at least existence for inputs)
    # Text file is not needed if only generating embedding
    if not args.generate_embedding_only:
        if not args.text_file:
            print_error("Input text file (--text-file) is required for TTS.")
            sys.exit(1)
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
                # Ensure the embedding is on the CPU for saving, regardless of the processing device.
                speaker_embedding_tensor = zonos_model.make_speaker_embedding(wav, sr).cpu()
                speaker_embedding = speaker_embedding_tensor # For use in TTS if not exiting
                print("INFO: Speaker embedding created successfully.")

                if args.generate_embedding_only:
                    if args.output_embedding_file:
                        print(f"INFO: Saving speaker embedding to '{args.output_embedding_file}'...")
                        # Ensure output directory for embedding exists
                        embedding_output_dir = os.path.dirname(args.output_embedding_file)
                        if embedding_output_dir:
                            os.makedirs(embedding_output_dir, exist_ok=True)
                        torch.save(speaker_embedding_tensor, args.output_embedding_file)
                        print(f"INFO: Speaker embedding successfully saved to '{args.output_embedding_file}'.")
                        print("SUCCESS") # Signal success for embedding generation
                        sys.exit(0) # Exit after saving embedding
                    else:
                        # This case should be caught by arg validation, but as a safeguard:
                        print_error("No output file specified for saving the speaker embedding.")
                        sys.exit(1)
            except Exception as e:
                print_error(f"Failed to load reference audio or create/save speaker embedding from '{args.speaker_ref_file}': {e}")
                traceback.print_exc(file=sys.stderr)
                if args.generate_embedding_only:
                    sys.exit(1) # Critical error if only generating embedding
                # For TTS, continue without speaker embedding if it failed.
                speaker_embedding = None
        elif args.generate_embedding_only:
            # This case implies --generate-embedding-only without --speaker-ref-file, caught by arg validation.
            # However, as a safeguard during development:
            print_error("Cannot generate embedding without a speaker reference file (--speaker-ref-file).")
            sys.exit(1)

        # Proceed with TTS if not in generate_embedding_only mode
        print(f"INFO: Preparing conditioning dictionary for language '{args.language}', rate '{args.rate}'.")
        # Basic conditioning dictionary
        conditioning_params = {
            'text': text_to_synthesize,
            'speaker': speaker_embedding, # This will be None if embedding creation failed and not generate_embedding_only
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
