# Installation script for WuBu and its dependencies.
# Handles: virtual environment setup (optionally with uv), package installation, model setup placeholders.

import os
import subprocess
import sys
import platform
import shutil # For checking if uv is in PATH

# --- Configuration ---
VENV_NAME = ".venv-wubu"
REQUIREMENTS_FILE = "requirements.txt"
PYTHON_EXECUTABLE = sys.executable # Path to current Python interpreter

def run_command(command_list, error_message="Command failed", check=True, capture_output=False, **kwargs):
    """Helper to run a shell command."""
    print(f"\nRunning command: \"{' '.join(command_list)}\"")
    try:
        process = subprocess.run(command_list, check=check, text=True, capture_output=capture_output, **kwargs)
        if capture_output:
            if process.stdout: print(process.stdout)
            if process.stderr: print(process.stderr, file=sys.stderr)
        return process.returncode == 0
    except FileNotFoundError:
        print(f"\nError: Command not found: {command_list[0]}. Is it installed and in PATH?")
        if "uv" in command_list[0]: # Specific hint for uv
            print("Hint: 'uv' might not be installed. This script can try to install it if you allow.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: {error_message}. Exit code: {e.returncode}")
        if e.stdout: print(f"Stdout:\n{e.stdout}")
        if e.stderr: print(f"Stderr:\n{e.stderr}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nAn unexpected error occurred while running command: \"{' '.join(command_list)}\"")
        print(f"Error details: {e}")
        sys.exit(1)

def check_python_version():
    print("Checking Python version...")
    if sys.version_info < (3, 9):
        print(f"Error: Python 3.9 or higher is required. You have Python {platform.python_version()}.")
        sys.exit(1)
    print(f"Python version {platform.python_version()} is suitable.")

def is_tool_available(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def create_venv_with_python(venv_path):
    if os.path.exists(os.path.join(venv_path, "pyvenv.cfg")): # A common marker for venv
        print(f"Virtual environment already exists at: {venv_path}")
        return True
    print(f"Creating virtual environment at: {venv_path} using 'python -m venv'...")
    if run_command([PYTHON_EXECUTABLE, "-m", "venv", venv_path], error_message="Failed to create venv with python -m venv"):
        print("Virtual environment created with python -m venv.")
        return True
    return False

def get_python_in_venv(venv_path):
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    return os.path.join(venv_path, "bin", "python")

def get_pip_in_venv(venv_path): # Though uv might not need this directly
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    return os.path.join(venv_path, "bin", "pip")

def install_deps_with_pip(python_in_venv, requirements_path):
    print(f"Installing dependencies from: {requirements_path} using pip from venv...")
    pip_commands = [
        [python_in_venv, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [python_in_venv, "-m", "pip", "install", "-r", requirements_path]
    ]
    for cmd in pip_commands:
        if not run_command(cmd, error_message=f"pip command failed: {cmd[3]}"):
            return False
    print("Dependencies installed with pip.")
    return True

def create_venv_with_uv(venv_path, python_interp_for_uv=None):
    """Creates a venv using uv. python_interp_for_uv can specify which python uv should build the venv for."""
    if os.path.exists(os.path.join(venv_path, "pyvenv.cfg")):
        print(f"Virtual environment already exists at: {venv_path} (uv check)")
        return True
    print(f"Creating virtual environment at: {venv_path} using 'uv venv'...")
    cmd = ["uv", "venv", venv_path]
    if python_interp_for_uv: # E.g. "3.11" or path to python.exe
        cmd.extend(["-p", python_interp_for_uv])

    if run_command(cmd, error_message="Failed to create venv with uv"):
        print("Virtual environment created with uv.")
        return True
    return False

def install_deps_with_uv(venv_path_for_uv_python, requirements_path):
    """Installs dependencies using 'uv pip install' or 'uv pip sync'."""
    # uv pip sync is generally preferred if you have a requirements.lock or want exact matching.
    # For requirements.txt, `uv pip install -r` is fine.
    # uv needs to know which Python interpreter's environment to install into.
    # This can be done by activating the venv first, or by using `uv pip ... --python /path/to/venv/bin/python`
    # Or, if uv created the venv, it might activate it or know.
    # Let's assume we need to point to the venv's python.

    python_in_uv_venv = get_python_in_venv(venv_path_for_uv_python) # Path to python in the venv uv just made

    print(f"Installing dependencies from: {requirements_path} using 'uv pip install' for venv python {python_in_uv_venv}...")
    cmd = ["uv", "pip", "install", "-r", requirements_path, "--python", python_in_uv_venv]

    if run_command(cmd, error_message="Failed to install dependencies with uv pip install"):
        print("Dependencies installed with uv.")
        return True
    return False

def ensure_requirements_file(requirements_path):
    if not os.path.exists(requirements_path):
        print(f"Warning: '{requirements_path}' not found.")
        print("Creating a basic requirements.txt for WuBu core functionality...")
        basic_reqs = [
            "torch", "torchaudio", "TTS", "sounddevice", "soundfile", "numpy", "PyYAML",
            "requests", "tqdm", "ollama", "Flask", "python-dotenv", "rich", "prompt_toolkit",
            "transformers", "Pillow", "pyautogui", "pyperclip", "pycaw", "psutil",
            "PyGetWindow", "PyWinCtl", "pytesseract", "pandas", "scipy", "openai-whisper"
            # pyttsx3 removed as Coqui is primary
        ]
        # Sort and unique
        basic_reqs = sorted(list(set(basic_reqs)))
        with open(requirements_path, "w") as f:
            for req in basic_reqs:
                f.write(req + "\n")
        print(f"'{requirements_path}' created with basic dependencies for WuBu. Please review.")


def download_and_setup_models(): # Placeholder from original
    print("\n--- WuBu Model Setup (Placeholder) ---")
    wubu_glados_model_dir = os.path.join("src", "wubu", "tts", "glados_tts_models")
    print(f"For WuBu's GLaDOS-style voice (XTTSv2): Ensure custom model files are in: {os.path.abspath(wubu_glados_model_dir)}")
    wubu_kokoro_model_dir = os.path.join("src", "wubu", "tts", "kokoro_tts_models")
    print(f"\nFor WuBu's Kokoro voice: If local, place files in: {os.path.abspath(wubu_kokoro_model_dir)}")
    vosk_config_path = "asr_models/vosk-model-small-en-us-0.15"
    print(f"\nFor WuBu's ASR (Vosk): Ensure model files at path in config (e.g., {os.path.abspath(vosk_config_path)}).")
    print(f"\nFor WuBu's LLM (Ollama): Ensure Ollama is running and model pulled (e.g., 'ollama pull phi').")
    print("--- End WuBu Model Setup ---")

def main():
    print("--- WuBu Setup Script ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"WuBu project root identified as: {project_root}")
    os.chdir(project_root)

    check_python_version()

    venv_path = os.path.join(project_root, VENV_NAME)
    requirements_path = os.path.join(project_root, REQUIREMENTS_FILE)
    ensure_requirements_file(requirements_path) # Create if not exists

    use_uv = False
    if is_tool_available("uv"):
        print("'uv' executable found in PATH.")
        use_uv = True
    else:
        print("'uv' not found in PATH.")
        try_install_uv = input("Do you want to try installing 'uv' globally using pip? (y/N): ").lower().strip()
        if try_install_uv == 'y':
            print("Attempting to install 'uv' globally using system pip...")
            # Use sys.executable to ensure we're using the Python that ran this script
            if run_command([PYTHON_EXECUTABLE, "-m", "pip", "install", "uv"], error_message="Failed to install uv."):
                print("'uv' installed successfully.")
                use_uv = True
            else:
                print("'uv' installation failed. Falling back to standard venv/pip.")
        else:
            print("Skipping 'uv' installation. Using standard venv/pip.")

    venv_created = False
    if use_uv:
        # Pass the current python interpreter to uv to ensure venv matches
        if create_venv_with_uv(venv_path, PYTHON_EXECUTABLE):
            if install_deps_with_uv(venv_path, requirements_path):
                venv_created = True
            else:
                print("Failed to install dependencies with uv. You may need to activate and run manually.")
        else:
            print("Failed to create venv with uv. Try standard method or check uv installation.")

    if not venv_created: # Fallback to python -m venv and pip
        print("\nFalling back to standard 'python -m venv' and 'pip' for setup.")
        if create_venv_with_python(venv_path):
            python_in_venv = get_python_in_venv(venv_path)
            if install_deps_with_pip(python_in_venv, requirements_path):
                venv_created = True
            else:
                print("Failed to install dependencies with pip. Check errors above.")
        else:
            print("Failed to create virtual environment with python -m venv.")

    if not venv_created:
        print("\nCritical error: Virtual environment setup failed for WuBu. Exiting.")
        sys.exit(1)

    download_and_setup_models()

    print("\n--- WuBu Setup Complete ---")
    print(f"Virtual environment for WuBu is at: {venv_path}")
    if platform.system() == "Windows":
        activate_command = f".\\{VENV_NAME}\\Scripts\\activate.bat"
    else:
        activate_command = f"source {VENV_NAME}/bin/activate"
    print(f"To activate it, run in your terminal (from project root '{project_root}'):")
    print(f"  {activate_command}")
    print(f"Then you can run WuBu, e.g., python -m wubu.cli --config wubu_config.yaml")

if __name__ == "__main__":
    main()
