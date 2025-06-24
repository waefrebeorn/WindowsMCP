# WuBu: Your Desktop AI Assistant


**WuBu** is a Python-based AI assistant designed to understand your commands and interact with your Windows desktop environment. It leverages Large Language Models (LLMs like Google's Gemini or local models via Ollama) for natural language understanding and task execution. WuBu uses dedicated libraries for screen interaction, OCR, vision analysis, voice control, system interaction, and more.

## Features

*   **Desktop Control**: Programmatic control over mouse and keyboard actions.
*   **Screen Interaction**:
    *   Capture the full screen or specific regions.
    *   OCR (Optical Character Recognition) using Tesseract to find text and its coordinates on screen.
    *   Click on text found via OCR.
*   **Vision Analysis**: Utilizes Moondream v2 (or a similar vision model if configured) for general image description tasks.

*   **Window Management**: List open windows, get active window title, focus windows, get window geometry, and control active window (minimize, maximize, restore, close).
*   **Application Management**: Launch and close applications.
*   **System Monitoring**: Query system information like CPU usage, memory, disk space, and battery status.
*   **Web Interaction (Browser)**: Open URLs or perform web searches in the default browser.
*   **File System Tools**: Includes tools to list directory contents, read text files, and **perform write operations like creating files/folders, renaming, copying, and deleting. Exercise extreme caution with write operations.**
*   **Contextual Codebase Understanding**:
    *   **Codebase Indexing**: When WuBu starts, it indexes the files in the current working directory (Windows Search Index is used if available, falling back to manual scan). It builds a Merkle tree of file hashes to efficiently track the state of your project. This helps WuBu understand the broader context of your work.
    *   **Ignore Files**: Respects `.gitignore` and an additional `.wubuignore` file in the project root to exclude certain files/directories from indexing.
    *   **@-Mentions for Files**: You can refer to specific files in your commands using `@` notation (e.g., "summarize @src/main.py"). WuBu will load the content of the mentioned file to provide more relevant responses.
*   **Voice Interaction**:
    *   **Activation Phrases**: Activate WuBu by saying "WuBu", "Hey WuBu", "Yo WuBu", "WooBoo", or "WuhBoo" followed by your command when using voice input mode (`--voice`).
    *   Speech-to-Text (STT): Uses local Whisper models for accurate transcription.
    *   Text-to-Speech (TTS): Provides spoken responses using various engines:
        *   **Zonos TTS (Local)**: High-quality, voice cloning TTS now runs locally (recommended for Windows).
        *   Coqui XTTSv2 based: GLaDOS-style voice.
        *   Coqui TTS based: Kokoro (neutral voice).
        *   Basic system voices: via `pyttsx3` (if others are disabled).
*   **LLM Integration**: Supports Google Gemini (via API) and local LLMs (e.g., Llama, Phi, Qwen) through Ollama for natural language understanding and tool orchestration.
*   **Command-Line Interface**: Allows users to type commands or use voice to interact with their desktop.
*   **Configurable**: Settings for API keys, model names, and behavior can be managed through `wubu_config.yaml` and `.env` files.

## Prerequisites

*   **Python**: Version 3.9 or higher is recommended. Ensure Python is added to your system's PATH.
*   **Git**: For cloning the repository.
*   **Tesseract OCR**: Required for the `find_text_on_screen_and_click` tool and other precise text-location tasks.
    *   Installation instructions: [Tesseract OCR Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   Ensure Tesseract is added to your system's PATH. Alternatively, you can set the `TESSERACT_CMD_PATH` in your `wubu_config.yaml` (under a relevant section, e.g., `vision` or a new `ocr_settings`) to the full path of the Tesseract executable if it's not in your PATH or you need to specify a particular installation. (Note: Check `config_template.yaml` for exact structure if adding this).
*   **Ollama**: (Optional, if using local LLMs/Moondream) Install from [ollama.com](https://ollama.com).
*   **ffmpeg**: (Required by `openai-whisper`) A cross-platform solution to record, convert and stream audio and video.
    *   Linux: `sudo apt update && sudo apt install ffmpeg`
    *   macOS: `brew install ffmpeg`
    *   Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or via Chocolatey: `choco install ffmpeg`
*   **psutil**: (Required for System Monitoring tools) `pip install psutil` (this is handled by `install_uv_qinglong.ps1` via `requirements.txt`).
*   **For Local Zonos TTS (High-Quality Voice Cloning on Windows):**
    *   **Windows Operating System**: Zonos local TTS setup is currently focused on Windows.
    *   **PowerShell Execution Policy**: Needs to be set to `Unrestricted` to allow setup scripts to run. Open PowerShell as Administrator and run: `Set-ExecutionPolicy Unrestricted` (and choose 'A' for Yes to All).
    *   **MSVC (Visual Studio C++ Build Tools)**: Required for compiling some Python dependencies. Install Visual Studio 2022 (Community edition is fine) with the "Desktop development with C++" workload.
    *   **CUDA Toolkit 12.4**: If you intend to use Zonos TTS with GPU acceleration. Download and install from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64). Ensure it's correctly installed and environment variables (like `CUDA_PATH`) are set.
    *   **eSpeak NG**: A system-level dependency for `phonemizer` (used by Zonos). The `install_uv_qinglong.ps1` script will attempt to download and install `espeak-ng.msi` (version 1.52.0) automatically.

## Setup Instructions (Windows - Recommended using `install_uv_qinglong.ps1`)

The recommended way to set up WuBu on Windows, especially for using Zonos Local TTS, is via the `install_uv_qinglong.ps1` PowerShell script. This script automates several steps.

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory>   # Replace <repository_directory>
    ```

2.  **Windows Setup (Recommended)**:
    *   Ensure you have met the Windows-specific prerequisites listed above (especially PowerShell Execution Policy, MSVC, CUDA if using GPU).
    *   Run `setup_venv.bat` **as Administrator**.
        *   You can do this by right-clicking `setup_venv.bat` and choosing "Run as administrator".
    *   This batch script will then execute the `install_uv_qinglong.ps1` PowerShell script.
    *   The PowerShell script (`install_uv_qinglong.ps1`) automates:
        *   Installation of `uv` (a fast Python package manager).
        *   Creation of a Python virtual environment (`.venv`).
        *   System-wide installation of `espeak-ng` (for Zonos TTS).
        *   Installation of all Python dependencies from `requirements.txt` using `uv`, including PyTorch with the correct CUDA version for Zonos.
    *   Monitor the script output in the console window for any errors or prompts.

3.  **Manual/Cross-Platform Setup (Alternative)**:
    *   Ensure all system prerequisites listed above are met (Python, Git, Tesseract, ffmpeg, and Zonos-specific ones if using Zonos TTS).
    *   Create and activate a Python virtual environment:
        ```bash
        python -m venv .venv
        # Activate:
        # Windows: .\.venv\Scripts\activate
        # macOS/Linux: source .venv/bin/activate
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        # For PyTorch with specific CUDA on Windows (e.g., CUDA 12.4 for Zonos):
        # pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
        ```

4.  **Configuration (`wubu_config.yaml`)**:
    *   If `wubu_config.yaml` does not exist in the project root, copy `src/wubu/config_template.yaml` to `wubu_config.yaml`.
        ```bash
        # Example: cp src/wubu/config_template.yaml wubu_config.yaml (use 'copy' on Windows cmd)
        copy src\wubu\config_template.yaml wubu_config.yaml
        ```
    *   **Edit `wubu_config.yaml`**: Review and update the settings. Key sections:
        *   `llm`: Configure your LLM provider (e.g., `ollama_settings.model`).
        *   `tts`: Configure Text-to-Speech. For Zonos Local TTS:
            ```yaml
            tts:
              default_voice: "zonos_engine_local_cloning" # Makes Zonos local the default
              estimated_max_speech_duration: 7

              zonos_local_engine:
                enabled: true
                language: 'en' # Default: "en", supports "ja", "zh", "fr", "de"
                model_id: "Zyphra/Zonos-v0.1-transformer" # Model to load
                device: "cpu" # Change to "cuda" for GPU. Ensure CUDA 12.4 is set up.
                default_reference_audio_path: "" # Optional: path to a .wav for default cloning
                # unconditional_keys: ["emotion"] # Example: Zonos default

              # Disable other TTS engines if Zonos is the primary one
              wubu_glados_style_voice:
                enabled: false
              wubu_kokoro_voice:
                enabled: false
            ```
        *   Other settings as needed (API keys in `.env`, paths, etc.). Refer to comments in `wubu_config.yaml` or `src/wubu/config_template.yaml`.
    *   **API Keys (e.g., Gemini)**:
        If using cloud services like Gemini, create a `.env` file in the project root:
        ```env
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   **Ollama Setup (if using Ollama)**:
        (As before...)
    *   **Microphone Access**: Ensure the application has permission to access your microphone for voice input.

## Running the Assistant

1.  **Activate your virtual environment**.
2.  **Run `main.py` using one of the provided batch scripts (e.g., `run_wubu.bat`) or directly**:
    ```bash
    python main.py
    ```
    **Command-Line Arguments**:
    *   `--voice`: Enable voice input mode (uses Whisper).
    *   `--llm_provider {gemini,ollama}`: Override LLM provider settings from `wubu_config.yaml`.
    *   `--ollama_model <model_name>`: (If Ollama) Override the Ollama model specified in `wubu_config.yaml`.
    *   `--test_command "<command>"`: Execute a single command.
    *   `--test_file <filepath>`: Execute commands from a file.

### Interactive Mode
Type commands or use voice (if `--voice` flag is used).

**Voice Interaction:**
When using the `--voice` flag, activate WuBu by saying one of the activation phrases: "WuBu", "Hey WuBu", "Yo WuBu", "WooBoo", or "WuhBoo". Follow with your command. If you only say the activation phrase, WuBu will prompt you to type your command.

**Contextual Understanding with @-Mentions:**
WuBu indexes files in its startup directory (current working directory). Use `@` to refer to files (e.g., "summarize @src/app.py"). The mentioned file becomes the "current file" for subsequent related commands.

**Note on Project Context:** Run WuBu from the root of the project you want it to be aware of. A `.wubuignore` file (similar to `.gitignore`) can exclude files/directories from indexing.

### Available Tools (Examples)

WuBu has access to a variety of tools to interact with your system:

*   **Screen & Vision**:
    *   `capture_screen_region`, `capture_full_screen`, `get_screen_resolution`
    *   `analyze_image_with_vision_model` (e.g., with Moondream for OCR or image description)
    *   `find_text_on_screen_and_click` (combines OCR and mouse click)
*   **Mouse & Keyboard**:
    *   `mouse_move`, `mouse_click`, `mouse_drag`, `mouse_scroll`
    *   `keyboard_type`, `keyboard_press_key`, `keyboard_hotkey`
*   **Window Management**:
    *   `list_windows`: Lists titles of open windows (can be filtered).
    *   `control_active_window`: Actions like "minimize", "maximize", "restore", "close", "get_title", "get_geometry" on the active window.
    *   `focus_window`: Brings a window to the foreground by its title.
*   **Application Management**:
    *   `start_application`: Launches an application by name or path (e.g., "notepad", "chrome.exe").
    *   `get_running_processes`: Lists currently running processes.
    *   `close_application_by_pid`, `close_application_by_title`: Terminates applications.
*   **System Information & Control**:
    *   `get_system_information`: Queries "cpu_usage", "memory_usage", "disk_usage" (optional path), "battery_status".
    *   `get_clipboard_text`, `set_clipboard_text`
    *   `get_system_volume`, `set_system_volume` (Windows-only)
    *   `lock_windows_session` (Windows-only)
    *   `shutdown_windows_system` (mode: "shutdown", "restart", "logoff")
*   **Web Interaction**:
    *   `open_url_or_search_web`: Opens a URL directly or performs a web search (e.g., for "Python tutorials") in the default browser.
*   **File System Tools**:
    *   `list_directory`: Lists contents of a directory.
    *   `read_text_file`: Reads content from a text file.
    *   `get_file_properties`: Gets metadata about a file or folder.
    *   **WARNING: The following file system tools modify files and directories. Use them with EXTREME CAUTION and ensure WuBu correctly understands your intent, especially with paths and destructive operations.**
        *   `create_folder`: Creates a new folder.
        *   `write_text_file`: Writes text to a file (can overwrite).
        *   `append_text_to_file`: Appends text to a file.
        *   `move_or_rename_item`: Moves/renames files or folders.
        *   `copy_item`: Copies files or folders.
        *   `delete_item`: Deletes files or folders. **This is highly destructive, especially with non-empty folders if forced.**

### Example Workflow

1.  **User (Voice)**: *"Hey WuBu, launch Notepad."*
2.  **WuBu**: Calls `start_application` tool with `application_name_or_path="notepad.exe"`. (Notepad opens)
3.  **User (Text)**: *"What's the CPU usage?"*
4.  **WuBu**: Calls `get_system_information` with `query="cpu_usage"`.
5.  **WuBu (Response)**: "Current CPU usage is 15.7%."
6.  **User (Text)**: *"Summarize the main points in @project_docs/feature_spec.md"*
7.  **WuBu (LLM Decision with Context)**: WuBu loads `project_docs/feature_spec.md`. The LLM analyzes the file content.
8.  **WuBu (Response)**: "The document outlines features X, Y, and Z..."
9.  **User (Voice)**: *"WuBu, create a folder named 'old_specs' in @project_docs."*
10. **WuBu (after confirming intent, if it were that cautious)**: Calls `create_folder` with `path="project_docs/old_specs"`.
11. **WuBu (Response)**: "Folder 'project_docs/old_specs' created."
12. **User (Text)**: *"Minimize this window."* (referring to the terminal/console WuBu is in)
13. **WuBu**: Calls `control_active_window` with `action="minimize"`.

---
*Various scripts are provided in the repository for common tasks:*
*   **`install_uv_qinglong.ps1` (Recommended for Windows)**: Automates comprehensive setup including `uv`, Python virtual environment (`.venv`), `espeak-ng`, and all Python dependencies from `requirements.txt` with correct CUDA support for PyTorch. Use this for the full local Zonos TTS experience on Windows.
*   **`setup_venv.bat` / `setup_python_env.bat`**: Basic Python virtual environment (`venv`) creation and `pip install -r requirements.txt`. Does not handle `espeak-ng` or specific PyTorch CUDA versions needed for Zonos GPU. Suitable for basic setup or non-Windows environments where the PowerShell script is not applicable.
*   **`ollama_setup.bat`**: Assists with Ollama setup if you're using local LLMs.
*   **`run_wubu.bat`, `run_ollama_wubu.bat`**: Convenience scripts for running WuBu.
*Always review scripts before running, especially if they perform installations or modifications.*

